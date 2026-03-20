"""
Standalone aggregation script for RUSH Gait Analysis Pipeline.

Reads bout-level, window-level, and daily PA CSVs from a pipeline output directory
and computes subject-level summary statistics.

This is separated from the main pipeline so you can:
  - Run GPU-heavy feature extraction once
  - Iterate on summary statistics without re-processing
  - Add new metrics without re-running models

Usage:
    python aggregate_subjects.py                           # Use default config
    python aggregate_subjects.py --output-dir /path/to/output
    python aggregate_subjects.py --config custom_config.yaml

Output:
    {output_dir}/subject_summary.csv  — one row per subject with all summary statistics
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, median_abs_deviation, entropy, wasserstein_distance, norm

# Optional advanced statistics packages
try:
    import lmoments3 as lm
    HAS_LMOMENTS = True
except ImportError:
    HAS_LMOMENTS = False
    logging.warning("lmoments3 not available - L-skewness/L-kurtosis will be NaN")

try:
    import diptest
    HAS_DIPTEST = True
except ImportError:
    HAS_DIPTEST = False
    logging.warning("diptest not available - Hartigan's Dip statistic will be NaN")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Global histogram bin ranges (ensures cross-subject comparability)
# ============================================================================

GLOBAL_RANGES = {
    'duration_sec':         {'min': 10,  'max': 3200},
    'speed':                {'min': 0,   'max': 1.8},
    'cadence':              {'min': 40,  'max': 160},
    'gait_length':          {'min': 0,   'max': 2},
    'gait_length_indirect': {'min': 0,   'max': 2},
    'regularity_eldernet':  {'min': 0,   'max': 1},
    'regularity_sp':        {'min': 0,   'max': 1},
    'entropy':              {'min': 0,   'max': 3},
    'dom_freq':             {'min': 0,   'max': 15},
    'psd_amp':              {'min': 0,   'max': 2.5},
    'psd_width':            {'min': 0,   'max': 1.0},
    'psd_slope':            {'min': 0,   'max': 300},
    'bout_pa_mean':         {'min': 0.8, 'max': 2.0},
}

N_BINS = 10

# Metrics for which to compute within-bout variability (var-of-var)
VAR_OF_VAR_METRICS = ['speed', 'cadence', 'gait_length', 'regularity_eldernet']

# Core metrics for distribution shape and stability analysis
CORE_METRICS = ['speed', 'cadence', 'gait_length', 'regularity_eldernet']

# Minimum sample size for robust statistics
MIN_SAMPLE_SIZE = 30

# Bout-level metrics to compute summary statistics on
BOUT_METRICS = [
    'duration_sec', 'speed', 'cadence', 'gait_length', 'gait_length_indirect',
    'regularity_eldernet', 'regularity_sp', 'entropy',
    'dom_freq', 'psd_amp', 'psd_width', 'psd_slope',
    'bout_pa_mean', 'bout_pa_std', 'total_steps',
]


# ============================================================================
# Core aggregation function
# ============================================================================

def aggregate_subject(
    bout_df: pd.DataFrame,
    window_df: pd.DataFrame,
    daily_pa: dict,
    subject_id: str,
    num_days: int,
) -> dict:
    """
    Compute all summary statistics for one subject.

    Args:
        bout_df: DataFrame with bout-level features.
        window_df: DataFrame with window-level features.
        daily_pa: dict with keys 'daily_pa_mean', 'daily_pa_std', 'tdpa' (arrays).
        subject_id: Subject identifier.
        num_days: Number of valid recording days.

    Returns:
        dict with all summary statistics (one row of final output).
    """
    row = {'sub_id': subject_id, 'wear_days': num_days}

    # --- Daily walking minutes ---
    row.update(_daily_walking_stats(bout_df, 'daily_walking_'))

    # --- Daily step count ---
    row.update(_daily_step_stats(bout_df, 'daily_step_count_'))

    # --- Bout-level statistics (including robust metrics: MAD, L-skewness, L-kurtosis) ---
    for metric in BOUT_METRICS:
        prefix = f'bout_{metric}_'
        if metric in bout_df.columns:
            data = bout_df[metric].dropna().values
            row.update(_calc_stats(data, prefix))
            row.update(_histogram_features(data, prefix, metric))
        else:
            row.update(_calc_stats(np.array([]), prefix))

    # --- Distribution shape metrics (Dip, Entropy, Wasserstein to Normal) ---
    row.update(_calc_distribution_shape(bout_df))

    # --- Between-day stability (distributional drift) ---
    row.update(_calc_between_day_stability(bout_df))

    # --- Daily bout volume statistics ---
    row.update(_daily_bout_volume_stats(bout_df))

    # --- Time of day statistics ---
    row.update(_time_of_day_stats(bout_df))

    # --- Variance-of-variance (within-bout variability) ---
    row.update(_var_of_var(window_df))

    # --- Daily physical activity ---
    for key in ['daily_pa_mean', 'daily_pa_std', 'tdpa']:
        data = daily_pa.get(key, np.array([]))
        if isinstance(data, np.ndarray) and data.size > 0:
            row.update(_calc_stats(data, f'{key}_'))
        else:
            row.update(_calc_stats(np.array([]), f'{key}_'))

    # --- Bout count ---
    row['n_bouts'] = len(bout_df)

    return row


# ============================================================================
# Statistics helpers
# ============================================================================

def _calc_stats(data: np.ndarray, prefix: str) -> dict:
    """
    Compute descriptive statistics for an array.

    Includes robust scale and shape metrics:
    - MAD (Median Absolute Deviation)
    - L-skewness (Tau-3) via L-moments
    - L-kurtosis (Tau-4) via L-moments
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)
    data = data.flatten()
    data = data[np.isfinite(data)]

    # stat_names = ['median', 'mean', 'std', 'p5', 'p10', 'p90', 'p95',
    #               'kurtosis', 'skewness', 'range', 'iqr', 'cv',
    #               'mad', 'l_skewness', 'l_kurtosis']
    stat_names = ['median', 'mean', 'std', 'p10', 'p90',
                  'kurtosis', 'skewness', 'range', 'iqr', 'cv',
                  'mad', 'l_skewness', 'l_kurtosis']

    nan_dict = {f'{prefix}{s}': np.nan for s in stat_names}

    if len(data) == 0:
        return nan_dict

    mean_val = np.mean(data)
    n = len(data)

    # Compute MAD (Median Absolute Deviation)
    mad_val = float(median_abs_deviation(data, nan_policy='omit')) if n >= MIN_SAMPLE_SIZE else np.nan

    # Compute L-moments (L-skewness and L-kurtosis)
    l_skew = np.nan
    l_kurt = np.nan
    if HAS_LMOMENTS and n >= MIN_SAMPLE_SIZE:
        try:
            # lmoments3.lmom_ratios returns [l1, l2, t3, t4, ...] where t3=L-skewness, t4=L-kurtosis
            lmom_ratios = lm.lmom_ratios(data, nmom=4)
            if len(lmom_ratios) >= 4:
                l_skew = float(lmom_ratios[2]) if np.isfinite(lmom_ratios[2]) else np.nan
                l_kurt = float(lmom_ratios[3]) if np.isfinite(lmom_ratios[3]) else np.nan
        except Exception:
            pass  # Keep NaN on any computation error

    return {
        f'{prefix}median': float(np.median(data)),
        f'{prefix}mean': float(mean_val),
        f'{prefix}std': float(np.std(data)),
        # f'{prefix}p5': float(np.percentile(data, 5)),
        f'{prefix}p10': float(np.percentile(data, 10)),
        f'{prefix}p90': float(np.percentile(data, 90)),
        # f'{prefix}p95': float(np.percentile(data, 95)),
        f'{prefix}kurtosis': float(kurtosis(data)) if n > 3 else np.nan,
        f'{prefix}skewness': float(skew(data)) if n > 3 else np.nan,
        f'{prefix}range': float(np.ptp(data)),
        f'{prefix}iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        f'{prefix}cv': float(np.std(data) / mean_val) if mean_val != 0 else np.nan,
        f'{prefix}mad': mad_val,
        f'{prefix}l_skewness': l_skew,
        f'{prefix}l_kurtosis': l_kurt,
    }


def _histogram_features(data: np.ndarray, prefix: str, metric_name: str,
                        n_bins: int = N_BINS) -> dict:
    """Compute histogram frequency and probability features with global bins."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    result = {}
    if len(data) == 0:
        for i in range(n_bins):
            result[f'{prefix}freq_bin{i}'] = 0
            result[f'{prefix}prob_bin{i}'] = 0.0
        return result

    # Use global range if available, else data range
    if metric_name in GLOBAL_RANGES:
        lo = GLOBAL_RANGES[metric_name]['min']
        hi = GLOBAL_RANGES[metric_name]['max']
    else:
        lo, hi = float(np.min(data)), float(np.max(data))
        if lo == hi:
            hi = lo + 1

    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(data, bins=bins)
    total = counts.sum()
    probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    for i in range(n_bins):
        result[f'{prefix}freq_bin{i}'] = int(counts[i])
        result[f'{prefix}prob_bin{i}'] = float(probs[i])

    return result


def _var_of_var(window_df: pd.DataFrame) -> dict:
    """
    Compute variance-of-variance: the spread of within-bout gait variability.

    For each metric, compute the std within each bout (from window data),
    then report the median and std of those per-bout stds across all bouts.
    """
    result = {}

    if window_df.empty:
        for metric in VAR_OF_VAR_METRICS:
            result[f'var_var_{metric}_median'] = np.nan
            result[f'var_var_{metric}_std'] = np.nan
        return result

    for metric in VAR_OF_VAR_METRICS:
        if metric not in window_df.columns:
            result[f'var_var_{metric}_median'] = np.nan
            result[f'var_var_{metric}_std'] = np.nan
            continue

        # Group windows by bout, compute std within each bout
        bout_stds = []
        for bout_id, group in window_df.groupby('bout_id'):
            vals = group[metric].dropna().values
            if len(vals) >= 5:  # Need enough windows for meaningful std
                bout_stds.append(float(np.std(vals)))

        if bout_stds:
            result[f'var_var_{metric}_median'] = float(np.median(bout_stds))
            result[f'var_var_{metric}_std'] = float(np.std(bout_stds))
        else:
            result[f'var_var_{metric}_median'] = np.nan
            result[f'var_var_{metric}_std'] = np.nan

    return result


def _daily_walking_stats(bout_df: pd.DataFrame, prefix: str) -> dict:
    """Compute daily walking minutes statistics."""
    if bout_df.empty or 'start_time' not in bout_df.columns:
        return _calc_stats(np.array([]), prefix)

    try:
        df = bout_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['day'] = df['start_time'].dt.date
        daily_walking_min = df.groupby('day')['duration_sec'].sum() / 60
        return _calc_stats(daily_walking_min.values, prefix)
    except Exception:
        return _calc_stats(np.array([]), prefix)


def _daily_step_stats(bout_df: pd.DataFrame, prefix: str) -> dict:
    """Compute daily step count statistics."""
    if bout_df.empty or 'start_time' not in bout_df.columns:
        return _calc_stats(np.array([]), prefix)

    try:
        df = bout_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['day'] = df['start_time'].dt.date
        daily_steps = df.groupby('day')['total_steps'].sum()
        return _calc_stats(daily_steps.values, prefix)
    except Exception:
        return _calc_stats(np.array([]), prefix)


# ============================================================================
# Distribution Shape Metrics (Modality & Complexity)
# ============================================================================

def _calc_distribution_shape(bout_df: pd.DataFrame) -> dict:
    """
    Compute distribution shape metrics for core gait measures.

    For each metric in CORE_METRICS, calculates:
    - Hartigan's Dip Statistic: Measures multimodality (higher = more multimodal)
    - Shannon Entropy: Measures distributional complexity/uniformity
    - Wasserstein Distance to Gaussian: Measures deviation from normality

    Args:
        bout_df: DataFrame with bout-level features.

    Returns:
        dict with distribution shape metrics for each core metric.
    """
    result = {}

    for metric in CORE_METRICS:
        prefix = f'dist_{metric}_'

        # Initialize NaN outputs
        result[f'{prefix}dip_stat'] = np.nan
        result[f'{prefix}dip_pvalue'] = np.nan
        result[f'{prefix}shannon_entropy'] = np.nan
        result[f'{prefix}wasserstein_to_normal'] = np.nan

        if metric not in bout_df.columns:
            continue

        data = bout_df[metric].dropna().values
        if len(data) < MIN_SAMPLE_SIZE:
            continue

        # --- Hartigan's Dip Statistic ---
        if HAS_DIPTEST:
            try:
                dip_stat, dip_pval = diptest.diptest(data)
                result[f'{prefix}dip_stat'] = float(dip_stat)
                result[f'{prefix}dip_pvalue'] = float(dip_pval)
            except Exception:
                pass

        # --- Shannon Entropy (histogram-based) ---
        try:
            # Use global range if available for consistent binning
            if metric in GLOBAL_RANGES:
                lo = GLOBAL_RANGES[metric]['min']
                hi = GLOBAL_RANGES[metric]['max']
            else:
                lo, hi = float(np.min(data)), float(np.max(data))
                if lo == hi:
                    hi = lo + 1

            bins = np.linspace(lo, hi, N_BINS + 1)
            counts, _ = np.histogram(data, bins=bins)
            # Add small constant to avoid log(0)
            probs = counts / counts.sum()
            # Use scipy.stats.entropy which handles zeros gracefully
            result[f'{prefix}shannon_entropy'] = float(entropy(probs + 1e-10))
        except Exception:
            pass

        # --- Wasserstein Distance to Gaussian ---
        try:
            # Fit normal distribution to data
            mu, sigma = norm.fit(data)
            if sigma > 0:
                # Generate theoretical normal sample with same size
                np.random.seed(42)  # Reproducibility
                normal_sample = norm.rvs(loc=mu, scale=sigma, size=len(data))
                # Compute Wasserstein (Earth Mover's) distance
                w_dist = wasserstein_distance(data, normal_sample)
                result[f'{prefix}wasserstein_to_normal'] = float(w_dist)
        except Exception:
            pass

    return result


# ============================================================================
# Between-Day Stability (Distributional Drift)
# ============================================================================

def _calc_between_day_stability(bout_df: pd.DataFrame) -> dict:
    """
    Compute day-to-day stability of gait distributions.

    For each core metric, computes pairwise Wasserstein distances between
    daily distributions and returns the mean pairwise distance as a measure
    of global stability/drift.

    Args:
        bout_df: DataFrame with bout-level features including 'start_time'.

    Returns:
        dict with between-day stability metrics for each core metric.
    """
    result = {}

    # Initialize NaN outputs
    for metric in CORE_METRICS:
        result[f'stability_{metric}_mean_wasserstein'] = np.nan
        result[f'stability_{metric}_std_wasserstein'] = np.nan
        result[f'stability_{metric}_n_day_pairs'] = 0

    if bout_df.empty or 'start_time' not in bout_df.columns:
        return result

    try:
        df = bout_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['day'] = df['start_time'].dt.date
    except Exception:
        return result

    # Get unique days with sufficient data
    days = df['day'].unique()
    if len(days) < 2:
        return result

    for metric in CORE_METRICS:
        if metric not in df.columns:
            continue

        # Build per-day distributions
        daily_data = {}
        for day in days:
            day_vals = df[df['day'] == day][metric].dropna().values
            # Require minimum sample size per day
            if len(day_vals) >= MIN_SAMPLE_SIZE:
                daily_data[day] = day_vals

        if len(daily_data) < 2:
            continue

        # Compute pairwise Wasserstein distances
        day_list = list(daily_data.keys())
        pairwise_distances = []

        for day_i, day_j in combinations(day_list, 2):
            try:
                w_dist = wasserstein_distance(daily_data[day_i], daily_data[day_j])
                pairwise_distances.append(w_dist)
            except Exception:
                continue

        if pairwise_distances:
            result[f'stability_{metric}_mean_wasserstein'] = float(np.mean(pairwise_distances))
            result[f'stability_{metric}_std_wasserstein'] = float(np.std(pairwise_distances))
            result[f'stability_{metric}_n_day_pairs'] = len(pairwise_distances)

    return result


# ============================================================================
# Daily Bout Volume Metrics
# ============================================================================

def _daily_bout_volume_stats(bout_df: pd.DataFrame) -> dict:
    """
    Compute daily bout volume statistics.

    Extracts per-day:
    - Number of walking bouts
    - Number of bouts > 30 seconds
    - Number of bouts > 60 seconds

    Then reports descriptive stats (mean, median, etc.) across wear days.

    Args:
        bout_df: DataFrame with bout-level features.

    Returns:
        dict with bout volume statistics.
    """
    result = {}

    # Initialize empty stats for all prefixes
    for prefix in ['daily_n_bouts_', 'daily_n_bouts_gt30s_', 'daily_n_bouts_gt60s_']:
        result.update(_calc_stats(np.array([]), prefix))

    if bout_df.empty or 'start_time' not in bout_df.columns:
        return result

    try:
        df = bout_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['day'] = df['start_time'].dt.date

        # Total bouts per day
        daily_n_bouts = df.groupby('day').size().values
        result.update(_calc_stats(daily_n_bouts, 'daily_n_bouts_'))

        # Bouts > 30 seconds per day
        if 'duration_sec' in df.columns:
            df_gt30 = df[df['duration_sec'] > 30]
            if not df_gt30.empty:
                daily_n_gt30 = df_gt30.groupby('day').size().reindex(
                    df['day'].unique(), fill_value=0
                ).values
            else:
                daily_n_gt30 = np.zeros(len(df['day'].unique()))
            result.update(_calc_stats(daily_n_gt30, 'daily_n_bouts_gt30s_'))

            # Bouts > 60 seconds per day
            df_gt60 = df[df['duration_sec'] > 60]
            if not df_gt60.empty:
                daily_n_gt60 = df_gt60.groupby('day').size().reindex(
                    df['day'].unique(), fill_value=0
                ).values
            else:
                daily_n_gt60 = np.zeros(len(df['day'].unique()))
            result.update(_calc_stats(daily_n_gt60, 'daily_n_bouts_gt60s_'))

    except Exception:
        pass

    return result


# ============================================================================
# Time of Day Statistics
# ============================================================================

def _time_of_day_stats(bout_df: pd.DataFrame) -> dict:
    """
    Compute time-of-day distribution of walking bouts.

    Categorizes bouts into:
    - Morning: 06:00 - 11:59
    - Afternoon: 12:00 - 16:59
    - Evening: 17:00 - 23:59

    Returns proportions and counts for each period.

    Args:
        bout_df: DataFrame with bout-level features including 'start_time'.

    Returns:
        dict with time-of-day statistics.
    """
    result = {
        'tod_morning_proportion': np.nan,
        'tod_afternoon_proportion': np.nan,
        'tod_evening_proportion': np.nan,
        'tod_night_proportion': np.nan,  # 00:00 - 05:59
        'tod_morning_n_bouts': 0,
        'tod_afternoon_n_bouts': 0,
        'tod_evening_n_bouts': 0,
        'tod_night_n_bouts': 0,
    }

    if bout_df.empty or 'start_time' not in bout_df.columns:
        return result

    try:
        df = bout_df.copy()
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['hour'] = df['start_time'].dt.hour

        total_bouts = len(df)
        if total_bouts == 0:
            return result

        # Categorize by time of day
        morning = df[(df['hour'] >= 6) & (df['hour'] < 12)]
        afternoon = df[(df['hour'] >= 12) & (df['hour'] < 17)]
        evening = df[(df['hour'] >= 17) & (df['hour'] < 24)]
        night = df[(df['hour'] >= 0) & (df['hour'] < 6)]

        result['tod_morning_n_bouts'] = len(morning)
        result['tod_afternoon_n_bouts'] = len(afternoon)
        result['tod_evening_n_bouts'] = len(evening)
        result['tod_night_n_bouts'] = len(night)

        result['tod_morning_proportion'] = float(len(morning) / total_bouts)
        result['tod_afternoon_proportion'] = float(len(afternoon) / total_bouts)
        result['tod_evening_proportion'] = float(len(evening) / total_bouts)
        result['tod_night_proportion'] = float(len(night) / total_bouts)

    except Exception:
        pass

    return result


# ============================================================================
# Batch aggregation from saved CSVs
# ============================================================================

def concatenate_bouts(output_dir: str, output_file: str = None) -> pd.DataFrame:
    """
    Concatenate all bouts CSVs across devices into one DataFrame.

    Expects the output directory to contain device subdirectories (e.g. geneactive/, axivity/),
    each with a bouts/ folder inside:

        output_dir/
            geneactive/
                bouts/
                windows/
                daily_pa/
            axivity/
                bouts/
                windows/
                daily_pa/

    The device name is taken from the subdirectory name.

    Args:
        output_dir: Parent directory containing device subdirectories.
        output_file: Optional path to save the concatenated CSV.

    Returns:
        DataFrame with all bouts, plus 'subject_id' and 'device' columns.
    """
    output_dir = Path(output_dir)

    # Find all subdirectories that contain a bouts/ folder
    device_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir() and (d / 'bouts').exists()]

    if not device_dirs:
        raise FileNotFoundError(
            f"No device subdirectories with a bouts/ folder found in {output_dir}"
        )

    all_dfs = []
    for device_dir in device_dirs:
        device = device_dir.name
        bout_files = sorted((device_dir / 'bouts').glob('*.csv'))
        logger.info(f"Device '{device}': found {len(bout_files)} bouts files")

        for bout_file in bout_files:
            df = pd.read_csv(bout_file)
            df.insert(0, 'subject_id', bout_file.stem)
            df.insert(1, 'device', device)
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No bouts files found.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if output_file:
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved concatenated bouts to {output_file}")

    return combined


def aggregate_from_directory(output_dir: str) -> pd.DataFrame:
    """
    Aggregate all subjects from a pipeline output directory.

    Args:
        output_dir: Path containing bouts/, windows/, daily_pa/ subdirectories.

    Returns:
        DataFrame with one row per subject.
    """
    output_dir = Path(output_dir)
    bout_dir = output_dir / 'bouts'
    window_dir = output_dir / 'windows'
    daily_pa_dir = output_dir / 'daily_pa'

    if not bout_dir.exists():
        raise FileNotFoundError(f"Bout directory not found: {bout_dir}")

    results = []
    bout_files = sorted(bout_dir.glob('*.csv'))

    logger.info(f"Found {len(bout_files)} subjects to aggregate")

    for bout_file in bout_files:
        subject_id = bout_file.stem

        # Load bout data
        bout_df = pd.read_csv(bout_file)

        # Load window data
        window_file = window_dir / f'{subject_id}.csv'
        window_df = pd.read_csv(window_file) if window_file.exists() else pd.DataFrame()

        # Load daily PA
        daily_pa_file = daily_pa_dir / f'{subject_id}.csv'
        daily_pa = {}
        num_days = 0
        if daily_pa_file.exists():
            pa_df = pd.read_csv(daily_pa_file)
            for col in pa_df.columns:
                if col != 'day':
                    daily_pa[col] = pa_df[col].values
            num_days = len(pa_df)

        row = aggregate_subject(
            bout_df=bout_df,
            window_df=window_df,
            daily_pa=daily_pa,
            subject_id=subject_id,
            num_days=num_days,
        )
        results.append(row)

    return pd.DataFrame(results)


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate bout/window features into subject-level summary statistics"
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Path to pipeline output directory (contains bouts/, windows/, daily_pa/)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML to read output_path from'
    )
    parser.add_argument(
        '--out-file', type=str, default='subject_summary.csv',
        help='Output filename (default: subject_summary.csv)'
    )
    parser.add_argument(
        '--concatenate-bouts', action='store_true',
        help='Concatenate all bouts CSVs across devices into one file instead of aggregating'
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.config:
        from config import load_config
        cfg = load_config(args.config)
        output_dir = Path(cfg.data.output_path)
    else:
        # Try default config
        try:
            from config import load_config
            cfg = load_config(None)
            output_dir = Path(cfg.data.output_path)
        except Exception:
            parser.error("Must specify --output-dir or --config")

    if args.concatenate_bouts:
        out_path = output_dir / args.out_file
        logger.info(f"Concatenating bouts from: {output_dir}")
        df = concatenate_bouts(str(output_dir), output_file=str(out_path))
        print(f"\nConcatenation complete:")
        print(f"  Rows: {len(df)}")
        print(f"  Devices: {df['device'].unique().tolist()}")
        print(f"  Output: {out_path}")
        return

    logger.info(f"Aggregating from: {output_dir}")

    # Run aggregation
    df = aggregate_from_directory(str(output_dir))

    # Save
    out_path = output_dir / args.out_file
    df.to_csv(out_path, index=False)
    logger.info(f"Done. {len(df)} subjects -> {out_path}")

    # Print summary
    print(f"\nAggregation complete:")
    print(f"  Subjects: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Output: {out_path}")


if __name__ == '__main__':
    main()
