"""
Unified signal preprocessing for RUSH pipeline.

Replaces the 3 separate preprocessing implementations that previously existed
in rush_pipeline, calc_freq_all, and calc_entropy_all (which produced
inconsistent results due to different imputation strategies).

Pipeline: raw .mat → actipy calibration + nonwear → resample → impute → trim
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from actipy.reader import process as actipy_process

logger = logging.getLogger(__name__)


def preprocess_subject(acc: np.ndarray, fs: float, start_time_str: str,
                       target_fs: int = 30,
                       drop_first_last: bool = True,
                       min_wear_hours: float = 72.0,
                       require_all_hours: bool = True) -> Optional[pd.DataFrame]:
    """
    Full preprocessing pipeline for a single subject.

    Args:
        acc: Raw acceleration (N, 3).
        fs: Original sampling frequency.
        start_time_str: Recording start time as string.
        target_fs: Target resampled frequency.
        drop_first_last: Whether to drop first and last days (partial days).
        min_wear_hours: Minimum hours of valid wear data required (default 72 = 3 days).
        require_all_hours: If True, require wear data in each hour of the 24-hour cycle.

    Returns:
        DataFrame with columns ['x', 'y', 'z'] and DatetimeIndex, or None.
    """
    try:
        # Build datetime-indexed DataFrame
        df_raw = _acc_to_df(acc, fs, start_time_str)

        # actipy: calibrate gravity, detect nonwear, resample
        df_proc, info = actipy_process(
            df_raw,
            sample_rate=fs,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=target_fs,
            verbose=False
        )

        # Validate wear time BEFORE imputation (when we can still see missing data)
        is_valid, wear_info = validate_wear_time(
            df_proc, target_fs, min_wear_hours, require_all_hours
        )
        if not is_valid:
            logger.warning(f"Wear time validation failed: {wear_info}")
            return None

        # Impute missing segments (time-of-day matching)
        df_imputed = _impute_missing(df_proc, target_fs=target_fs)

        # Drop partial first/last days
        if drop_first_last:
            df_imputed = _drop_edge_days(df_imputed, which='both')

        if len(df_imputed) == 0:
            logger.warning("No data remaining after preprocessing")
            return None

        return df_imputed

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None


def validate_wear_time(df: pd.DataFrame, target_fs: int,
                       min_wear_hours: float = 72.0,
                       require_all_hours: bool = True) -> tuple[bool, dict]:
    """
    Validate that the subject has sufficient wear time.

    Criteria:
    1. At least min_wear_hours (default 72) hours of valid (non-NaN) data
    2. If require_all_hours=True, must have wear data in each one-hour period
       of the 24-hour cycle (i.e., at least one sample from hours 0-23)

    Args:
        df: DataFrame with acceleration data (may contain NaN for nonwear).
        target_fs: Sampling frequency.
        min_wear_hours: Minimum required hours of wear data.
        require_all_hours: Whether to require coverage of all 24 hours.

    Returns:
        Tuple of (is_valid, info_dict with details).
    """
    # Count valid (non-NaN) samples
    valid_mask = ~df['x'].isna()
    valid_samples = valid_mask.sum()
    total_wear_hours = valid_samples / (target_fs * 3600)

    info = {
        'total_wear_hours': round(total_wear_hours, 2),
        'min_required_hours': min_wear_hours,
    }

    # Check minimum wear hours
    if total_wear_hours < min_wear_hours:
        info['reason'] = f"Insufficient wear time: {total_wear_hours:.1f}h < {min_wear_hours}h"
        return False, info

    # Check 24-hour coverage
    if require_all_hours:
        valid_df = df[valid_mask]
        hours_with_data = valid_df.index.hour.unique()
        missing_hours = set(range(24)) - set(hours_with_data)

        info['hours_covered'] = len(hours_with_data)
        info['missing_hours'] = sorted(missing_hours) if missing_hours else []

        if missing_hours:
            info['reason'] = f"Missing data in hours: {sorted(missing_hours)}"
            return False, info

    info['reason'] = 'passed'
    return True, info


def _acc_to_df(acc: np.ndarray, fs: float, start_time_str: str) -> pd.DataFrame:
    """Convert raw acceleration array to datetime-indexed DataFrame."""
    start_time = pd.to_datetime(start_time_str, errors='raise')
    period_ms = int(round(1000 / fs))
    dt_index = pd.date_range(
        start=start_time, periods=len(acc), freq=f'{period_ms}ms', name='time'
    )
    return pd.DataFrame(acc, index=dt_index, columns=['x', 'y', 'z'])


def _impute_missing(data: pd.DataFrame, target_fs: int = 30) -> pd.DataFrame:
    """
    Impute non-wear segments using time-of-day averages.

    Strategy (from OxWearables/biobankAccelerometerAnalysis):
    1. Try same day-of-week + time
    2. Try weekday/weekend + time
    3. Use any day + time

    This replaces the simpler reindex-nearest approach that was used in the
    frequency/entropy scripts, ensuring consistent preprocessing everywhere.
    """
    # Extrapolate to full days
    step = pd.Timedelta(seconds=1) / target_fs
    full_range = pd.date_range(
        data.index[0].floor('D'),
        data.index[-1].ceil('D'),
        freq=step, inclusive='left', name='time'
    )
    data = data.reindex(full_range, method='nearest',
                        tolerance=pd.Timedelta('1m'), limit=1)

    def fillna(subframe):
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            nan = np.isnan(x)
            nanlen = len(x[nan])
            if 0 < nanlen < len(x):
                x[nan] = np.nanmean(x)
                return x
            return subframe

    # Cascading imputation
    data = (
        data
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(fillna)
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(fillna)
        .groupby([data.index.hour, data.index.minute])
        .transform(fillna)
    )
    return data


def _drop_edge_days(df: pd.DataFrame, which: str = 'both') -> pd.DataFrame:
    """Drop first and/or last days (partial recording days)."""
    if len(df) == 0:
        return df
    dates = df.index.date
    if which in ('first', 'both'):
        df = df[dates != dates[0]]
    if which in ('last', 'both') and len(df) > 0:
        dates = df.index.date
        df = df[dates != dates[-1]]
    return df


def compute_enmo(acc: np.ndarray) -> np.ndarray:
    """
    Compute ENMO (Euclidean Norm Minus One) in milli-g.
    Assumes calibrated data in 'g' units.
    """
    vm = np.linalg.norm(acc, axis=1)
    enmo = np.clip(vm - 1.0, a_min=0, a_max=None)
    return enmo * 1000  # Convert to mg


def compute_daily_pa(enmo_mg: np.ndarray, target_fs: int = 30):
    """
    Compute daily physical activity measures from ENMO signal.

    Returns:
        dict with 'daily_pa_mean', 'daily_pa_std', 'tdpa', 'num_days'
    """
    samples_per_day = int(24 * 60 * 60 * target_fs)
    num_days = len(enmo_mg) // samples_per_day

    result = {
        'daily_pa_mean': np.array([]),
        'daily_pa_std': np.array([]),
        'tdpa': np.array([]),
        'num_days': num_days
    }

    if num_days == 0:
        return result

    # Daily mean/std of ENMO
    enmo_daily = enmo_mg[:num_days * samples_per_day].reshape(num_days, samples_per_day)
    result['daily_pa_mean'] = enmo_daily.mean(axis=1)
    result['daily_pa_std'] = enmo_daily.std(axis=1)

    # TDPA: sum of 15-second epoch sums per day (Buchman-style)
    # Use raw g-unit signal (not mg) to avoid inflated numbers
    enmo_g = enmo_mg / 1000.0
    epoch_len = int(target_fs * 15)  # 15-second epochs
    n_epochs = len(enmo_g) // epoch_len
    if n_epochs > 0:
        epoch_sums = enmo_g[:n_epochs * epoch_len].reshape(n_epochs, epoch_len).sum(axis=1)
        epochs_per_day = int(24 * 60 * 60 / 15)  # 5760
        n_full_days = len(epoch_sums) // epochs_per_day
        if n_full_days > 0:
            result['tdpa'] = (
                epoch_sums[:n_full_days * epochs_per_day]
                .reshape(n_full_days, epochs_per_day)
                .sum(axis=1)
            )

    return result
