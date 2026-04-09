"""
Sleep and rest-activity rhythm (RAR) features from wrist-worn accelerometer.

Two complementary families of features, both derived without a sleep diary:

  A. Per-night sleep architecture via the van Hees HDCZA algorithm
     (Heuristic algorithm for Distinguishing rest periods using the Z-angle;
     van Hees et al. 2015/2018, as implemented in GGIR). Produces for each
     calendar night: sleep onset, wake time, SPT duration, total sleep time
     (TST), wake after sleep onset (WASO), sleep efficiency (SE), number of
     awakenings, fragmentation index, sleep midpoint.

  B. Nonparametric rest-activity rhythm metrics (Witting / Van Someren 1990;
     Lim, Yu, Buchman et al. on the MAP/ROS cohort): interdaily stability
     (IS), intradaily variability (IV), L5, M10, L5 onset, relative amplitude
     (RA), and sleep regularity index (SRI; Lunsford-Avery 2018).

The module is pure NumPy/Pandas. Input is the calibrated, resampled,
gap-imputed acceleration signal and its DatetimeIndex -- i.e. the exact
state of the signal at `run_pipeline.py` just after `compute_daily_pa`.
"""
from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# A. HDCZA per-night sleep detection
# ---------------------------------------------------------------------------

def compute_arm_angle(acc: np.ndarray, fs: int, epoch_sec: int = 5) -> np.ndarray:
    """
    Compute the van Hees "arm angle" (tilt of the wrist relative to horizontal)
    on non-overlapping epochs.

        angle = atan2(z, sqrt(x^2 + y^2)) * 180 / pi

    The angle is computed per-sample and then the median within each epoch is
    taken to suppress high-frequency motion artefacts.
    """
    n = acc.shape[0]
    samples_per_epoch = int(fs * epoch_sec)
    if samples_per_epoch <= 0:
        raise ValueError("epoch_sec * fs must be >= 1 sample")
    n_epochs = n // samples_per_epoch
    if n_epochs == 0:
        return np.array([])

    acc = acc[: n_epochs * samples_per_epoch]
    x, y, z = acc[:, 0], acc[:, 1], acc[:, 2]
    angle = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    angle = angle.reshape(n_epochs, samples_per_epoch)
    return np.median(angle, axis=1)


def detect_sib(
    angle_epochs: np.ndarray,
    epoch_sec: int = 5,
    smooth_min: float = 5.0,
    threshold_deg: float = 0.13,
    min_dur_min: float = 30.0,
) -> np.ndarray:
    """
    Detect Sustained Inactivity Bouts (SIB) from the per-epoch arm angle.

    Steps (HDCZA):
      1. |diff| of consecutive epoch angles.
      2. 5-minute rolling mean of the absolute diff.
      3. Mark epochs where the smoothed diff < threshold_deg.
      4. Keep runs of marked epochs whose duration >= min_dur_min.

    Returns a boolean array (same length as angle_epochs) where True = SIB.
    """
    if angle_epochs.size < 2:
        return np.zeros(angle_epochs.shape, dtype=bool)

    diff = np.abs(np.diff(angle_epochs, prepend=angle_epochs[0]))

    win = max(1, int(round(smooth_min * 60.0 / epoch_sec)))
    kernel = np.ones(win) / win
    smoothed = np.convolve(diff, kernel, mode='same')

    below = smoothed < threshold_deg

    min_run = max(1, int(round(min_dur_min * 60.0 / epoch_sec)))
    sib = np.zeros_like(below)
    # Scan runs of True
    i = 0
    n = len(below)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < n and below[j]:
            j += 1
        if (j - i) >= min_run:
            sib[i:j] = True
        i = j
    return sib


def detect_spt_windows(
    sib: np.ndarray,
    epoch_index: pd.DatetimeIndex,
    epoch_sec: int = 5,
) -> List[Tuple[int, int, pd.Timestamp, pd.Timestamp]]:
    """
    Collapse SIB epochs into one Sleep Period Time (SPT) window per night,
    defined as the longest run of SIB epochs (allowing brief wake gaps --
    standard HDCZA) inside each noon-to-noon window.

    Returns a list of (start_epoch_idx, end_epoch_idx, start_ts, end_ts).
    end is exclusive.
    """
    if sib.size == 0:
        return []

    # Identify contiguous SIB runs
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(sib)
    while i < n:
        if sib[i]:
            j = i
            while j < n and sib[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    if not runs:
        return []

    # Assign each run to the noon-to-noon window containing its midpoint.
    # Noon-to-noon windowing follows GGIR convention so that a night beginning
    # at 23:00 and ending at 06:00 is one window, not split across days.
    starts = epoch_index.values
    noon_day = pd.to_datetime(starts).normalize() + pd.Timedelta(hours=12)
    # Convert to day key: date of the most recent noon boundary <= timestamp
    ts = pd.to_datetime(starts)
    noon_key = np.where(
        ts.hour < 12,
        (ts.normalize() - pd.Timedelta(days=1)),
        ts.normalize(),
    )

    by_night: Dict[pd.Timestamp, Tuple[int, int]] = {}
    for (s, e) in runs:
        mid = (s + e) // 2
        if mid >= len(noon_key):
            continue
        key = noon_key[mid]
        duration = e - s
        prev = by_night.get(key)
        if prev is None or (prev[1] - prev[0]) < duration:
            by_night[key] = (s, e)

    out = []
    for key in sorted(by_night.keys()):
        s, e = by_night[key]
        start_ts = pd.Timestamp(epoch_index[s])
        # end is exclusive
        end_idx = min(e, len(epoch_index) - 1)
        end_ts = pd.Timestamp(epoch_index[end_idx])
        out.append((s, e, start_ts, end_ts))
    return out


def compute_nightly_sleep_metrics(
    spt_windows: List[Tuple[int, int, pd.Timestamp, pd.Timestamp]],
    sib: np.ndarray,
    epoch_sec: int = 5,
) -> List[Dict]:
    """
    For each SPT window, compute HDCZA-derived sleep metrics.

    Within the SPT window, SIB epochs count as sleep and non-SIB epochs count
    as wake (WASO). Awakenings are counted as transitions sleep->wake inside
    the SPT window.
    """
    metrics: List[Dict] = []
    epoch_hr = epoch_sec / 3600.0

    for night_idx, (s, e, start_ts, end_ts) in enumerate(spt_windows):
        seg = sib[s:e]
        spt_epochs = len(seg)
        if spt_epochs == 0:
            continue

        tst_epochs = int(seg.sum())
        waso_epochs = spt_epochs - tst_epochs

        # Awakenings = transitions from sleep (True) to wake (False)
        awakenings = int(np.sum((seg[:-1] == True) & (seg[1:] == False)))

        spt_hours = spt_epochs * epoch_hr
        tst_hours = tst_epochs * epoch_hr
        waso_hours = waso_epochs * epoch_hr
        se = tst_hours / spt_hours if spt_hours > 0 else np.nan
        frag = awakenings / tst_hours if tst_hours > 0 else np.nan

        midpoint = start_ts + (end_ts - start_ts) / 2

        metrics.append({
            'night': night_idx,
            'sleep_onset_hour': _to_clock_hour(start_ts),
            'wake_time_hour': _to_clock_hour(end_ts),
            'midsleep_hour': _to_clock_hour(midpoint),
            'spt_hours': spt_hours,
            'tst_hours': tst_hours,
            'waso_hours': waso_hours,
            'sleep_efficiency': se,
            'awakenings': awakenings,
            'frag_index': frag,
        })
    return metrics


def _to_clock_hour(ts: pd.Timestamp) -> float:
    """Convert a timestamp to fractional clock hour, wrapped to [-12, 12)
    around midnight so that 23:00 and 01:00 average correctly for circadian
    variability summaries (23:00 -> -1.0, 01:00 -> 1.0)."""
    h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    if h >= 12:
        h -= 24
    return float(h)


# ---------------------------------------------------------------------------
# B. Rest-activity rhythm metrics (IS / IV / L5 / M10 / RA / SRI)
# ---------------------------------------------------------------------------

def _resample_enmo_to_minute(enmo_mg: np.ndarray, fs: int,
                             start_ts: pd.Timestamp) -> pd.Series:
    """Average the per-sample ENMO (mg) signal into 1-minute bins aligned to
    clock minutes."""
    idx = pd.date_range(start=start_ts, periods=len(enmo_mg),
                        freq=pd.Timedelta(seconds=1.0 / fs))
    s = pd.Series(enmo_mg, index=idx)
    # Mean per minute
    return s.resample('1min').mean()


def compute_rar_metrics(enmo_minute: pd.Series) -> Dict[str, float]:
    """
    Nonparametric rest-activity rhythm metrics computed on minute-level ENMO.

    IS / IV follow Witting & Van Someren (1990). L5/M10/RA follow
    Van Someren et al. Expressed with activity in mg.
    """
    out = {
        'rar_is': np.nan, 'rar_iv': np.nan,
        'rar_l5': np.nan, 'rar_m10': np.nan,
        'rar_l5_onset_hour': np.nan, 'rar_ra': np.nan,
    }
    x = enmo_minute.dropna().values.astype(float)
    if x.size < 60 * 24 * 2:  # need at least ~2 full days
        return out
    n = x.size
    mean_all = x.mean()
    var_all = np.mean((x - mean_all) ** 2)

    # IS: variance of the average day-profile / total variance
    minute_of_day = (enmo_minute.dropna().index.hour * 60
                     + enmo_minute.dropna().index.minute).values
    day_profile = np.zeros(1440)
    counts = np.zeros(1440)
    for m, v in zip(minute_of_day, x):
        day_profile[m] += v
        counts[m] += 1
    valid = counts > 0
    day_profile[valid] /= counts[valid]
    # Only use minutes with data in the profile
    if valid.sum() > 0 and var_all > 0:
        mean_prof = day_profile[valid].mean()
        var_prof = np.mean((day_profile[valid] - mean_prof) ** 2)
        # IS = n * var(day profile) / (p * var(all)), with p=1440
        out['rar_is'] = float(
            (n * var_prof) / (1440 * var_all)
        )

    # IV: mean squared successive difference / variance
    if var_all > 0:
        diffs = np.diff(x)
        out['rar_iv'] = float(np.mean(diffs ** 2) / var_all)

    # L5 / M10 on the mean day-profile via circular rolling window
    prof = day_profile.copy()
    # Fill missing minutes with overall mean so the rolling window is defined
    if (~valid).any():
        prof[~valid] = prof[valid].mean() if valid.any() else 0.0

    def _circ_rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
        ext = np.concatenate([a, a[: w - 1]])
        c = np.cumsum(ext)
        out = (c[w - 1:] - np.concatenate([[0.0], c[: len(a) - 1]])) / w
        return out

    r5 = _circ_rolling_mean(prof, 5 * 60)
    r10 = _circ_rolling_mean(prof, 10 * 60)
    l5_start = int(np.argmin(r5))
    l5_val = float(r5[l5_start])
    m10_val = float(r10.max())
    out['rar_l5'] = l5_val
    out['rar_m10'] = m10_val
    out['rar_l5_onset_hour'] = l5_start / 60.0
    if (m10_val + l5_val) > 0:
        out['rar_ra'] = (m10_val - l5_val) / (m10_val + l5_val)

    return out


def compute_sri(sleep_wake_minute: pd.Series) -> float:
    """
    Sleep Regularity Index (Lunsford-Avery 2018).

    SRI = 200 * P(same state at t and t+24h) - 100, where state is binary
    sleep/wake on the 1-minute grid. Ranges from -100 (anti-regular) to
    100 (perfectly regular).
    """
    x = sleep_wake_minute.values
    if x.size < 2 * 1440:
        return np.nan
    a = x[:-1440]
    b = x[1440:]
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() == 0:
        return np.nan
    same = (a[mask] == b[mask]).mean()
    return float(200.0 * same - 100.0)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_sleep_features(
    acc: np.ndarray,
    enmo_mg: np.ndarray,
    df_index: pd.DatetimeIndex,
    target_fs: int,
    epoch_sec: int = 5,
) -> Dict:
    """
    Compute per-night HDCZA sleep metrics and subject-level rest-activity
    rhythm metrics.

    Args:
        acc: (N, 3) calibrated acceleration in g, sampled at target_fs Hz.
        enmo_mg: (N,) ENMO in mg, same length as acc.
        df_index: DatetimeIndex of length N corresponding to acc / enmo_mg.
        target_fs: sampling rate in Hz.
        epoch_sec: epoch length for HDCZA (default 5 s).

    Returns:
        dict with keys:
            'nightly' : list[dict]  (one per detected night)
            'rar'     : dict        (subject-level RAR metrics + SRI)
    """
    result = {'nightly': [], 'rar': {}}

    if acc.shape[0] == 0 or len(df_index) == 0:
        return result

    # --- HDCZA ---
    angle = compute_arm_angle(acc, fs=target_fs, epoch_sec=epoch_sec)
    if angle.size > 0:
        sib = detect_sib(angle, epoch_sec=epoch_sec)

        # Epoch-level timestamps (start of each epoch)
        samples_per_epoch = int(target_fs * epoch_sec)
        epoch_starts = df_index[: len(angle) * samples_per_epoch: samples_per_epoch]
        # Make sure lengths line up
        k = min(len(epoch_starts), len(sib))
        epoch_starts = epoch_starts[:k]
        sib = sib[:k]

        spt_windows = detect_spt_windows(sib, epoch_starts, epoch_sec=epoch_sec)
        result['nightly'] = compute_nightly_sleep_metrics(
            spt_windows, sib, epoch_sec=epoch_sec
        )

        # Build a minute-level sleep/wake series for SRI using the SIB mask
        # projected inside SPT windows.
        sw_epoch = np.zeros_like(sib, dtype=float)
        for (s, e, _st, _et) in spt_windows:
            sw_epoch[s:e] = sib[s:e].astype(float)
        # Resample epoch-level sleep/wake to minute-level by max (any sleep -> sleep)
        if len(epoch_starts) > 0:
            sw_series = pd.Series(sw_epoch, index=epoch_starts)
            sw_minute = sw_series.resample('1min').max()
            sri = compute_sri(sw_minute)
        else:
            sri = np.nan
    else:
        sri = np.nan

    # --- RAR on minute-level ENMO ---
    if enmo_mg.size > 0 and len(df_index) > 0:
        enmo_series = pd.Series(enmo_mg, index=df_index[: len(enmo_mg)])
        enmo_minute = enmo_series.resample('1min').mean()
        rar = compute_rar_metrics(enmo_minute)
    else:
        rar = {
            'rar_is': np.nan, 'rar_iv': np.nan,
            'rar_l5': np.nan, 'rar_m10': np.nan,
            'rar_l5_onset_hour': np.nan, 'rar_ra': np.nan,
        }
    rar['rar_sri'] = sri
    result['rar'] = rar

    return result
