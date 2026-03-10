"""
Signal-processing feature extraction: regularity, sample entropy, PSD.

Unified from regularity.txt, calc_entropy_all.txt, calc_freq_all.txt.

DESIGN NOTE on regularity:
    The original pipeline computed stride regularity on each 10-sec window with
    90% overlap, producing ~N highly autocorrelated estimates per bout. This is
    methodologically flawed — consecutive windows share 9 seconds of data, so
    "bout median regularity" has a much smaller effective sample size than N.

    The refactored version computes regularity ONCE per bout on the full bout
    signal, which is the correct approach for a bout-level measure. Window-level
    regularity from the ElderNet model is still available separately.
"""

import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

logger = logging.getLogger(__name__)

# Try numba for fast entropy
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.info("numba not installed — sample entropy will be slow. "
                "Install with: pip install numba")


# ============================================================================
# Stride Regularity (autocorrelation method)
# ============================================================================

def calc_regularity(acc_magnitude: np.ndarray, sample_rate: float) -> float:
    """
    Calculate stride regularity from acceleration magnitude signal.

    Uses autocorrelation peak detection to find the stride regularity
    coefficient. Designed for the FULL BOUT signal, not individual windows.

    Args:
        acc_magnitude: 1D array, acceleration magnitude (single axis or norm).
        sample_rate: Sampling frequency in Hz.

    Returns:
        Stride regularity (0-1), or 0 if peaks can't be detected.
    """
    if len(acc_magnitude) < int(2 * sample_rate):
        return 0.0

    # Biased autocovariance
    c, lags = _xcov(acc_magnitude, biased=True)
    normalized_c = c / c.max()

    # Keep positive lags only
    normalized_c = normalized_c[lags >= 0]
    lags = lags[lags >= 0]

    # Smooth
    smoothed = _smooth(normalized_c, sample_rate)

    # Find peaks
    min_dist = sample_rate / 4
    locs, _ = find_peaks(smoothed, distance=int(min_dist))
    pks = smoothed[locs]

    # Keep positive peaks
    mask = pks > 0
    pks, locs = pks[mask], locs[mask]

    # Correct peaks to raw signal
    pks, locs = _correct_peaks(normalized_c, pks, locs)

    if pks.size > 1:
        return float(pks[1])  # Second peak = stride regularity
    return 0.0


def _xcov(x, biased=False):
    """Autocovariance (similar to MATLAB's xcov)."""
    n = len(x)
    x_centered = x - np.mean(x)
    c_full = np.correlate(x_centered, x_centered, mode='full')
    lags = np.arange(-n + 1, n)
    if biased:
        c = c_full / n
    else:
        c = c_full / (n - np.abs(lags))
    return c, lags


def _smooth(x, fs):
    """Moving average smoothing."""
    window_size = int(0.2 * fs)
    if window_size % 2 == 0:
        window_size += 1
    smoothed = pd.Series(x).rolling(window=window_size, center=True).mean().to_numpy()
    # Fill edge NaNs with original values
    nan_mask = np.isnan(smoothed)
    smoothed[nan_mask] = x[nan_mask]
    return smoothed


def _correct_peaks(data, pks, locs):
    """Correct peak locations from smoothed to raw signal."""
    if len(locs) < 2:
        return pks, locs

    locale_win = int(np.ceil(0.2 * np.median(np.diff(locs))))

    # Remove peaks too close to edges
    valid = (locs > locale_win) & (locs < (len(data) - locale_win))
    pks, locs = pks[valid], locs[valid]

    # Align to raw peaks
    for i in range(len(locs)):
        start = locs[i] - locale_win
        end = locs[i] + (locale_win // 2) + 1
        window = data[start:end]
        max_idx = np.argmax(window)
        pks[i] = window[max_idx]
        locs[i] = start + max_idx

    # Remove duplicates from correction
    close = np.where(np.diff(locs) < locale_win)[0]
    for idx in close[::-1]:
        if pks[idx] > pks[idx + 1]:
            pks = np.delete(pks, idx + 1)
            locs = np.delete(locs, idx + 1)
        else:
            pks = np.delete(pks, idx)
            locs = np.delete(locs, idx)

    return pks, locs


# ============================================================================
# Sample Entropy
# ============================================================================

def _sample_entropy_pure(signal_data, m, r, threshold):
    """Pure Python sample entropy (O(N^2), slow without numba)."""
    N = len(signal_data)
    count_m = 0.0
    count_m1 = 0.0

    for i in range(N - m):
        for j in range(N - m):
            if i == j:
                continue
            dist = 0.0
            for k in range(m):
                d = abs(signal_data[i + k] - signal_data[j + k])
                if d > dist:
                    dist = d
            if dist <= threshold:
                count_m += 1.0
                if (i + m < N) and (j + m < N):
                    d_next = abs(signal_data[i + m] - signal_data[j + m])
                    if max(dist, d_next) <= threshold:
                        count_m1 += 1.0

    if count_m == 0 or count_m1 == 0:
        return np.nan
    return -np.log(count_m1 / count_m)


# Compile with numba if available
if HAS_NUMBA:
    _sample_entropy_fast = jit(nopython=True)(_sample_entropy_pure)
else:
    _sample_entropy_fast = _sample_entropy_pure


def calc_sample_entropy(signal_data: np.ndarray, m: int = 2, r: float = 0.15,
                        max_samples: int = 3600) -> float:
    """
    Compute sample entropy of a signal.

    Args:
        signal_data: 1D signal array.
        m: Embedding dimension.
        r: Tolerance (as fraction of signal std).
        max_samples: Truncate signal to this length (entropy is O(N^2)).

    Returns:
        Sample entropy value, or NaN if signal too short.
    """
    if len(signal_data) <= 200:
        return np.nan
    if len(signal_data) > max_samples:
        signal_data = signal_data[:max_samples]

    std_val = np.std(signal_data)
    if std_val == 0:
        return np.nan

    threshold = r * std_val
    return _sample_entropy_fast(signal_data, m, r, threshold)


# ============================================================================
# PSD / Frequency Features
# ============================================================================

def compute_psd_features(signal_data: np.ndarray, fs: float,
                         nperseg: int = 256) -> dict:
    """
    Extract frequency-domain features from a signal.

    Args:
        signal_data: 1D signal (e.g., acceleration magnitude of a bout).
        fs: Sampling frequency.
        nperseg: FFT segment length for Welch's method.

    Returns:
        dict with: dom_freq, psd_amp, psd_width, psd_slope
    """
    nans = {'dom_freq': np.nan, 'psd_amp': np.nan,
            'psd_width': np.nan, 'psd_slope': np.nan}

    if len(signal_data) < 2 * fs:
        return nans

    try:
        f, psd = welch(signal_data, fs=fs, nperseg=min(nperseg, len(signal_data)))

        # Dominant frequency & amplitude
        idx = np.argmax(psd)
        dom_freq = f[idx]
        amp = psd[idx]

        # Width at half maximum
        half_max = amp / 2
        left_idx = idx
        while left_idx > 0 and psd[left_idx] > half_max:
            left_idx -= 1
        right_idx = idx
        while right_idx < len(psd) - 1 and psd[right_idx] > half_max:
            right_idx += 1

        width = f[right_idx] - f[left_idx]
        if width == 0:
            width = np.nan

        slope = amp / width if (width and not np.isnan(width)) else np.nan

        return {'dom_freq': dom_freq, 'psd_amp': amp,
                'psd_width': width, 'psd_slope': slope}
    except Exception:
        return nans
