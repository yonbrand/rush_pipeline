"""
Stage 3: Feature extraction at window and bout level.

For each detected bout:
  - Extract overlapping windows
  - Run DL models (speed, cadence, gait length, regularity, step count)
  - Compute signal-processing features on the FULL BOUT signal:
    regularity (autocorrelation), sample entropy, PSD features
  - Compute intensity features (bout PA mean/std)

Outputs:
  - Window-level features (one row per window)
  - Bout-level features (one row per bout, DL features aggregated from windows)
"""

import logging

import numpy as np
import pandas as pd
import torch

from signal_features import calc_regularity, calc_sample_entropy, compute_psd_features

logger = logging.getLogger(__name__)


def extract_features_for_subject(
    acc: np.ndarray,
    df_index: pd.DatetimeIndex,
    bouts: list,
    models: dict,
    device,
    target_fs: int = 30,
    window_sec: int = 10,
    overlap_sec: int = 9,
    batch_size: int = 512,
    entropy_cfg: dict = None,
    frequency_cfg: dict = None,
) -> tuple:
    """
    Extract all features for one subject's detected bouts.

    Args:
        acc: Preprocessed acceleration (N, 3).
        df_index: DatetimeIndex of the preprocessed signal.
        bouts: List of bout dicts from gait_detection.detect_bouts().
        models: Dict with keys 'step_count', 'gait_speed', 'cadence',
                'gait_length', 'regularity' → torch models.
        device: torch device.
        target_fs: Sampling frequency.
        window_sec, overlap_sec: Window parameters.
        batch_size: Inference batch size.
        entropy_cfg: dict with 'm', 'r', 'max_seconds'.
        frequency_cfg: dict with 'nperseg', 'min_bout_seconds'.

    Returns:
        (window_rows, bout_rows): lists of dicts for DataFrame construction.
    """
    window_len = int(target_fs * window_sec)
    step_len = int(target_fs * (window_sec - overlap_sec))

    if entropy_cfg is None:
        entropy_cfg = {'m': 2, 'r': 0.15, 'max_seconds': 120}
    if frequency_cfg is None:
        frequency_cfg = {'nperseg': 256, 'min_bout_seconds': 2}

    max_entropy_samples = int(target_fs * entropy_cfg['max_seconds'])

    all_window_rows = []
    all_bout_rows = []

    for bout in bouts:
        bout_id = bout['bout_id']
        s_start = bout['start_sample']
        s_end = min(bout['end_sample'], len(acc))

        if s_end - s_start < window_len:
            continue

        acc_bout = acc[s_start:s_end]

        # --- Windowing ---
        n_win = max(0, (len(acc_bout) - window_len) // step_len + 1)
        if n_win == 0:
            continue

        windows = np.array([
            acc_bout[i * step_len: i * step_len + window_len]
            for i in range(n_win)
        ])

        # --- DL predictions (batched) ---
        dl_results = _run_all_models(windows, models, device, batch_size)

        # --- Bout-level signal features (computed on FULL bout, not per window) ---
        bout_vm = np.linalg.norm(acc_bout, axis=1)

        # Regularity (SP) — on full bout signal
        regularity_sp = calc_regularity(bout_vm, target_fs)

        # Entropy — on full bout signal (truncated for speed)
        entropy = calc_sample_entropy(
            bout_vm,
            m=entropy_cfg['m'],
            r=entropy_cfg['r'],
            max_samples=max_entropy_samples
        )

        # PSD features — on full bout signal
        psd = compute_psd_features(bout_vm, target_fs,
                                   nperseg=frequency_cfg['nperseg'])

        # Intensity
        bout_pa_mean = float(np.mean(bout_vm))
        bout_pa_std = float(np.std(bout_vm))

        # --- Timestamps ---
        bout_start_time = df_index[s_start] if s_start < len(df_index) else None

        # --- Step count ---
        # Per-window step counts → bout total
        # With 90% overlap (1-sec step), each window covers 10 sec but advances 1 sec.
        # Total steps = sum of per-window steps × (step_len / window_len)
        # This correctly accounts for overlap: each second is covered by 10 windows,
        # so we scale by 1/10 to avoid counting steps 10 times.
        overlap_factor = step_len / window_len  # 30/300 = 0.1 for 90% overlap
        per_window_steps = np.round(dl_results['step_count'])
        bout_total_steps = np.round(float(np.sum(per_window_steps) * overlap_factor))

        # --- Store window-level rows ---
        for w_idx in range(n_win):
            win_start_sample = s_start + w_idx * step_len
            win_time = df_index[win_start_sample] if win_start_sample < len(df_index) else None

            all_window_rows.append({
                'bout_id': bout_id,
                'window_id': w_idx,
                'window_start_time': win_time,
                'speed': float(dl_results['gait_speed'][w_idx]),
                'cadence': float(dl_results['cadence'][w_idx]),
                'gait_length': float(dl_results['gait_length'][w_idx]),
                'regularity_eldernet': float(dl_results['regularity'][w_idx]),
                'step_count': float(per_window_steps[w_idx]),
            })

        # --- Compute gait_length_indirect at bout level ---
        median_speed = float(np.median(dl_results['gait_speed']))
        median_cadence = float(np.median(dl_results['cadence']))
        gait_length_indirect = (120 * median_speed / median_cadence
                                if median_cadence > 0 else np.nan)

        # --- Store bout-level row ---
        all_bout_rows.append({
            'bout_id': bout_id,
            'start_time': bout_start_time,
            'duration_sec': bout['duration_sec'],
            'n_windows': n_win,
            'total_steps': bout_total_steps,
            # DL aggregates (median across windows)
            'speed': float(np.median(dl_results['gait_speed'])),
            'cadence': float(np.median(dl_results['cadence'])),
            'gait_length': float(np.median(dl_results['gait_length'])),
            'gait_length_indirect': gait_length_indirect,
            'regularity_eldernet': float(np.median(dl_results['regularity'])),
            # Signal-processing features (computed on FULL bout)
            'regularity_sp': regularity_sp,
            'entropy': entropy,
            'dom_freq': psd['dom_freq'],
            'psd_amp': psd['psd_amp'],
            'psd_width': psd['psd_width'],
            'psd_slope': psd['psd_slope'],
            # Intensity
            'bout_pa_mean': bout_pa_mean,
            'bout_pa_std': bout_pa_std,
        })

    return all_window_rows, all_bout_rows


def _run_all_models(windows: np.ndarray, models: dict, device,
                    batch_size: int) -> dict:
    """
    Run all gait quality models on a set of windows.

    Returns dict with arrays: gait_speed, cadence, gait_length, regularity, step_count
    """
    results = {}
    for name, model in models.items():
        preds = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            X = torch.tensor(batch, dtype=torch.float32).to(device)
            if X.shape[1] != 3:
                X = X.transpose(1, 2)
            with torch.no_grad():
                pred = model(X)
            preds.extend(pred.cpu().numpy().flatten())
        results[name] = np.array(preds)

    return results
