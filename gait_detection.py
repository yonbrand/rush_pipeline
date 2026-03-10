"""
Stage 2: Gait detection and bout assembly.

Takes preprocessed acceleration, runs ElderNet gait detection,
maps window predictions to seconds, merges nearby bouts, and
outputs a bout table.

BUG FIX: merge_bouts() now correctly handles chains of mergeable bouts
using a while-loop instead of a for-loop that mutated arrays during iteration.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def run_gait_detection(acc: np.ndarray, model, device,
                       window_len: int, step_len: int,
                       batch_size: int = 512,
                       static_variance_percentile: float = 10.0) -> np.ndarray:
    """
    Run gait detection model on windowed acceleration data.

    Args:
        acc: Preprocessed acceleration (N, 3).
        model: Gait detection model (classification, 2 outputs).
        device: torch device.
        window_len: Window length in samples.
        step_len: Step between windows in samples.
        batch_size: Inference batch size.
        static_variance_percentile: Windows with variance below this percentile
                                    are considered static and marked as non-gait
                                    without running inference. Set to 0 to disable.

    Returns:
        Binary predictions per window (1=walking, 0=not).
    """
    # Extract windows
    n_windows = max(0, (len(acc) - window_len) // step_len + 1)
    if n_windows == 0:
        return np.array([])

    windows = np.array([acc[i * step_len: i * step_len + window_len]
                        for i in range(n_windows)])

    # Pre-filter: calculate variance per window (mean across 3 axes)
    # Static signals have very low variance and can't be gait
    window_variance = np.var(windows, axis=1).mean(axis=1)

    # Adaptive threshold: use percentile of this subject's variance distribution
    if static_variance_percentile > 0:
        variance_threshold = np.percentile(window_variance, static_variance_percentile)
        active_mask = window_variance > variance_threshold
    else:
        active_mask = np.ones(n_windows, dtype=bool)

    active_indices = np.where(active_mask)[0]

    n_filtered = n_windows - len(active_indices)
    if n_filtered > 0:
        logger.debug(f"Filtered {n_filtered}/{n_windows} low-variance windows "
                     f"(percentile={static_variance_percentile}, threshold={variance_threshold:.4f})")

    # Initialize all predictions as non-gait
    predictions = np.zeros(n_windows, dtype=int)

    if len(active_indices) == 0:
        return predictions

    # Only run inference on active (high-variance) windows
    active_windows = windows[active_indices]

    active_preds = []
    for i in range(0, len(active_windows), batch_size):
        batch = active_windows[i:i + batch_size]
        preds = _run_batch(batch, model, device, is_classification=True)
        active_preds.extend(preds)

    # Map predictions back to original indices
    predictions[active_indices] = active_preds

    return predictions


def window_predictions_to_seconds(pred_walk: np.ndarray,
                                  window_sec: int) -> np.ndarray:
    """
    Map window-level predictions to second-level using the center-second approach.

    Each window's prediction is assigned to its center second.
    """
    n_seconds = len(pred_walk) + window_sec - 1
    second_preds = np.zeros(n_seconds, dtype=int)

    for idx, pred in enumerate(pred_walk):
        center = idx + window_sec // 2
        if center < n_seconds:
            second_preds[center] = int(pred)

    return second_preds


def merge_bouts(pred_walk: np.ndarray,
                min_bout_sec: int = 10,
                merge_gap_sec: int = 3) -> np.ndarray:
    """
    Post-process gait predictions: merge nearby bouts and filter short ones.

    FIXED: Uses while-loop to correctly handle chains of mergeable bouts.
    The original for-loop mutated bout_starts/bout_ends during iteration,
    producing incorrect results when 3+ bouts were within merge distance.

    Args:
        pred_walk: Second-level binary walking predictions.
        min_bout_sec: Minimum bout duration in seconds.
        merge_gap_sec: Maximum gap to merge across.

    Returns:
        Post-processed binary predictions.
    """
    if len(pred_walk) == 0:
        return pred_walk

    # Find bout boundaries
    diff = np.diff(np.concatenate([[0], pred_walk, [0]]))
    starts = list(np.where(diff == 1)[0])
    ends = list(np.where(diff == -1)[0])

    if not starts:
        return np.zeros_like(pred_walk)

    # Merge close bouts (while-loop handles chains correctly)
    i = 0
    while i < len(starts) - 1:
        if starts[i + 1] - ends[i] <= merge_gap_sec:
            # Merge: extend current bout to cover the next one
            ends[i] = ends[i + 1]
            starts.pop(i + 1)
            ends.pop(i + 1)
            # Don't increment i — check if the merged bout can merge with the next
        else:
            i += 1

    # Filter short bouts
    result = np.zeros_like(pred_walk)
    for start, end in zip(starts, ends):
        if end - start >= min_bout_sec:
            result[start:end] = 1

    return result


def detect_bouts(second_preds: np.ndarray, target_fs: int):
    """
    Detect individual bouts and return bout metadata.

    Args:
        second_preds: Merged second-level walking predictions.
        target_fs: Sampling frequency (for sample-level expansion).

    Returns:
        list of dicts, each with:
            'bout_id': int
            'start_sec': start second index
            'end_sec': end second index
            'duration_sec': bout duration
            'start_sample': start sample in resampled signal
            'end_sample': end sample in resampled signal
    """
    diff = np.diff(np.concatenate([[0], second_preds, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    bouts = []
    for bout_id, (start, end) in enumerate(zip(starts, ends), 1):
        bouts.append({
            'bout_id': bout_id,
            'start_sec': int(start),
            'end_sec': int(end),
            'duration_sec': int(end - start),
            'start_sample': int(start * target_fs),
            'end_sample': int(end * target_fs),
        })

    return bouts


def _run_batch(batch: np.ndarray, model, device,
               is_classification: bool = False) -> np.ndarray:
    """Run a single batch through a model."""
    X = torch.tensor(batch, dtype=torch.float32).to(device)
    if X.shape[1] != 3:
        X = X.transpose(1, 2)

    with torch.no_grad():
        preds = model(X)
        if is_classification and preds.shape[1] == 2:
            preds = torch.argmax(preds, dim=1)

    return preds.cpu().numpy().flatten()
