"""
RUSH Gait Analysis Pipeline — Feature Extraction

Processes daily-living accelerometer recordings (GENEActive / Axivity) to extract
gait quality measures using deep learning and signal processing.

Architecture (3 stages):

  Stage 1: Preprocessing    — raw .mat → calibrated, resampled, imputed signal
  Stage 2: Gait Detection   — signal → walking bouts (ElderNet classification)
  Stage 3: Feature Extract  — bouts → window-level & bout-level gait features

Output structure:
  {output_path}/
    bouts/          — one CSV per subject (bout-level features)
    windows/        — one CSV per subject (window-level features)
    daily_pa/       — one CSV per subject (daily physical activity)

For summary statistics, run aggregate_subjects.py separately:
    python aggregate_subjects.py --output-dir {output_path}

Usage:
    python run_pipeline.py                           # use default config.yaml
    python run_pipeline.py --config path/to/cfg.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import load_config
from io_utils import (
    list_mat_files, extract_raw_data, parse_subject_id, setup_model,
)
from preprocessing import preprocess_subject, compute_enmo, compute_daily_pa
from gait_detection import (
    run_gait_detection, window_predictions_to_seconds,
    merge_bouts, detect_bouts,
)
from feature_extraction import extract_features_for_subject

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_models(cfg, device):
    """Load all models from config and set to eval mode."""
    epoch_len = cfg.pipeline.epoch_len
    model_names = [
        'gait_detection', 'step_count', 'gait_speed',
        'cadence', 'gait_length', 'regularity'
    ]

    models = {}
    for name in model_names:
        logger.info(f"Loading model: {name}")
        model_cfg = cfg.models[name]
        model = setup_model(model_cfg, epoch_len=epoch_len, device=device)
        model.eval()
        models[name] = model

    return models


def process_subject(
    file_path,
    models: dict,
    cfg,
    device,
    output_dir: Path,
) -> bool:
    """
    Full pipeline for a single subject: Stages 1-3 + save CSVs.

    Returns:
        True if processed successfully, False otherwise.
    """
    subject_id = parse_subject_id(file_path)
    if subject_id is None:
        logger.warning(f"Could not parse subject ID from {file_path}")
        return False

    # Check if already processed
    bout_file = output_dir / 'bouts' / f'{subject_id}.csv'
    if bout_file.exists():
        logger.info(f"Skipping {subject_id} (already processed)")
        return False

    logger.info(f"Processing {subject_id}: {Path(file_path).name}")

    # ================================================================
    # STAGE 1: Preprocessing
    # ================================================================
    raw_data = extract_raw_data(file_path, cfg.sensor_device)
    if raw_data is None:
        return False

    target_fs = cfg.pipeline.resampled_hz
    df_proc = preprocess_subject(
        acc=raw_data['acc'],
        fs=raw_data['fs'],
        start_time_str=raw_data['start_time_str'],
        target_fs=target_fs,
        drop_first_last=cfg.pipeline.get('drop_first_last_days', True),
        min_wear_hours=cfg.pipeline.get('min_wear_hours', 72.0),
        require_all_hours=cfg.pipeline.get('require_all_hours', True),
    )
    if df_proc is None:
        return False

    processed_acc = df_proc[['x', 'y', 'z']].to_numpy()
    df_index = df_proc.index

    # Compute daily PA from full signal
    enmo_mg = compute_enmo(processed_acc)
    daily_pa = compute_daily_pa(enmo_mg, target_fs)
    num_days = daily_pa['num_days']

    if num_days == 0:
        logger.warning(f"{subject_id}: No full days of data")
        return False

    # ================================================================
    # STAGE 2: Gait Detection + Bout Assembly
    # ================================================================
    window_sec = cfg.pipeline.window_sec
    overlap_sec = cfg.pipeline.window_overlap_sec
    window_len = int(target_fs * window_sec)
    step_len = int(target_fs * (window_sec - overlap_sec))

    # Run gait detection
    pred_walk = run_gait_detection(
        processed_acc, models['gait_detection'], device,
        window_len, step_len,
        batch_size=cfg.pipeline.get('inference_batch_size', 512),
        static_variance_percentile=cfg.pipeline.get('static_variance_percentile', 10.0),
    )

    if len(pred_walk) == 0:
        logger.warning(f"{subject_id}: No gait predictions")
        return False

    # Map to seconds → merge → detect bouts
    second_preds = window_predictions_to_seconds(pred_walk, window_sec)
    merged_preds = merge_bouts(
        second_preds,
        min_bout_sec=cfg.pipeline.min_bout_duration_sec,
        merge_gap_sec=cfg.pipeline.merge_gap_sec,
    )
    bouts = detect_bouts(merged_preds, target_fs)

    if not bouts:
        logger.warning(f"{subject_id}: No walking bouts detected")
        # Still save daily PA and empty bout/window files
        _save_daily_pa(daily_pa, num_days, subject_id, output_dir)
        _save_empty_bouts(subject_id, output_dir)
        return True

    # Assign day to each bout
    samples_per_day = int(24 * 60 * 60 * target_fs)
    for bout in bouts:
        bout['day'] = bout['start_sample'] // samples_per_day

    # ================================================================
    # STAGE 3: Feature Extraction
    # ================================================================
    quality_models = {k: v for k, v in models.items() if k != 'gait_detection'}

    window_rows, bout_rows = extract_features_for_subject(
        acc=processed_acc,
        df_index=df_index,
        bouts=bouts,
        models=quality_models,
        device=device,
        target_fs=target_fs,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        batch_size=cfg.pipeline.get('inference_batch_size', 512),
        entropy_cfg=dict(cfg.entropy) if hasattr(cfg, 'entropy') else None,
        frequency_cfg=dict(cfg.frequency) if hasattr(cfg, 'frequency') else None,
    )

    # Add day info to bout rows
    bout_day_map = {b['bout_id']: b['day'] for b in bouts}
    for row in bout_rows:
        row['day'] = bout_day_map.get(row['bout_id'], -1)

    # Build DataFrames
    bout_df = pd.DataFrame(bout_rows)
    window_df = pd.DataFrame(window_rows)

    # Add subject_id
    bout_df.insert(0, 'subject_id', subject_id)
    window_df.insert(0, 'subject_id', subject_id)

    # ================================================================
    # SAVE OUTPUT CSVs
    # ================================================================
    _save_bout_csv(bout_df, subject_id, output_dir)
    _save_window_csv(window_df, subject_id, output_dir)
    _save_daily_pa(daily_pa, num_days, subject_id, output_dir)

    logger.info(f"{subject_id}: Saved {len(bout_df)} bouts, {len(window_df)} windows")
    return True


# ============================================================================
# File saving helpers
# ============================================================================

def _save_bout_csv(df, subject_id, output_dir):
    path = output_dir / 'bouts' / f'{subject_id}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_window_csv(df, subject_id, output_dir):
    path = output_dir / 'windows' / f'{subject_id}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_daily_pa(daily_pa, num_days, subject_id, output_dir):
    path = output_dir / 'daily_pa' / f'{subject_id}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in range(num_days):
        row = {'day': d}
        for key in ['daily_pa_mean', 'daily_pa_std', 'tdpa']:
            arr = daily_pa.get(key, np.array([]))
            row[key] = float(arr[d]) if d < len(arr) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_empty_bouts(subject_id, output_dir):
    for subdir in ['bouts', 'windows']:
        path = output_dir / subdir / f'{subject_id}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(path, index=False)


# ============================================================================
# Main entry points
# ============================================================================

def run_full_pipeline(cfg):
    """Run the feature extraction pipeline on all subjects."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(cfg.data.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models = load_all_models(cfg, device)

    # Discover files
    files = list_mat_files(cfg.data.data_path)
    logger.info(f"Found {len(files)} .mat files to process")

    # Process subjects
    n_success = 0
    for file_path in tqdm(files, desc="Processing subjects"):
        try:
            success = process_subject(file_path, models, cfg, device, output_dir)
            if success:
                n_success += 1
        except Exception as e:
            logger.error(f"Failed on {file_path}: {e}", exc_info=True)
        finally:
            torch.cuda.empty_cache()

    logger.info(f"Feature extraction complete: {n_success}/{len(files)} subjects")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"To compute summary statistics, run:")
    logger.info(f"  python aggregate_subjects.py --output-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="RUSH Gait Analysis Pipeline — Feature Extraction (Stages 1-3)"
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (default: config.yaml in package dir)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_full_pipeline(cfg)


if __name__ == '__main__':
    main()
