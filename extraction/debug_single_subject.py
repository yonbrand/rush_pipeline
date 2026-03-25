"""
Debug runner for PyCharm.

How to use:
  1. Open this file in PyCharm
  2. Right-click → Debug 'debug_single_subject'
  3. Set breakpoints in ANY module — they will be hit

Processes a single .mat file through the full pipeline (Stages 1-4).
Adjust the file path and config path below to match your setup.
"""

import logging
import torch
from pathlib import Path

# ── Configure these ──────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
SINGLE_FILE = "<path-to-your-mat-file>"  # ← pick one real .mat file
OUTPUT_DIR  = "debug_output"
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG,   # DEBUG = maximum verbosity
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    from config import load_config
    from io_utils import extract_raw_data, parse_subject_id, setup_model
    from preprocessing import preprocess_subject, compute_enmo, compute_daily_pa
    from gait_detection import (run_gait_detection, window_predictions_to_seconds,
                                merge_bouts, detect_bouts)
    from feature_extraction import extract_features_for_subject
    from aggregation import aggregate_subject

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(CONFIG_PATH)
    # Override output path for debug isolation
    cfg.data.output_path = OUTPUT_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    target_fs = cfg.pipeline.resampled_hz
    window_sec = cfg.pipeline.window_sec
    overlap_sec = cfg.pipeline.window_overlap_sec
    window_len = int(target_fs * window_sec)
    step_len = int(target_fs * (window_sec - overlap_sec))

    # ── Load models ──────────────────────────────────────────────────────────
    # TIP: If model loading is slow, load once and reuse.
    # To skip model loading entirely during preprocessing debugging,
    # comment out this block and the Stage 2/3 sections below.
    logger.info("Loading models...")
    epoch_len = cfg.pipeline.epoch_len
    models = {}
    for name in ['gait_detection', 'step_count', 'gait_speed',
                  'cadence', 'gait_length', 'regularity']:
        logger.info(f"  Loading {name}...")
        models[name] = setup_model(cfg.models[name], epoch_len=epoch_len, device=device)
        models[name].eval()
    logger.info("All models loaded.")

    # ── STAGE 1: Preprocessing ───────────────────────────────────────────────
    file_path = Path(SINGLE_FILE)
    subject_id = parse_subject_id(file_path)
    logger.info(f"Subject ID: {subject_id}")

    raw = extract_raw_data(file_path, cfg.sensor_device)
    assert raw is not None, f"Failed to load {file_path}"
    logger.info(f"Raw acc shape: {raw['acc'].shape}, fs={raw['fs']}")

    df_proc = preprocess_subject(
        acc=raw['acc'], fs=raw['fs'],
        start_time_str=raw['start_time_str'],
        target_fs=target_fs,
        drop_first_last=cfg.pipeline.get('drop_first_last_days', True),
        min_wear_hours=cfg.pipeline.get('min_wear_hours', 72.0),
        require_all_hours=cfg.pipeline.get('require_all_hours', True),
    )
    assert df_proc is not None, "Preprocessing returned None"
    logger.info(f"Preprocessed: {len(df_proc)} samples, "
                f"{df_proc.index[0]} → {df_proc.index[-1]}")

    processed_acc = df_proc[['x', 'y', 'z']].to_numpy()
    enmo_mg = compute_enmo(processed_acc)
    daily_pa = compute_daily_pa(enmo_mg, target_fs)
    logger.info(f"Days: {daily_pa['num_days']}, "
                f"daily PA mean: {daily_pa['daily_pa_mean']}")

    # ── STAGE 2: Gait Detection ─────────────────────────────────────────────
    logger.info("Running gait detection...")
    pred_walk = run_gait_detection(
        processed_acc, models['gait_detection'], device,
        window_len, step_len,
        batch_size=cfg.pipeline.get('inference_batch_size', 512),
        filter_static=cfg.pipeline.get('filter_static_windows', True),
    )
    logger.info(f"Window predictions: {len(pred_walk)}, "
                f"walking={pred_walk.sum()}/{len(pred_walk)} "
                f"({100*pred_walk.mean():.1f}%)")

    second_preds = window_predictions_to_seconds(pred_walk, window_sec)
    merged = merge_bouts(second_preds,
                         min_bout_sec=cfg.pipeline.min_bout_duration_sec,
                         merge_gap_sec=cfg.pipeline.merge_gap_sec)
    bouts = detect_bouts(merged, target_fs)
    logger.info(f"Detected {len(bouts)} bouts")

    if bouts:
        durations = [b['duration_sec'] for b in bouts]
        logger.info(f"Bout durations: min={min(durations)}s, max={max(durations)}s, "
                     f"median={sorted(durations)[len(durations)//2]}s")

    # Quick sanity check — print first 3 bouts
    for b in bouts[:3]:
        logger.info(f"  Bout {b['bout_id']}: {b['start_sec']}s → {b['end_sec']}s "
                     f"({b['duration_sec']}s)")

    # ── STAGE 3: Feature Extraction ──────────────────────────────────────────
    logger.info("Extracting features...")
    quality_models = {k: v for k, v in models.items() if k != 'gait_detection'}

    window_rows, bout_rows = extract_features_for_subject(
        acc=processed_acc,
        df_index=df_proc.index,
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
    logger.info(f"Extracted: {len(window_rows)} windows, {len(bout_rows)} bouts")

    # Inspect first bout's features
    if bout_rows:
        b = bout_rows[0]
        logger.info(f"First bout: speed={b['speed']:.3f} m/s, "
                     f"cadence={b['cadence']:.1f} steps/min, "
                     f"regularity_sp={b['regularity_sp']:.3f}, "
                     f"entropy={b['entropy']:.3f}")

    # ── STAGE 4: Aggregation ─────────────────────────────────────────────────
    import pandas as pd
    bout_df = pd.DataFrame(bout_rows)
    window_df = pd.DataFrame(window_rows)

    logger.info("Aggregating...")
    summary = aggregate_subject(
        bout_df=bout_df, window_df=window_df,
        daily_pa=daily_pa, subject_id=subject_id,
        num_days=daily_pa['num_days'], target_fs=target_fs,
    )

    # Print key summary stats
    for key in ['bout_speed_median', 'bout_cadence_median',
                'bout_regularity_sp_median', 'bout_entropy_median',
                'daily_walking_mean', 'daily_step_count_mean', 'n_bouts']:
        if key in summary:
            logger.info(f"  {key}: {summary[key]}")

    # ── Save debug output ────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    bout_df.to_csv(out / f'{subject_id}_bouts.csv', index=False)
    window_df.to_csv(out / f'{subject_id}_windows.csv', index=False)
    pd.DataFrame([summary]).to_csv(out / f'{subject_id}_summary.csv', index=False)
    logger.info(f"Debug output saved to {out}")

    # ── PUT BREAKPOINT HERE to inspect everything ────────────────────────────
    print("\n✓ Pipeline completed successfully. Inspect variables above.")  # ← breakpoint here


if __name__ == '__main__':
    main()
