"""
Debug runner for PyCharm.

How to use:
  1. Open this file in PyCharm
  2. Right-click → Debug 'debug_single_subject'
  3. Set breakpoints in ANY module — they will be hit

Processes a single .mat file through the full pipeline (Stages 1-3).
Adjust the file path, config path, and STAGES below to match your setup.
"""

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Configure these ──────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
SINGLE_FILE = r"C:\Users\yonbr\rush_pipeline\recording\2020_03_23180_50957643_05_02282020.mat"  # ← pick one real .mat file
OUTPUT_DIR  = "debug_output"
STAGES = {'sleep'}   # ← subset of {'gait', 'pa', 'sleep'}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG,   # DEBUG = maximum verbosity
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    from config import load_config
    from io_utils import extract_raw_data, parse_subject_id, setup_model
    from preprocessing import preprocess_subject, compute_enmo, compute_daily_pa
    from sleep_features import compute_sleep_features
    from gait_detection import (run_gait_detection, window_predictions_to_seconds,
                                merge_bouts, detect_bouts)
    from feature_extraction import extract_features_for_subject

    stages = STAGES
    logger.info(f"Stages enabled: {sorted(stages)}")

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(CONFIG_PATH)
    # Override output path for debug isolation
    cfg.data.output_path = OUTPUT_DIR

    target_fs = cfg.pipeline.resampled_hz

    # ── Load models (only when gait stage is requested) ──────────────────────
    if 'gait' in stages:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        logger.info("Loading models...")
        epoch_len = cfg.pipeline.epoch_len
        models = {}
        for name in ['gait_detection', 'step_count', 'gait_speed',
                      'cadence', 'gait_length', 'regularity']:
            logger.info(f"  Loading {name}...")
            models[name] = setup_model(cfg.models[name], epoch_len=epoch_len, device=device)
            models[name].eval()
        logger.info("All models loaded.")
    else:
        device = torch.device("cpu")
        models = {}
        logger.info("Gait stage disabled — skipping DL model loading")

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
        max_gap_minutes=cfg.pipeline.get('max_gap_minutes', 180.0),
        detect_nonwear=cfg.pipeline.get('detect_nonwear', True),
        nonwear_patience=cfg.pipeline.get('nonwear_patience', '120m'),
        nonwear_stdtol=cfg.pipeline.get('nonwear_stdtol', 0.013),
    )
    assert df_proc is not None, "Preprocessing returned None"
    logger.info(f"Preprocessed: {len(df_proc)} samples, "
                f"{df_proc.index[0]} → {df_proc.index[-1]}")

    processed_acc = df_proc[['x', 'y', 'z']].to_numpy()
    df_index = df_proc.index

    # ── ENMO (needed by PA and Sleep) ────────────────────────────────────────
    need_enmo = ('pa' in stages) or ('sleep' in stages)
    enmo_mg = compute_enmo(processed_acc) if need_enmo else None

    # ── PA stage ─────────────────────────────────────────────────────────────
    daily_pa = None
    num_days = None
    if 'pa' in stages:
        daily_pa = compute_daily_pa(enmo_mg, target_fs)
        num_days = daily_pa['num_days']
        logger.info(f"Days: {num_days}, daily PA mean: {daily_pa['daily_pa_mean']}")

    # ── Sleep stage ──────────────────────────────────────────────────────────
    sleep = None
    if 'sleep' in stages:
        logger.info("Computing sleep features...")
        sleep = compute_sleep_features(processed_acc, enmo_mg, df_index, target_fs)
        nightly = sleep.get('nightly', []) or []
        rar = sleep.get('rar', {}) or {}
        logger.info(f"Sleep: {len(nightly)} nights detected")
        if nightly:
            n = nightly[0]
            logger.info(f"  Night 0: onset={n.get('sleep_onset_hour')}, "
                         f"wake={n.get('wake_time_hour')}, "
                         f"efficiency={n.get('sleep_efficiency')}")
        if rar:
            logger.info(f"  RAR keys: {list(rar.keys())}")

    # ── Gait stages (2 & 3) ─────────────────────────────────────────────────
    bout_df = pd.DataFrame()
    window_df = pd.DataFrame()
    bouts = []

    if 'gait' in stages:
        window_sec = cfg.pipeline.window_sec
        overlap_sec = cfg.pipeline.window_overlap_sec
        window_len = int(target_fs * window_sec)
        step_len = int(target_fs * (window_sec - overlap_sec))

        # ── STAGE 2: Gait Detection ─────────────────────────────────────────
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

        # ── STAGE 3: Feature Extraction ──────────────────────────────────────
        if bouts:
            logger.info("Extracting features...")
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
            logger.info(f"Extracted: {len(window_rows)} windows, {len(bout_rows)} bouts")

            # Inspect first bout's features
            if bout_rows:
                b = bout_rows[0]
                logger.info(f"First bout: speed={b['speed']:.3f} m/s, "
                             f"cadence={b['cadence']:.1f} steps/min, "
                             f"regularity_sp={b['regularity_sp']:.3f}, "
                             f"entropy={b['entropy']:.3f}")

            bout_df = pd.DataFrame(bout_rows)
            window_df = pd.DataFrame(window_rows)
        else:
            logger.warning("No walking bouts detected — skipping feature extraction")

    # ── Save debug output ────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    if 'gait' in stages:
        bout_df.to_csv(out / f'{subject_id}_bouts.csv', index=False)
        window_df.to_csv(out / f'{subject_id}_windows.csv', index=False)
        logger.info(f"Saved {len(bout_df)} bouts, {len(window_df)} windows")

    if 'pa' in stages and daily_pa and num_days:
        rows = []
        for d in range(num_days):
            row = {'day': d}
            for key in ['daily_pa_mean', 'daily_pa_std', 'tdpa']:
                arr = daily_pa.get(key, np.array([]))
                row[key] = float(arr[d]) if d < len(arr) else np.nan
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f'{subject_id}_daily_pa.csv', index=False)
        logger.info(f"Saved daily PA ({num_days} days)")

    if 'sleep' in stages and sleep:
        nightly = sleep.get('nightly', []) or []
        rar = sleep.get('rar', {}) or {}
        if nightly:
            pd.DataFrame(nightly).to_csv(out / f'{subject_id}_daily_sleep.csv', index=False)
        else:
            pd.DataFrame().to_csv(out / f'{subject_id}_daily_sleep.csv', index=False)
        pd.DataFrame([rar]).to_csv(out / f'{subject_id}_rar.csv', index=False)
        logger.info(f"Saved sleep ({len(nightly)} nights) + RAR")

    logger.info(f"Debug output saved to {out}")

    # ── PUT BREAKPOINT HERE to inspect everything ────────────────────────────
    print("\n✓ Pipeline completed successfully. Inspect variables above.")  # ← breakpoint here


if __name__ == '__main__':
    main()
