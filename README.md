# RUSH Gait Analysis Pipeline

## File Structure

```
rush_pipeline/
├── config.yaml              # All settings: paths, model weights, hyperparameters
├── config.py                # YAML loader with dot-access
├── models.py                # ElderNet / ResNet architectures (unchanged for weight compat)
├── io_utils.py              # .mat loading, subject ID parsing, model setup + weight loading
├── preprocessing.py         # Unified signal preprocessing (actipy + impute + ENMO + daily PA)
├── signal_features.py       # Signal-processing features: regularity, sample entropy, PSD
├── gait_detection.py        # ElderNet gait detection, second-mapping, bout merging
├── feature_extraction.py    # Window-level & bout-level feature extraction (DL + SP)
├── aggregate_subjects.py    # Standalone aggregation script (summary statistics)
├── run_pipeline.py          # Feature extraction pipeline (Stages 1-3)
└── README.md                # This file
```

## Architecture

### Feature Extraction Pipeline (run_pipeline.py)
```
┌─────────────────────┐
│    config.yaml      │
└─────────┬───────────┘
          │
┌─────────▼───────────┐     ┌────────────────┐     ┌────────────────────┐
│      Stage 1        │     │    Stage 2     │     │      Stage 3       │
│    Preprocess       │────►│ Gait Detection │────►│ Feature Extraction │
│                     │     │ + Bout Merge   │     │                    │
│ preprocessing.py    │     │ gait_          │     │ feature_           │
│ io_utils.py         │     │  detection.py  │     │  extraction.py     │
└─────────┬───────────┘     └────────────────┘     │ signal_features.py │
          │                                        └─────────┬──────────┘
          │                                                  │
          ▼                                                  ▼
    daily_pa/*.csv                              bouts/*.csv  windows/*.csv
```

### Aggregation (aggregate_subjects.py — separate script)
```
┌─────────────────────────────────────────────────────────────────────┐
│  Reads: bouts/*.csv, windows/*.csv, daily_pa/*.csv                  │
│                                                                     │
│  Computes (NO GPU needed):                                          │
│  - Descriptive stats (median, mean, std, percentiles, IQR, CV)      │
│  - Histogram features (global bins for cross-subject comparison)    │
│  - Variance-of-variance (within-bout gait variability)              │
│  - Daily walking amounts and step counts                            │
│  - Daily physical activity summaries                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                     subject_summary.csv
```

## Module Descriptions

| File | Stage | What it does |
|---|---|---|
| `config.yaml` | — | All settings: sensor device, model weights, signal parameters, paths |
| `config.py` | — | Loads YAML into a dot-access object (`cfg.pipeline.resampled_hz`) |
| `models.py` | — | `Resnet`, `ElderNet`, `LinearLayers` — architectures must match trained weights |
| `io_utils.py` | 1 | `loadmat()`, `extract_raw_data()`, `parse_subject_id()`, `setup_model()` |
| `preprocessing.py` | 1 | `preprocess_subject()` actipy calibration, resampling, imputation, edge trimming. Also `compute_enmo()`, `compute_daily_pa()` |
| `gait_detection.py` | 2 | `run_gait_detection()` window predictions. `merge_bouts()` post-processing. `detect_bouts()` bout boundaries |
| `signal_features.py` | 3 | `calc_regularity()` (autocorrelation), `calc_sample_entropy()` (numba-accelerated), `compute_psd_features()` (Welch PSD) |
| `feature_extraction.py` | 3 | `extract_features_for_subject()` runs all DL models on windows, SP features on full bouts, returns window-level + bout-level rows |
| `run_pipeline.py` | 1-3 | Feature extraction pipeline. Saves bouts/, windows/, daily_pa/ CSVs |
| `aggregate_subjects.py` | 4 | Standalone aggregation. Reads CSVs, computes summary stats, histograms, var-of-var |

## Output Structure

```
{output_path}/
├── bouts/                      # One CSV per subject (bout-level features)
│   ├── 12345678_0.csv          #   bout_id, start_time, duration_sec,
│   └── ...                     #   speed, cadence, gait_length, regularity_sp,
│                               #   entropy, dom_freq, psd_amp, bout_pa_mean, ...
├── windows/                    # One CSV per subject (window-level features)
│   ├── 12345678_0.csv          #   bout_id, window_id, window_start_time,
│   └── ...                     #   speed, cadence, gait_length, regularity_eldernet, ...
├── daily_pa/                   # One CSV per subject (daily physical activity)
│   ├── 12345678_0.csv          #   day, daily_pa_mean, daily_pa_std, tdpa
│   └── ...
└── subject_summary.csv         # One row per subject (all summary statistics)
```

## Usage

```bash
# Step 1: Feature extraction (Stages 1-3) — requires GPU
python run_pipeline.py --config config.yaml

# Step 2: Aggregation (Stage 4) — no GPU needed, fast
python aggregate_subjects.py --output-dir /path/to/output

# Or use config to find output path:
python aggregate_subjects.py --config config.yaml
```

**Workflow benefits:**
- Run feature extraction once (GPU-intensive)
- Iterate on aggregation (add new summary measures, change bin ranges, etc.)
- Re-aggregate without re-processing raw data

## Bug Fixes from Original Code

### 1. Bout Merging — fixed in gait_detection.py
**Bug:** `merge_gait_bouts()` used a for-loop that mutated `bout_starts`/`bout_ends` during iteration. With 3+ close bouts, results depended on iteration order and could produce overlapping bouts.
**Fix:** While-loop that correctly collapses chains of mergeable bouts.

### 2. ElderNet Head Config — fixed in io_utils.py
**Bug:** `setup_model()` did not forward `num_layers_regressor` or `batch_norm` to `ElderNet`. It silently used `num_layers=3` and `use_bn=True` defaults regardless of config.
**Fix:** `setup_model()` now explicitly passes `num_layers` and `use_bn` from config.

### 3. Inconsistent Preprocessing — fixed in preprocessing.py
**Bug:** Main pipeline used `imputeMissing()` + `drop_first_last_days()`. Frequency/entropy scripts used simpler `reindex(method='nearest')`. These produce **different signals**.
**Fix:** Single `preprocessing.py` used everywhere.

### 4. Regularity on Overlapping Windows — fixed in signal_features.py
**Bug:** SP regularity computed on each 10-sec window with 90% overlap. Consecutive windows share 9 seconds of data, producing highly autocorrelated estimates with deflated variance.
**Fix:** SP regularity computed once per bout on the full bout signal.

### 5. Step Count Formula — fixed in feature_extraction.py
**Bug:** `adj_factor = 0.1 * (bout_len - 1) + 1; total = adj_factor * median_steps`. Hardcoded 90% overlap, used median (biased for skewed distributions).
**Fix:** `total_steps = sum(per_window_steps) * (step_len / window_len)`. Works for any overlap, uses sum.

### 6. Var-of-Var Alignment — fixed in aggregation.py
**Bug:** Reconstructed bout-to-window mapping from durations + windowing constants. Failed silently when bout boundaries didn't align with window boundaries.
**Fix:** Uses `bout_id` column directly from window CSV — no reconstruction needed.

## Dependencies

```
numpy scipy pandas torch tqdm pyyaml actipy mat73
# Optional (10x faster entropy):
numba
```

## Important Note on Model Weights

The `setup_model()` fix in `io_utils.py` means `num_layers` is now correctly forwarded to ElderNet. If your trained weights were saved with the old code (which silently used `num_layers=3`), set `num_layers: 3` in `config.yaml` for all models to match. Mismatched architecture will crash at `load_state_dict`. Update only after confirming what each model was trained with.
