# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wrist-worn accelerometer gait analysis pipeline for the Rush Memory and Aging Project (MAP) / Religious Orders Study (ROS). Published in Nature npj Digital Medicine (2026). Two-phase architecture: **extraction** (raw sensor → gait features) and **modeling** (features → clinical outcome prediction).

## Commands

```bash
# Extraction (Stages 1-3, requires GPU + CUDA)
cd extraction && python run_pipeline.py --config config.yaml

# Aggregation (Stage 4, CPU)
cd extraction && python aggregate_subjects.py --output-dir ../output

# Merge gait summaries with clinical/demographic data
cd extraction && python merge_dataset.py

# Prediction modeling (Stage 5, CPU — runs ~30-60 min)
python modeling/prediction_pipeline.py

# Generate PDF documentation from Markdown
python make_pdf.py
```

No test suite, no linter config, no CI/CD. No requirements.txt — dependencies are: `numpy scipy pandas torch tqdm pyyaml actipy mat73` (extraction, GPU), `numpy scipy pandas scikit-learn xgboost` (modeling, CPU), `markdown xhtml2pdf` (PDF generation). Optional: `numba` (10x faster entropy), `lmoments3` (L-moments stats), `diptest` (multimodality test). Model weights (`.pt` files) tracked via **git-lfs**.

## Architecture

### Extraction (`extraction/`)

Five-stage pipeline processing raw `.mat` accelerometer files into participant-level gait summaries.

**Import chain:** `run_pipeline.py` → `config.py`, `io_utils.py` → `models.py` (ElderNet/ResNet), `preprocessing.py`, `gait_detection.py`, `feature_extraction.py` → `signal_features.py`

- `config.yaml` controls all paths (model weights, data dirs) and hyperparameters (window size, overlap, bout merging gaps, wear-time thresholds)
- `models/` contains 6 pretrained `.pt` weights; `models.py` architecture must **exactly match training code** or pretrained weights silently produce garbage
- `aggregate_subjects.py`, `merge_dataset.py` are standalone scripts (no local imports) — safe to modify independently
- `aggregation.py` is legacy, superseded by `aggregate_subjects.py`

**Preprocessing order matters:** actipy calibration → nonwear detection → resample → validate wear time (before imputation, so validation sees real data) → impute missing via time-of-day matching → drop first/last partial days.

**Signal processing algorithms:**
- **Stride regularity** (`signal_features.py`): Autocorrelation on the FULL BOUT signal, not per-window. The `regularity_sp` (signal-processing) metric is methodologically distinct from `regularity_eldernet` (DL model, median across windows).
- **Sample entropy**: O(N²) pure Python with optional **numba JIT** — warns at runtime if numba is missing.
- **Step counting** (`feature_extraction.py`): Three-step method accounting for 90% window overlap: per-window steps → steps-per-second → multiply by actual bout duration. Prevents counting the same steps ~10x.
- **Static window filtering** (`gait_detection.py`): Otsu's method filters low-variance windows before the gait model. Bout merging uses a while-loop (not for-loop) to handle chained mergeable bouts.

**Subject ID parsing** (`io_utils.py`): Tries Axivity format (202X prefix, preserves leading zeros) first, then GENEActive format (strips leading zeros via int() cast).

### Modeling (`modeling/`)

Single standalone script `prediction_pipeline.py` (no local imports). Reads `output/merged_gait_clinical_abl.csv`, writes results back to `output/`.

**Key design decisions:**
- **Nested CV**: outer 5×3 repeated K-fold (performance estimation), inner 3-fold GridSearchCV (hyperparameter tuning). All preprocessing inside sklearn Pipeline to prevent data leakage.
- **Custom sklearn transformers**: `MissingRateFilter`, `CorrelationFilter`, `ConsensusSelector` — must implement `fit`/`transform` and work with numpy arrays (not DataFrames).
- **Feature sets**: Gait Bout, Daily PA, Combined, plus regression-only sets (8ft Gait Speed baseline, Combined + 8ft Speed). The baseline uses only "No Selection" strategy (triggered when feature count ≤ 3).
- **Statistical comparison**: Nadeau & Bengio (2003) corrected repeated CV t-test for pairwise feature-set comparison. Per-fold scores stored in `_fold_R2`/`_fold_AP` keys (stripped before CSV save).
- Paths resolve via `_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` — works regardless of CWD.
- `xgboost` is optional — models using it are gracefully skipped if not installed.

### Merge script (`extraction/merge_dataset.py`)

Standalone with **hardcoded paths**. Reads clinical data from `wrist_sensor_metadata_Yonatan.xlsx` (not in git, in .gitignore). Toggle flags (`EXPORT_ABL`, `EXPORT_LV`, `EXPORT_ALL_VISITS`, `EXPORT_POSTMORTEM`) control which of 4 output CSVs are produced. Binary outcome derivation happens inside this script.

### Data flow

```
Raw .mat files → extraction/run_pipeline.py → output/bouts/*.csv, output/windows/*.csv
  → extraction/aggregate_subjects.py → output/subject_summary.csv
  → extraction/merge_dataset.py → output/merged_gait_clinical_abl.csv
  → modeling/prediction_pipeline.py → output/results_*_nested_cv.csv
```

### Documentation

`PIPELINE_OVERVIEW.md` (extraction) and `MODELING_OVERVIEW.md` (modeling) detail methodology. Both convert to styled PDFs via `make_pdf.py`.

## Key variables

- `gait_bout_cols`: derived by exclusion — everything in the DataFrame that isn't in `exclude_from_features`, `daily_pa_cols`, or `id_cols`. When adding new clinical/outcome columns to the merge, they **must** be added to `exclude_from_features` or they'll leak into gait features.
- `gait_speed`: the 8-foot walk test (clinical measure), intentionally excluded from sensor feature sets but included as a regression baseline. **Circularity warning:** may contribute to `parksc`/`motor10` score computation — interpret those outcomes with caution.
- Demographic covariates (`age_bl`, `msex`, `educ`) are appended to all feature sets automatically via `prepare_data(demographics=True)`.

## Aggregation statistics

`aggregate_subjects.py` computes 200+ columns per subject including robust statistics (MAD, L-skewness, L-kurtosis), distribution shape (Hartigan's Dip, Shannon entropy, Wasserstein distance to Gaussian), day-to-day stability (pairwise Wasserstein distances, requires ≥30 bouts), within-bout variability, histogram features (10 global bins with fixed ranges for cross-subject comparability), and time-of-day walking patterns.
