# RUSH Gait Analysis Pipeline

**Paper**: ["Continuous Assessment of Daily-Living Gait Using Self-Supervised Learning of Wrist-Worn Accelerometer Data"](https://www.nature.com/articles/s41746-026-02528-2)

## Repository Structure

```
rush_pipeline/
├── extraction/                    # Stages 1–4: raw sensor data → features → summaries
│   ├── config.yaml                #   All settings: paths, model weights, hyperparameters
│   ├── config.py                  #   YAML loader with dot-access
│   ├── models.py                  #   ElderNet / ResNet architectures
│   ├── io_utils.py                #   .mat loading, subject ID parsing, model setup
│   ├── preprocessing.py           #   Signal preprocessing (actipy + impute + ENMO + daily PA)
│   ├── gait_detection.py          #   ElderNet gait detection, bout merging
│   ├── signal_features.py         #   Regularity, sample entropy, PSD
│   ├── feature_extraction.py      #   Window-level & bout-level feature extraction
│   ├── run_pipeline.py            #   Main entry point for Stages 1–3 (requires GPU)
│   ├── aggregate_subjects.py      #   Stage 4: subject-level summary statistics
│   ├── merge_dataset.py           #   Merge gait summaries with clinical data
│   ├── models/                    #   Pre-trained ElderNet model weights (.pt)
│   ├── PIPELINE_OVERVIEW.md       #   Detailed documentation of extraction metrics
│   └── PIPELINE_OVERVIEW.pdf
│
├── modeling/                      # Stage 5: nested CV prediction pipeline
│   ├── prediction_pipeline.py     #   Nested cross-validation: gait → clinical outcomes
│   ├── MODELING_OVERVIEW.md       #   Detailed documentation of modeling methodology
│   └── MODELING_OVERVIEW.pdf
│
├── output/                        # Shared data (CSVs produced by extraction, consumed by modeling)
│   ├── subject_summary.csv        #   One row per subject (all gait summary statistics)
│   ├── merged_gait_clinical_abl.csv   # Gait + clinical data, analytic baseline
│   └── results_*.csv             #   Prediction results
│
├── eda_gait_clinical.ipynb        # Exploratory data analysis notebook
├── eda_pipeline_outputs.ipynb     # Pipeline output inspection notebook
├── make_pdf.py                    # Convert Markdown docs to styled PDFs
└── README.md                      # This file
```

## Pipeline Overview

```
Raw accelerometer data (.mat)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  EXTRACTION (extraction/)                                    │
│                                                              │
│  Stage 1: Preprocessing — calibrate, resample, impute        │
│  Stage 2: Gait Detection — ElderNet walking bout detection   │
│  Stage 3: Feature Extraction — speed, cadence, stride, etc.  │
│  Stage 4: Aggregation — participant-level summary statistics  │
│           Merge — combine with clinical/demographic data      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                     subject_summary.csv
                     merged_gait_clinical_abl.csv
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  MODELING (modeling/)                                        │
│                                                              │
│  Nested cross-validation prediction pipeline                 │
│  • Binary: parkinsonism, mobility disability, falls,         │
│            cognitive impairment                               │
│  • Continuous: parkinsonism score, motor composite,           │
│                global cognition                               │
│  • Models: Logistic Regression, Random Forest, XGBoost       │
│  • Feature selection: KBest, MI, L1, Consensus, PCA          │
│  • Statistical comparison of feature sets (Nadeau & Bengio)  │
└─────────────────────────────────────────────────────────────┘
```

## Usage

```bash
# Step 1: Feature extraction (Stages 1–3) — requires GPU
cd extraction
python run_pipeline.py --config config.yaml

# Step 2: Aggregation (Stage 4) — no GPU needed
python aggregate_subjects.py --output-dir ../output

# Step 3: Merge with clinical data
python merge_dataset.py

# Step 4: Prediction modeling — no GPU needed
cd ..
python modeling/prediction_pipeline.py

# Optional: generate PDF documentation
python make_pdf.py
```

## Documentation

- **[Extraction & Aggregation](extraction/PIPELINE_OVERVIEW.md)** — Detailed description of gait detection, feature extraction, and all summary statistics
- **[Prediction Modeling](modeling/MODELING_OVERVIEW.md)** — Nested CV methodology, feature selection strategies, models, evaluation metrics, and statistical comparisons

## Dependencies

```
# Extraction (GPU required)
numpy scipy pandas torch tqdm pyyaml actipy mat73
numba  # optional, 10x faster entropy

# Modeling (CPU only)
numpy scipy pandas scikit-learn xgboost
```

## Citation

If you use this code or ElderNet in your research, please cite:

```bibtex
@article{Brand2026Gait,
  title={Continuous Assessment of Daily-Living Gait Using Self-Supervised Learning of Wrist-Worn Accelerometer Data},
  author={Brand, Yonatan E and Buchman, Aron S and Kluge, Felix and Palmerini, Luca and Becker, Clemens and Cereatti, Andrea and Maetzler, Walter and Vereijken, Beatrix and Yarnall, Alison J and Rochester, Lynn and Del Din, Silvia and Mueller, Arne and Hausdorff, Jeffrey M and Perlman, Or},
  journal={npj Digital Medicine},
  year={2026},
  doi={10.1038/s41746-026-02528-2},
  url={https://www.nature.com/articles/s41746-026-02528-2}
}
```

---

## License

University of Oxford Academic Use License. See [LICENSE.md](LICENSE.md) for details.

Based on [ssl-wearables](https://github.com/OxWearables/ssl-wearables) from OxWearables.
