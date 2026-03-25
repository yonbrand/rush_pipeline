# Prediction Modeling Pipeline — Overview

---

## What Does This Pipeline Do?

This pipeline takes the **participant-level gait and physical activity summary statistics** produced by the extraction pipeline and uses them to **predict clinical outcomes** — motor impairment, falls, and cognitive decline — via machine learning models with rigorous cross-validation.

The goal is to determine whether daily-living gait metrics captured by a wrist-worn accelerometer carry predictive information about a person's clinical status, and whether they add value beyond a simple in-clinic gait speed test.

---

## Input

A single CSV file (`output/merged_gait_clinical_abl.csv`) containing one row per participant at their analytic baseline visit. Columns include:

| Column group | Examples | Source |
|---|---|---|
| **Gait bout metrics** | `bout_speed_median`, `bout_cadence_cv`, `stability_speed_mean_wasserstein`, ... | Extraction pipeline |
| **Daily physical activity** | `daily_pa_mean_median`, `daily_pa_std_mean`, `tdpa_median`, ... | Extraction pipeline |
| **Clinical gait speed** | `gait_speed` (8-foot walk test, m/s) | Clinical assessment |
| **Demographics** | `age_bl`, `msex`, `educ` | Clinical dataset |
| **Clinical outcomes** | `parkinsonism_yn`, `falls_binary`, `cogn_global`, ... | Clinical dataset |

---

## Outcomes

### Binary (classification)

| Variable | Description |
|---|---|
| `parkinsonism_yn` | Clinical parkinsonism diagnosis (yes/no) |
| `mobility_disability_binary` | Self-reported mobility disability |
| `falls_binary` | Any fall in the past year |
| `cognitive_impairment` | Mild cognitive impairment or dementia |

### Continuous (regression)

| Variable | Description |
|---|---|
| `parksc` | Composite parkinsonism score (higher = more signs) |
| `motor10` | Motor composite score |
| `cogn_global` | Global cognition composite (higher = better) |

---

## Feature Sets

Each model is evaluated with five feature input configurations to test where predictive information comes from:

| Feature set | What it contains | Purpose |
|---|---|---|
| **Gait Bout** | ~100+ summary statistics from daily-living walking bouts | Core question: does free-living gait predict clinical status? |
| **Daily PA** | Daily physical activity intensity and volume metrics | Does overall activity level predict outcomes? |
| **Combined** | Gait bout + Daily PA features together | Does combining gait quality and activity quantity help? |
| **8ft Gait Speed (baseline)** | Single clinical gait speed measure + demographics | Benchmark: how much can a simple clinic test predict? |
| **Combined + 8ft Speed** | All sensor features + clinical gait speed | Does adding clinical gait to sensor data improve prediction? |

The last two feature sets are evaluated for **regression outcomes only**. The 8-foot gait speed baseline provides the key comparison: if sensor-based metrics outperform or meaningfully augment a simple clinical walk test, that supports the clinical utility of continuous wearable monitoring.

**Circularity note:** The 8-foot walk test may contribute to the computation of `parksc` (parkinsonism score) and `motor10` (motor composite). Results for those outcomes should be interpreted with caution. The `cogn_global` comparison is free of this concern.

---

## Models

Three model families are evaluated, each with hyperparameters tuned via an inner cross-validation loop:

### Classification

| Model | Tuned hyperparameters |
|---|---|
| **Logistic Regression** | Regularisation strength `C` ∈ {0.01, 0.1, 1.0, 10.0} |
| **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {4, 8, None} |
| **XGBoost** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {3, 5}, `learning_rate` ∈ {0.01, 0.05, 0.1} |

### Regression

| Model | Tuned hyperparameters |
|---|---|
| **ElasticNet** | `alpha` ∈ {0.01, 0.1, 0.5, 1.0}, `l1_ratio` ∈ {0.3, 0.5, 0.7} |
| **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {4, 8, None} |
| **XGBoost** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {3, 5}, `learning_rate` ∈ {0.01, 0.05, 0.1} |

All models include class weighting (`class_weight='balanced'`) or appropriate regularisation for imbalanced outcomes.

---

## Feature Selection Strategies

Six strategies are evaluated independently, each applied **inside** the cross-validation loop to prevent data leakage:

| Strategy | Description | Tuned inside CV? |
|---|---|---|
| **No Selection** | All features passed to the model; rely on the model's own regularisation | No |
| **SelectKBest** | Univariate ANOVA F-test (classification) or f-regression; top *k* features retained | Yes — `k` ∈ {10, 20, 30} |
| **Mutual Information** | MI-based univariate scoring; top *k* features retained | Yes — `k` ∈ {10, 20, 30} |
| **L1-based** | Fit an L1-penalised model (Lasso / L1-Logistic Regression); keep features with non-zero coefficients | No |
| **Consensus** | Run SelectKBest, MI, and L1 independently; keep features selected by at least 2 of the 3 methods | No |
| **PCA** | Principal Component Analysis retaining 95% of variance | No |

For the **8ft Gait Speed (baseline)** feature set (only 4 features including demographics), only "No Selection" is used since feature selection on so few variables is meaningless.

---

## Methodology — Nested Cross-Validation

All results are produced using **nested cross-validation**, which provides unbiased estimates of out-of-sample performance while still allowing hyperparameter tuning:

```
Outer loop — performance estimation
  5-fold × 3 repeats = 15 evaluation folds
  Each fold holds out 20% of data for testing
  ┌──────────────────────────────────────────────┐
  │  Inner loop — hyperparameter tuning           │
  │  3-fold GridSearchCV on the 80% training data │
  │  Selects best hyperparameters + feature k     │
  └──────────────────────────────────────────────┘
  → Evaluate on held-out 20% test fold
```

### Why nested CV?

A common methodological error in clinical prediction studies is to tune hyperparameters (or select features) using the same data that is later used to evaluate performance. This leads to **optimistically biased** results — the model appears to generalise better than it actually does.

Nested CV eliminates this bias:
- The **inner loop** selects the best hyperparameters using only training data.
- The **outer loop** evaluates performance on data that was never seen during training *or* tuning.

### Leak-free preprocessing

All data-dependent preprocessing steps are implemented as **sklearn-compatible transformers** and executed inside the cross-validation pipeline. This means they are fit only on training data in each fold:

| Step | What it does | Why it matters |
|---|---|---|
| **MissingRateFilter** | Drops columns with > 60% missing values | Missing rates may differ between train/test in a leaky setup |
| **SimpleImputer** | Fills remaining NaN values with column medians | Medians must come from training data only |
| **VarianceThreshold** | Removes near-zero-variance columns | Variance estimate must come from training data only |
| **CorrelationFilter** | Drops one of each pair with \|r\| > 0.95 | Correlation structure must be estimated from training data only |
| **StandardScaler** | Centres and scales to unit variance | Mean and SD must come from training data only |

---

## Evaluation Metrics

### Classification

| Metric | Description | Why included |
|---|---|---|
| **Average Precision (AP)** | Area under the precision-recall curve | Primary metric — robust to class imbalance |
| **AUC** | Area under the ROC curve | Standard comparability |
| **Balanced Accuracy** | Mean of sensitivity and specificity | Accounts for imbalanced classes |
| **F1** | Harmonic mean of precision and recall | Emphasises the positive (minority) class |

### Regression

| Metric | Description |
|---|---|
| **R²** | Proportion of variance explained (primary metric) |
| **MAE** | Mean absolute error |

All metrics are reported as **mean ± SD across 15 outer folds** (5-fold × 3 repeats).

---

## Statistical Comparison of Feature Sets

To determine whether one feature set provides significantly better predictions than another, the pipeline uses the **corrected repeated k-fold CV t-test** (Nadeau & Bengio, 2003).

### Why not a standard paired t-test?

In repeated cross-validation, the fold scores are not independent: different folds share most of their training data. A naive paired t-test ignores this overlap and produces p-values that are too small (anticonservative). The Nadeau-Bengio correction adjusts the variance estimate:

```
corrected variance = (1/(k × r) + 1/(k − 1)) × s²
```

where *k* = number of folds, *r* = number of repeats, and *s²* is the sample variance of the paired score differences.

### How it works

1. For each outcome, the **best model configuration** (model × selection strategy) is identified per feature set based on mean R².
2. Since all feature sets use the same outer CV random seed, the 15 fold scores are **paired** (same participants in each fold).
3. The corrected t-test is applied to all pairwise combinations, producing:
   - The mean difference in R² (ΔR²)
   - A 95% confidence interval
   - A two-sided p-value

### Interpretation

| Comparison | What it tells you |
|---|---|
| Gait Bout vs. 8ft Speed | Does daily-living gait monitoring predict better than a simple clinic walk test? |
| Combined vs. 8ft Speed | Do all sensor features together outperform the clinical benchmark? |
| Combined + 8ft Speed vs. Combined | Does adding clinical gait speed to sensor features improve prediction? |
| Combined vs. Gait Bout | Does adding physical activity metrics to gait help? |

---

## Output Files

| File | Description |
|---|---|
| `output/results_classification_nested_cv.csv` | Full results for all binary outcomes — one row per (outcome × feature set × selection × model) |
| `output/results_regression_nested_cv.csv` | Full results for all continuous outcomes |
| `output/feature_set_comparisons_regression.csv` | Pairwise statistical comparisons between feature sets (ΔR², 95% CI, p-value) |

---

## Usage

```bash
# From the repository root:
python modeling/prediction_pipeline.py
```

The script reads from `output/merged_gait_clinical_abl.csv` (produced by the extraction and aggregation pipeline) and writes results back to `output/`.

---

## References

- Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. *Machine Learning*, 52(3), 239–281.
