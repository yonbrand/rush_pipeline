# Prediction Modeling Pipeline — Overview

---

## What Does This Pipeline Do?

This pipeline takes the **participant-level gait, physical activity, and sleep summary statistics** produced by the extraction pipeline and uses them to **predict clinical outcomes** — motor impairment, falls, and cognitive decline — via machine learning models with rigorous cross-validation.

The goal is to determine whether daily-living gait metrics captured by a wrist-worn accelerometer carry predictive information about a person's clinical status, and whether they add value beyond a simple in-clinic gait speed test.

---

## Input

A single CSV file (`outputs/tables/merged_gait_clinical_abl.csv`) containing one row per participant at their analytic baseline visit. Columns include:

| Column group | Examples | Source |
|---|---|---|
| **Gait bout metrics** | `bout_speed_median`, `bout_cadence_cv`, `stability_speed_mean_wasserstein`, ... | Extraction pipeline |
| **Daily physical activity** | `daily_pa_mean_median`, `daily_pa_std_mean`, `tdpa_median`, ... | Extraction pipeline |
| **Sleep features** | `sleep_tst_hours_median`, `sleep_efficiency_mean`, `rar_is`, `rar_iv`, ... | Extraction pipeline |
| **Clinical gait speed** | `gait_speed` (8-foot walk test, m/s) | Clinical assessment |
| **Demographics** | `age_at_visit`, `msex`, `educ` | Clinical dataset |
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

Each model is evaluated with seven feature input configurations to test where predictive information comes from:

### Sensor-based feature sets (full selection strategy sweep)

| Feature set | What it contains | Purpose |
|---|---|---|
| **Gait Bout** | ~100+ summary statistics from daily-living walking bouts | Core question: does free-living gait predict clinical status? |
| **Daily PA** | Daily physical activity intensity and volume metrics | Does overall activity level predict outcomes? |
| **Sleep** | HDCZA nightly sleep summary stats + rest-activity rhythm metrics (IS, IV, L5, M10, RA, SRI) | Do sleep and circadian patterns predict outcomes? |
| **Daily PA + Sleep** | Daily PA + Sleep features together | Does combining activity and sleep help? |
| **Combined** | Gait bout + Daily PA features together | Does combining gait quality and activity quantity help? |

### Clinical baseline & augmented sets (both classification and regression)

| Feature set | What it contains | Purpose |
|---|---|---|
| **8ft Gait Speed (baseline)** | Single clinical gait speed measure + demographics | Benchmark: how much can a simple clinic test predict? |
| **Combined + 8ft Speed** | All gait + PA features + clinical gait speed | Does adding clinical gait to sensor data improve prediction? |

The 8-foot gait speed baseline provides the key comparison: if sensor-based metrics outperform or meaningfully augment a simple clinical walk test, that supports the clinical utility of continuous wearable monitoring.

**Note:** `prob_bin_*` histogram features (exact duplicates of `freq_bin_*` after normalisation) are automatically dropped from the gait bout feature set to avoid inflating the feature space.

**Circularity note:** The 8-foot walk test may contribute to the computation of `parksc` (parkinsonism score) and `motor10` (motor composite). Results for those outcomes should be interpreted with caution. The `cogn_global` comparison is free of this concern.

---

## Models

Three model families are evaluated, each with hyperparameters tuned via an inner cross-validation loop:

### Classification

| Model | Tuned hyperparameters |
|---|---|
| **Logistic Regression** | Regularisation strength `C` in {0.01, 0.1, 1.0, 10.0} |
| **Random Forest** | `n_estimators` in {100, 200}, `max_depth` in {4, 8, None} |
| **XGBoost** | `n_estimators` in {100, 200}, `max_depth` in {3, 5}, `learning_rate` in {0.01, 0.05, 0.1} |

### Regression

| Model | Tuned hyperparameters |
|---|---|
| **ElasticNet** | `alpha` in {0.01, 0.1, 0.5, 1.0}, `l1_ratio` in {0.3, 0.5, 0.7} |
| **Random Forest** | `n_estimators` in {100, 200}, `max_depth` in {4, 8, None} |
| **XGBoost** | `n_estimators` in {100, 200}, `max_depth` in {3, 5}, `learning_rate` in {0.01, 0.05, 0.1} |

All models include class weighting (`class_weight='balanced'`) or appropriate regularisation for imbalanced outcomes.

---

## Feature Selection Strategies

Ten strategies are evaluated independently, each applied **inside** the cross-validation loop to prevent data leakage:

| Strategy | Description | Tuned inside CV? |
|---|---|---|
| **No Selection** | All features passed to the model; rely on the model's own regularisation | No |
| **SelectKBest** | Univariate ANOVA F-test (classification) or f-regression; top *k* features retained | Yes — `k` in {10, 20, 30} |
| **Mutual Information** | MI-based univariate scoring; top *k* features retained | Yes — `k` in {10, 20, 30} |
| **L1-based** | Fit an L1-penalised model (Lasso / L1-Logistic Regression); keep features with non-zero coefficients | No |
| **Consensus** | Run SelectKBest, MI, and L1 independently; keep features selected by >= 2 of the 3 methods | No |
| **PCA** | Principal Component Analysis retaining 95% of variance | No |
| **Stability** | Stability Selection (Meinshausen & Buhlmann, 2010): L1-penalised models on 50 random sub-samples; retain features selected in >= 60% of runs. Controls false-discovery rate better than a single L1 fit. | Yes — `threshold` tuned |
| **mRMR** | Minimum Redundancy Maximum Relevance (Peng, Long & Ding, 2005): greedy forward selection maximising MI with target minus mean squared correlation with already-selected features | Yes — `k` in {10, 15, 20, 25} |
| **Block PCA** | Domain-grouped PCA: gait features are assigned to biomechanical domains (pace, cadence, regularity, spectral/complexity, quantity, etc.) and PCA is applied *independently within each domain*, then concatenated. Preserves domain structure while reducing dimensionality. | Yes — `variance_retained` in {0.70, 0.80, 0.90, 0.95} |
| **Block PCA + Stability** | Block PCA followed by Stability Selection on the domain-level components. Two-stage approach: first reduce within-domain redundancy, then select the most stable cross-domain signals. | Yes — both `variance_retained` and `threshold` tuned |

For the **8ft Gait Speed (baseline)** feature set (only 4 features including demographics), only "No Selection" is used since feature selection on so few variables is meaningless.

### Block PCA domain structure

Features are assigned to biomechanical domains based on column-name prefixes:

| Domain | Prefixes | Captures |
|---|---|---|
| Speed | `bout_speed` | Walking pace |
| Gait Length | `bout_gait_length` | Stride length (DL + indirect) |
| Cadence | `bout_cadence` | Step frequency |
| Regularity | `bout_regularity_eldernet`, `bout_regularity_sp` | Stride-to-stride consistency |
| Gait Quantity | `bout_duration`, `bout_total`, `daily_n_bouts`, `daily_step`, `daily_walking`, `n_bouts` | How much walking |
| Bout Intensity | `bout_pa_amplitude`, `bout_pa_variability` | Movement vigor |
| Within-Bout Var | `var_var` | Stride-to-stride fluctuation |
| Spectral/Complexity | `bout_entropy`, `bout_dom`, `bout_psd_amp`, `bout_psd_width`, `bout_psd_slope` | Signal complexity |
| Day-to-Day | `stability_`, `dist_` | Between-day consistency |
| Temporal Pattern | `tod_` | Time-of-day walking distribution |

---

## Methodology — Nested Cross-Validation

All results are produced using **nested cross-validation**, which provides unbiased estimates of out-of-sample performance while still allowing hyperparameter tuning:

```
Outer loop — performance estimation
  5-fold x 3 repeats = 15 evaluation folds
  Each fold holds out 20% of data for testing
  +----------------------------------------------+
  |  Inner loop — hyperparameter tuning           |
  |  3-fold GridSearchCV on the 80% training data |
  |  Selects best hyperparameters + feature k     |
  +----------------------------------------------+
  -> Evaluate on held-out 20% test fold
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
| **CorrelationFilter** | Drops one of each pair with \|r\| > 0.85 | Correlation structure must be estimated from training data only |
| **StandardScaler** | Centres and scales to unit variance | Mean and SD must come from training data only |

Note: Block PCA strategies use their own internal preprocessing (per-domain imputation + scaling + PCA) instead of the standard preprocessing steps above.

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
| **R^2** | Proportion of variance explained (primary metric) |
| **MAE** | Mean absolute error |

All metrics are reported as **mean +/- SD across 15 outer folds** (5-fold x 3 repeats).

---

## Statistical Comparison of Feature Sets

To determine whether one feature set provides significantly better predictions than another, the pipeline uses the **corrected repeated k-fold CV t-test** (Nadeau & Bengio, 2003). Comparisons are performed for **both classification (AP) and regression (R^2)**.

### Why not a standard paired t-test?

In repeated cross-validation, the fold scores are not independent: different folds share most of their training data. A naive paired t-test ignores this overlap and produces p-values that are too small (anticonservative). The Nadeau-Bengio correction adjusts the variance estimate:

```
corrected variance = (1/(k x r) + 1/(k - 1)) x s^2
```

where *k* = number of folds, *r* = number of repeats, and *s^2* is the sample variance of the paired score differences.

### How it works

1. For each outcome, the **best model configuration** (model x selection strategy) is identified per feature set based on the primary metric.
2. Since all feature sets use the same outer CV random seed, the 15 fold scores are **paired** (same participants in each fold).
3. The corrected t-test is applied to all pairwise combinations, producing:
   - The mean difference in the metric
   - A 95% confidence interval
   - A two-sided p-value

### Interpretation

| Comparison | What it tells you |
|---|---|
| Gait Bout vs. 8ft Speed | Does daily-living gait monitoring predict better than a simple clinic walk test? |
| Combined vs. 8ft Speed | Do all sensor features together outperform the clinical benchmark? |
| Combined + 8ft Speed vs. Combined | Does adding clinical gait speed to sensor features improve prediction? |
| Combined vs. Gait Bout | Does adding physical activity metrics to gait help? |
| Sleep vs. Daily PA | Do sleep metrics carry different information than activity levels? |

---

## Visualisations

For each outcome, the pipeline generates diagnostic plots from the best model configuration per feature set:

### Classification
- **ROC curves** — one curve per feature set overlaid, with AUC in the legend.
- **Precision-Recall curves** — one curve per feature set overlaid, with AP in the legend.

### Regression
- **Predicted vs. Observed scatter plots** — out-of-fold predictions against true values, with identity line.

### Feature importance
- **Permutation importance** — for each best (outcome x feature set) configuration, the pipeline fits on a 75% train split and computes permutation importance on the 25% holdout. The top 30 features by importance are recorded per configuration.

---

## Output Files

| File | Description |
|---|---|
| `outputs/tables/results_classification_nested_cv.csv` | Full results for all binary outcomes — one row per (outcome x feature set x selection x model) |
| `outputs/tables/results_regression_nested_cv.csv` | Full results for all continuous outcomes |
| `outputs/tables/feature_set_comparisons_classification.csv` | Pairwise statistical comparisons between feature sets for classification (delta AP, 95% CI, p-value) |
| `outputs/tables/feature_set_comparisons_regression.csv` | Pairwise statistical comparisons between feature sets for regression (delta R^2, 95% CI, p-value) |
| `outputs/tables/feature_importance_classification.csv` | Permutation importance for top 30 features per best classification configuration |
| `outputs/tables/feature_importance_regression.csv` | Permutation importance for top 30 features per best regression configuration |
| `outputs/figures/roc_<outcome>.png` | ROC curves per classification outcome |
| `outputs/figures/pr_<outcome>.png` | Precision-recall curves per classification outcome |
| `outputs/figures/scatter_<outcome>.png` | Predicted vs. observed scatter plots per regression outcome |

---

## Usage

```bash
# From the repository root:
python modeling/prediction_pipeline.py
```

The script reads from `outputs/tables/merged_gait_clinical_abl.csv` (produced by the extraction and aggregation pipeline) and writes results back to `outputs/tables/` and `outputs/figures/`.

---

## References

- Meinshausen, N., & Buhlmann, P. (2010). Stability Selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473.
- Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. *Machine Learning*, 52(3), 239-281.
- Peng, H., Long, F., & Ding, C. (2005). Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226-1238.
