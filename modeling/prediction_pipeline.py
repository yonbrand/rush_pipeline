"""
Nested cross-validation prediction pipeline: gait metrics → clinical outcomes.
Uses ABL (analytic baseline) data only — one visit per subject.

Methodology (designed for peer-reviewed publication):
  - ALL preprocessing (missing-rate filtering, imputation, variance
    thresholding, correlation pruning, scaling) and feature selection
    are fitted INSIDE each CV fold to eliminate data leakage.
  - Nested CV architecture:
      Outer loop (5-fold × 3 repeats): unbiased generalisation estimates
      Inner loop (5-fold GridSearchCV): joint hyperparameter + feature-
      selection tuning (where applicable)
  - Results report mean ± SD across outer folds.
  - Most-common inner-CV best parameters are recorded per configuration.

Outcomes:
  Binary:     parkinsonism_yn, mobility_disability_binary, falls_binary,
              cognitive_impairment
  Continuous: parksc, motor10, cogn_global

Models (tuned via inner CV):
  Classification: Logistic Regression, Random Forest, XGBoost
  Regression:     ElasticNet, Random Forest, XGBoost

Feature selection strategies (fitted inside CV):
  1. No Selection      – rely on model regularisation
  2. SelectKBest       – univariate ANOVA / f_regression (k tuned)
  3. Mutual Information – MI-based univariate (k tuned)
  4. L1-based          – SelectFromModel with L1-penalised estimator
  5. Consensus         – features picked by ≥ 2 of {KBest, MI, L1}
  6. PCA               – 95 % variance retained

Feature buckets (evaluated independently):
  1. Gait bout metrics
  2. Daily physical-activity (PA) variables
  3. Combined (gait + daily PA)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import warnings
from collections import Counter
from itertools import combinations

# Resolve paths relative to the repository root (one level up from modeling/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")

warnings.filterwarnings("ignore")

from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RepeatedKFold,
    StratifiedKFold, KFold,
    cross_validate, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest,
    f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    SelectFromModel,
)
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost models")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading & feature-bucket definitions
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(os.path.join(_OUTPUT_DIR, "merged_gait_clinical_abl.csv"))
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

id_cols = ["sub_id", "projid", "fu_year", "wear_days", "study"]
daily_pa_cols = [
    c for c in df.columns
    if c.startswith("daily_pa_mean_") or c.startswith("daily_pa_std_")
    or c.startswith("tdpa_")
]

exclude_from_features = set(id_cols + daily_pa_cols + [
    "age_bl", "msex", "educ", "race7", "study", "device", "sub_id",
    "parkinsonism_yn", "dcfdx", "dementia", "cpd_ever", "cogdx",
    "rosbsum", "falls", "falls_binary", "mobility_disability_binary",
    "cognitive_impairment",
    "cogn_global", "cogn_ep", "cogn_po", "cogn_ps", "cogn_se", "cogn_wo",
    "cogng_demog_slope", "time_lastce2dod", "age_death",
    "cts_mmse30", "parksc", "motor10", "gait_speed", "scaled_to",
    "pert_noact_avg", "tactivity_acth_avgnew", "tactivity_d_avgnew",
    "gaitsc", "bradysc", "rigidsc", "tremsc",
    "r_depres", "bmi", "med_con_sum_cum", "vasc_risks_sum",
    "iadlsum", "katzsum", "late_life_cogact_freq", "phys5itemsum",
    "late_life_soc_act", "park_rx",
    "motor10_demog_slope", "sqrt_parksc_demog_slope",
    "motor_dexterity", "motor_gait", "motor_handstreng",
    "is", "iv", "kar", "kra", "age_at_visit",
])
gait_bout_cols = [
    c for c in df.columns
    if c not in exclude_from_features
    and not c.startswith("daily_pa_mean_")
    and not c.startswith("daily_pa_std_")
    and not c.startswith("tdpa_")
]
demographic_cols = ["age_at_visit", "msex", "educ"]

print(f"Gait bout features: {len(gait_bout_cols)}")
print(f"Daily PA features:  {len(daily_pa_cols)}")

# ── Outcome definitions ──────────────────────────────────────────────────────
BINARY_OUTCOMES = {
    "parkinsonism_yn": "Parkinsonism",
    "mobility_disability_binary": "Mobility Disability",
    "falls_binary": "Falls",
    "cognitive_impairment": "Cognitive Impairment",
}
CONTINUOUS_OUTCOMES = {
    "parksc": "Parkinsonism Score",
    "motor10": "Motor Composite",
    "cogn_global": "Global Cognition",
}

FEATURE_SETS = {
    "Gait Bout": gait_bout_cols,
    "Daily PA": daily_pa_cols,
    "Combined": gait_bout_cols + daily_pa_cols,
}

# Regression-only feature sets: clinical gait speed baseline & augmented combined
# NOTE: gait_speed (8-foot walk test) may partially overlap with parksc/motor10
#       scoring — interpret those outcomes with caution (see limitations).
REGRESSION_EXTRA_FEATURE_SETS = {
    "8ft Gait Speed (baseline)": ["gait_speed"],
    "Combined + 8ft Speed": gait_bout_cols + daily_pa_cols + ["gait_speed"],
}

SELECTION_STRATEGIES = [
    "No Selection",
    "SelectKBest",
    "Mutual Info",
    "L1-based",
    "Consensus",
    "PCA",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Custom sklearn-compatible transformers (leak-free preprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

class MissingRateFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate exceeds *threshold*. Fit on train fold only."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xf = np.asarray(X, dtype=float)
        miss = np.isnan(Xf).mean(axis=0)
        self.keep_mask_ = miss < self.threshold
        if not self.keep_mask_.any():
            self.keep_mask_[0] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop features with pairwise |r| > *threshold* (keep the earlier column)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(corr, 0.0)
        n = X.shape[1]
        drop = set()
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
        self.keep_idx_ = sorted(set(range(n)) - drop)
        if not self.keep_idx_:
            self.keep_idx_ = [0]
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]


class ConsensusSelector(BaseEstimator, TransformerMixin):
    """Select features voted by ≥ *min_votes* of {SelectKBest, MI, L1}."""

    def __init__(self, task_type="classification", k=30, min_votes=2):
        self.task_type = task_type
        self.k = k
        self.min_votes = min_votes

    def fit(self, X, y):
        p = X.shape[1]
        k = min(self.k, p)

        score_func = f_classif if self.task_type == "classification" else f_regression
        skb_mask = SelectKBest(score_func, k=k).fit(X, y).get_support()

        mi_func = _mi_classif if self.task_type == "classification" else _mi_regression
        mi_scores = mi_func(X, y)
        mi_mask = np.zeros(p, dtype=bool)
        mi_mask[np.argsort(mi_scores)[-k:]] = True

        if self.task_type == "classification":
            l1_est = LogisticRegression(
                penalty="l1", solver="saga", C=0.1,
                max_iter=5000, class_weight="balanced", random_state=42)
        else:
            l1_est = Lasso(alpha=0.1, max_iter=5000, random_state=42)
        l1_mask = SelectFromModel(l1_est).fit(X, y).get_support()

        votes = skb_mask.astype(int) + mi_mask.astype(int) + l1_mask.astype(int)
        self.mask_ = votes >= self.min_votes
        if self.mask_.sum() < 5:
            self.mask_ = votes >= 1
        if self.mask_.sum() == 0:
            self.mask_ = np.ones(p, dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.mask_]


# ── Reproducible MI wrappers (fix random_state) ─────────────────────────────

def _mi_classif(X, y):
    return mutual_info_classif(X, y, random_state=42)


def _mi_regression(X, y):
    return mutual_info_regression(X, y, random_state=42)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline & parameter-grid construction
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocessing_steps():
    """Common leak-free preprocessing (order matters)."""
    return [
        ("miss_filter", MissingRateFilter(threshold=0.6)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("corr_filter", CorrelationFilter(threshold=0.95)),
        ("scale", StandardScaler()),
    ]


def build_pipeline_and_grid(model, model_params, selection_strategy, task_type):
    """
    Build a full sklearn Pipeline and a combined parameter grid
    that jointly tunes model hyperparameters and selection params.

    Returns (Pipeline, param_grid dict).
    """
    steps = _preprocessing_steps()
    extra_grid = {}

    if selection_strategy == "No Selection":
        pass
    elif selection_strategy == "SelectKBest":
        score_func = f_classif if task_type == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func, k=20)))
        extra_grid["select__k"] = [10, 20, 30]
    elif selection_strategy == "Mutual Info":
        mi_func = _mi_classif if task_type == "classification" else _mi_regression
        steps.append(("select", SelectKBest(mi_func, k=20)))
        extra_grid["select__k"] = [10, 20, 30]
    elif selection_strategy == "L1-based":
        if task_type == "classification":
            l1_est = LogisticRegression(
                penalty="l1", solver="saga", C=0.1,
                max_iter=5000, class_weight="balanced", random_state=42)
        else:
            l1_est = Lasso(alpha=0.1, max_iter=5000, random_state=42)
        steps.append(("select", SelectFromModel(l1_est)))
    elif selection_strategy == "Consensus":
        steps.append(("select", ConsensusSelector(task_type=task_type, k=30)))
    elif selection_strategy == "PCA":
        steps.append(("select", PCA(n_components=0.95, svd_solver="full")))
    else:
        raise ValueError(f"Unknown strategy: {selection_strategy}")

    steps.append(("model", model))
    combined_grid = {**model_params, **extra_grid}
    return Pipeline(steps), combined_grid


# ── Model + grid definitions ────────────────────────────────────────────────

def get_clf_models():
    """Return {name: (estimator, param_grid)} for classification."""
    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=42),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [4, 8, None]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss", random_state=42, n_jobs=-1),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [3, 5],
             "model__learning_rate": [0.01, 0.05, 0.1]},
        )
    return models


def get_reg_models():
    """Return {name: (estimator, param_grid)} for regression."""
    models = {
        "ElasticNet": (
            ElasticNet(max_iter=5000, random_state=42),
            {"model__alpha": [0.01, 0.1, 0.5, 1.0],
             "model__l1_ratio": [0.3, 0.5, 0.7]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [4, 8, None]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [3, 5],
             "model__learning_rate": [0.01, 0.05, 0.1]},
        )
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation (no cleaning — that happens inside the pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(df, feature_cols, outcome_col, demographics=True):
    """Subset columns and drop rows missing the outcome. Returns raw X array and y."""
    feature_part = list(feature_cols) + (demographic_cols if demographics else [])
    cols = feature_part + [outcome_col]
    sub = df[cols].dropna(subset=[outcome_col])
    y = sub[outcome_col].values
    X = sub[feature_part].values.astype(float)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical comparison of feature sets
# ═══════════════════════════════════════════════════════════════════════════════

def corrected_repeated_cv_test(scores_a, scores_b, n_splits=5, n_repeats=3):
    """
    Corrected repeated k-fold CV paired t-test (Nadeau & Bengio, 2003).

    Accounts for the non-independence of CV fold scores, which makes
    the naive paired t-test anticonservative.

    Parameters
    ----------
    scores_a, scores_b : array-like, shape (n_splits * n_repeats,)
        Per-fold metric scores from repeated CV on the **same folds**.
    n_splits, n_repeats : int
        CV configuration (must match how scores were generated).

    Returns
    -------
    dict with keys: t_stat, p_value, mean_diff, ci_lower, ci_upper
    """
    d = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(d)
    d_bar = d.mean()
    s2_d = d.var(ddof=1)

    # Corrected variance: 1/(k*r) + n_test/n_train = 1/(k*r) + 1/(k-1)
    correction = 1.0 / n + 1.0 / (n_splits - 1)
    se = np.sqrt(correction * s2_d)

    if se < 1e-10:
        return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": d_bar,
                "ci_lower": d_bar, "ci_upper": d_bar}

    t_stat = d_bar / se
    df = n - 1
    p_value = 2.0 * stats.t.sf(abs(t_stat), df)

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = d_bar - t_crit * se
    ci_upper = d_bar + t_crit * se

    return {"t_stat": t_stat, "p_value": p_value, "mean_diff": d_bar,
            "ci_lower": ci_lower, "ci_upper": ci_upper}


def compare_feature_sets(all_results, metric, n_splits=5, n_repeats=3):
    """
    For each outcome, identify the best (model × selection) per feature set,
    then run pairwise corrected t-tests on their per-fold scores.

    Parameters
    ----------
    all_results : list[dict]
        Results from run_nested_cv, each dict must contain
        '_fold_<metric>' with the per-fold scores array.
    metric : str
        'R2' or 'AP' — the metric used for comparison.

    Returns
    -------
    comparison_rows : list[dict]
        One row per pairwise comparison.
    """
    fold_key = f"_fold_{metric}"
    mean_key = f"{metric}_mean"

    rows_by_outcome = {}
    for r in all_results:
        if r.get(fold_key) is None or np.isnan(r.get(mean_key, np.nan)):
            continue
        rows_by_outcome.setdefault(r["Outcome"], []).append(r)

    comparison_rows = []
    for outcome, rows in rows_by_outcome.items():
        # Best configuration per feature set
        best = {}
        for r in rows:
            fs = r["Features"]
            if fs not in best or r[mean_key] > best[fs][mean_key]:
                best[fs] = r

        feature_sets = sorted(best.keys())
        for fs_a, fs_b in combinations(feature_sets, 2):
            res = corrected_repeated_cv_test(
                best[fs_a][fold_key], best[fs_b][fold_key],
                n_splits=n_splits, n_repeats=n_repeats,
            )
            sig = ""
            if res["p_value"] < 0.001:
                sig = "***"
            elif res["p_value"] < 0.01:
                sig = "**"
            elif res["p_value"] < 0.05:
                sig = "*"

            comparison_rows.append({
                "Outcome": outcome,
                "Feature_Set_A": fs_a,
                "Model_A": f"{best[fs_a]['Model']} [{best[fs_a]['Selection']}]",
                f"{metric}_A": best[fs_a][mean_key],
                "Feature_Set_B": fs_b,
                "Model_B": f"{best[fs_b]['Model']} [{best[fs_b]['Selection']}]",
                f"{metric}_B": best[fs_b][mean_key],
                f"Δ{metric}": res["mean_diff"],
                "CI_lower": res["ci_lower"],
                "CI_upper": res["ci_upper"],
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "sig": sig,
            })

    return comparison_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Nested cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_nested_cv(X, y, task_type, outcome_name, feature_set_name,
                   selection_strategies=None):
    """
    Nested CV over all (selection × model) combinations for one
    outcome × feature-set pair.

    Outer loop: unbiased performance estimation (5×3 repeated K-fold)
    Inner loop: hyperparameter tuning via 3-fold GridSearchCV

    Parameters
    ----------
    selection_strategies : list[str] or None
        Subset of SELECTION_STRATEGIES to evaluate.  Defaults to all.
        Use ["No Selection"] for baseline feature sets with few features.
    """
    if selection_strategies is None:
        selection_strategies = SELECTION_STRATEGIES
    if task_type == "classification":
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = {
            "AUC": "roc_auc",
            "F1": make_scorer(f1_score, zero_division=0),
            "AP": "average_precision",
            "BalAcc": make_scorer(balanced_accuracy_score),
        }
        inner_scoring = "average_precision"
        model_defs = get_clf_models()
    else:
        outer_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scoring = {
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
        }
        inner_scoring = "r2"
        model_defs = get_reg_models()

    results = []

    for sel_name in selection_strategies:
        for model_name, (model_instance, model_params) in model_defs.items():
            try:
                pipe, grid = build_pipeline_and_grid(
                    clone(model_instance), model_params, sel_name, task_type)

                grid_search = GridSearchCV(
                    pipe, grid,
                    cv=inner_cv,
                    scoring=inner_scoring,
                    refit=True,
                    n_jobs=-1,
                    error_score=np.nan,
                )

                scores = cross_validate(
                    grid_search, X, y,
                    cv=outer_cv,
                    scoring=scoring,
                    n_jobs=1,  # inner GridSearchCV already parallelised
                    return_estimator=True,
                    error_score=np.nan,
                )

                # Collect best params from each outer fold
                best_params_list = [
                    est.best_params_ for est in scores["estimator"]
                    if hasattr(est, "best_params_")
                ]
                modal_params = (
                    Counter(str(p) for p in best_params_list).most_common(1)[0][0]
                    if best_params_list else "N/A"
                )

                row = {
                    "Outcome": outcome_name,
                    "Features": feature_set_name,
                    "Selection": sel_name,
                    "Model": model_name,
                    "n_samples": len(y),
                    "n_features_input": X.shape[1],
                    "best_params": modal_params,
                }

                if task_type == "classification":
                    row.update({
                        "AUC_mean": np.nanmean(scores["test_AUC"]),
                        "AUC_std": np.nanstd(scores["test_AUC"]),
                        "AP_mean": np.nanmean(scores["test_AP"]),
                        "AP_std": np.nanstd(scores["test_AP"]),
                        "BalAcc_mean": np.nanmean(scores["test_BalAcc"]),
                        "BalAcc_std": np.nanstd(scores["test_BalAcc"]),
                        "F1_mean": np.nanmean(scores["test_F1"]),
                        "F1_std": np.nanstd(scores["test_F1"]),
                        "prevalence": y.mean(),
                        "_fold_AP": scores["test_AP"],
                    })
                    print(f"    [{sel_name:14s}] {model_name:22s}  "
                          f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
                          f"AP={row['AP_mean']:.3f}  "
                          f"BalAcc={row['BalAcc_mean']:.3f}")
                else:
                    row.update({
                        "R2_mean": np.nanmean(scores["test_R2"]),
                        "R2_std": np.nanstd(scores["test_R2"]),
                        "MAE_mean": -np.nanmean(scores["test_MAE"]),
                        "MAE_std": np.nanstd(scores["test_MAE"]),
                        "_fold_R2": scores["test_R2"],
                    })
                    print(f"    [{sel_name:14s}] {model_name:22s}  "
                          f"R2={row['R2_mean']:.3f}±{row['R2_std']:.3f}  "
                          f"MAE={row['MAE_mean']:.3f}")

                results.append(row)

            except Exception as e:
                print(f"    [{sel_name:14s}] {model_name:22s}  FAILED: {e}")
                row = {
                    "Outcome": outcome_name,
                    "Features": feature_set_name,
                    "Selection": sel_name,
                    "Model": model_name,
                    "n_samples": len(y),
                    "n_features_input": X.shape[1],
                    "best_params": None,
                }
                if task_type == "classification":
                    row.update({k: np.nan for k in [
                        "AUC_mean", "AUC_std", "AP_mean", "AP_std",
                        "BalAcc_mean", "BalAcc_std", "F1_mean", "F1_std"]})
                    row["prevalence"] = y.mean()
                else:
                    row.update({k: np.nan for k in [
                        "R2_mean", "R2_std", "MAE_mean", "MAE_std"]})
                results.append(row)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Run pipeline
# ═══════════════════════════════════════════════════════════════════════════════

all_clf_results = []
all_reg_results = []

print("\n" + "=" * 70)
print("BINARY CLASSIFICATION  (nested CV)")
print("=" * 70)

for outcome_col, outcome_name in BINARY_OUTCOMES.items():
    print(f"\n{'─'*70}")
    print(f"  {outcome_name} ({outcome_col})")
    print(f"{'─'*70}")
    for fs_name, fs_cols in FEATURE_SETS.items():
        X, y = prepare_data(df, fs_cols, outcome_col, demographics=True)
        if len(y) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
            print(f"  {fs_name}: skipped (n={len(y)}, pos={y.sum():.0f})")
            continue
        print(f"\n  {fs_name}: n={len(y)}, raw_features={X.shape[1]}, "
              f"prevalence={y.mean():.2%}")

        results = run_nested_cv(X, y, "classification", outcome_name, fs_name)
        all_clf_results.extend(results)

print("\n" + "=" * 70)
print("CONTINUOUS REGRESSION  (nested CV)")
print("=" * 70)

for outcome_col, outcome_name in CONTINUOUS_OUTCOMES.items():
    print(f"\n{'─'*70}")
    print(f"  {outcome_name} ({outcome_col})")
    print(f"{'─'*70}")

    # Sensor-based feature sets (full selection strategy sweep)
    for fs_name, fs_cols in FEATURE_SETS.items():
        X, y = prepare_data(df, fs_cols, outcome_col, demographics=True)
        if len(y) < 50:
            print(f"  {fs_name}: skipped (n={len(y)})")
            continue
        print(f"\n  {fs_name}: n={len(y)}, raw_features={X.shape[1]}")

        results = run_nested_cv(X, y, "regression", outcome_name, fs_name)
        all_reg_results.extend(results)

    # Clinical baseline & augmented feature sets
    for fs_name, fs_cols in REGRESSION_EXTRA_FEATURE_SETS.items():
        X, y = prepare_data(df, fs_cols, outcome_col, demographics=True)
        if len(y) < 50:
            print(f"  {fs_name}: skipped (n={len(y)})")
            continue
        # For the baseline (few features), only "No Selection" is meaningful;
        # for the augmented combined set, run the full sweep.
        is_baseline = (len(fs_cols) <= 3)
        sel = ["No Selection"] if is_baseline else None
        print(f"\n  {fs_name}: n={len(y)}, raw_features={X.shape[1]}"
              + (" [baseline — No Selection only]" if is_baseline else ""))

        results = run_nested_cv(X, y, "regression", outcome_name, fs_name,
                                selection_strategies=sel)
        all_reg_results.extend(results)

# ── Statistical comparison of feature sets (regression) ──────────────────────
print("\n" + "=" * 70)
print("PAIRWISE FEATURE-SET COMPARISON  (corrected repeated CV t-test; Nadeau & Bengio 2003)")
print("  Compares best model per feature set for each outcome")
print("=" * 70)

reg_comparisons = compare_feature_sets(all_reg_results, metric="R2",
                                        n_splits=5, n_repeats=3)
if reg_comparisons:
    comp_df = pd.DataFrame(reg_comparisons)
    for outcome in comp_df["Outcome"].unique():
        print(f"\n  {outcome}")
        sub = comp_df[comp_df["Outcome"] == outcome]
        for _, r in sub.iterrows():
            print(f"    {r['Feature_Set_A']:30s} vs {r['Feature_Set_B']:30s}  "
                  f"ΔR²={r['ΔR2']:+.3f}  "
                  f"95%CI=[{r['CI_lower']:+.3f}, {r['CI_upper']:+.3f}]  "
                  f"p={r['p_value']:.4f} {r['sig']}")
    comp_df.to_csv(os.path.join(_OUTPUT_DIR, "feature_set_comparisons_regression.csv"), index=False)
    print(f"\n  Saved: output/feature_set_comparisons_regression.csv")
else:
    print("  No valid comparisons (need ≥2 feature sets with results)")

# ── Save results ─────────────────────────────────────────────────────────────
# Drop internal per-fold score arrays before saving to CSV
clf_df = pd.DataFrame(all_clf_results).drop(
    columns=[c for c in pd.DataFrame(all_clf_results).columns if c.startswith("_fold_")],
    errors="ignore")
reg_df = pd.DataFrame(all_reg_results).drop(
    columns=[c for c in pd.DataFrame(all_reg_results).columns if c.startswith("_fold_")],
    errors="ignore")
clf_df.to_csv(os.path.join(_OUTPUT_DIR, "results_classification_nested_cv.csv"), index=False)
reg_df.to_csv(os.path.join(_OUTPUT_DIR, "results_regression_nested_cv.csv"), index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)
print(f"  Classification: output/results_classification_nested_cv.csv ({len(clf_df)} rows)")
print(f"  Regression:     output/results_regression_nested_cv.csv ({len(reg_df)} rows)")

# ── Summary tables ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLASSIFICATION SUMMARY — Best (Model × Selection) per Outcome × Features")
print("  Ranked by Average Precision (more robust to class imbalance)")
print("=" * 70)
if len(clf_df) > 0:
    valid = clf_df.dropna(subset=["AP_mean"])
    if len(valid) > 0:
        best_clf = valid.loc[valid.groupby(["Outcome", "Features"])["AP_mean"].idxmax()]
        print(best_clf[["Outcome", "Features", "Selection", "Model",
                         "AP_mean", "AP_std", "AUC_mean", "BalAcc_mean",
                         "n_samples", "best_params"]].to_string(index=False))

print("\n" + "=" * 70)
print("REGRESSION SUMMARY — Best (Model × Selection) per Outcome × Features")
print("=" * 70)
if len(reg_df) > 0:
    valid = reg_df.dropna(subset=["R2_mean"])
    if len(valid) > 0:
        best_reg = valid.loc[valid.groupby(["Outcome", "Features"])["R2_mean"].idxmax()]
        print(best_reg[["Outcome", "Features", "Selection", "Model",
                         "R2_mean", "R2_std", "MAE_mean",
                         "n_samples", "best_params"]].to_string(index=False))

# ── Selection strategy comparison ────────────────────────────────────────────
print("\n" + "=" * 70)
print("SELECTION STRATEGY COMPARISON (avg across outcomes & feature sets)")
print("=" * 70)
if len(clf_df) > 0:
    valid = clf_df.dropna(subset=["AP_mean"])
    if len(valid) > 0:
        sel_summary = valid.groupby("Selection").agg(
            AP_mean=("AP_mean", "mean"),
            AUC_mean=("AUC_mean", "mean"),
            BalAcc_mean=("BalAcc_mean", "mean"),
        ).round(3).sort_values("AP_mean", ascending=False)
        print("\nClassification:")
        print(sel_summary.to_string())

if len(reg_df) > 0:
    valid = reg_df.dropna(subset=["R2_mean"])
    if len(valid) > 0:
        sel_summary = valid.groupby("Selection").agg(
            R2_mean=("R2_mean", "mean"),
            MAE_mean=("MAE_mean", "mean"),
        ).round(3).sort_values("R2_mean", ascending=False)
        print("\nRegression:")
        print(sel_summary.to_string())
