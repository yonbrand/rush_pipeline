"""
Cross-sectional prediction pipeline: gait metrics -> clinical outcomes.
Uses ABL (analytic baseline) data only — one visit per subject.

Outcomes:
  Binary:     parkinsonism_yn, mobility_disability_binary, falls_binary,
              cognitive_impairment
  Continuous: parksc, motor10, cogn_global

Models:
  Primary:     Logistic Regression / Linear Regression (ElasticNet)
  Sensitivity: Random Forest, XGBoost

Feature buckets evaluated independently and jointly:
  1. Gait bout metrics
  2. Daily PA variables
  3. Combined (gait + daily PA)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.metrics import (roc_auc_score, f1_score, make_scorer,
                             r2_score, mean_absolute_error, mean_squared_error)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost models")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("output/merged_gait_clinical_abl.csv")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Define feature buckets ────────────────────────────────────────────────
id_cols = ["sub_id", "projid", "fu_year", "wear_days", "study"]
daily_pa_cols = [c for c in df.columns
                 if c.startswith("daily_pa_mean_") or c.startswith("daily_pa_std_") or c.startswith("tdpa_")]

# Exclude all outcome / clinical / demographic columns from gait features
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
gait_bout_cols = [c for c in df.columns if c not in exclude_from_features
                  and not c.startswith("daily_pa_mean_")
                  and not c.startswith("daily_pa_std_")
                  and not c.startswith("tdpa_")]
demographic_cols = ["age_bl", "msex", "educ"]

print(f"Gait bout features: {len(gait_bout_cols)}")
print(f"Daily PA features:  {len(daily_pa_cols)}")

# ── Outcome definitions ───────────────────────────────────────────────────
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

# ── Feature buckets ───────────────────────────────────────────────────────
FEATURE_SETS = {
    "Gait Bout": gait_bout_cols,
    "Daily PA": daily_pa_cols,
    "Combined": gait_bout_cols + daily_pa_cols,
}


# ═══════════════════════════════════════════════════════════════════════════
# Cleaning & feature selection helpers
# ═══════════════════════════════════════════════════════════════════════════

def clean_features(X_df):
    """Remove columns with >60% missing, zero-variance, or highly correlated (>0.95)."""
    # Drop columns with too much missing
    miss_frac = X_df.isnull().mean()
    keep = miss_frac[miss_frac < 0.6].index.tolist()
    X_df = X_df[keep]

    # Drop zero/near-zero variance after imputing for variance check
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_df), columns=X_df.columns)
    vt = VarianceThreshold(threshold=1e-8)
    vt.fit(X_imp)
    keep = X_df.columns[vt.get_support()].tolist()
    X_df = X_df[keep]
    X_imp = X_imp[keep]

    # Remove highly correlated features (keep first)
    corr_matrix = X_imp.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_df = X_df.drop(columns=to_drop)

    return X_df


def prepare_data(df, feature_cols, outcome_col, demographics=True):
    """Subset data, clean features, return X, y."""
    cols = list(feature_cols) + (demographic_cols if demographics else []) + [outcome_col]
    sub = df[cols].dropna(subset=[outcome_col])

    y = sub[outcome_col].values
    feature_part = list(feature_cols)
    if demographics:
        feature_part = feature_part + demographic_cols

    X_df = sub[feature_part].copy()
    X_df = clean_features(X_df)

    return X_df, y


# ═══════════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════════

def get_classifiers(n_features):
    k = min(n_features, 30)
    models = {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                       solver="lbfgs", class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                           class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.8,
                                  scale_pos_weight=1,
                                  eval_metric="logloss",
                                  random_state=42, n_jobs=-1)),
        ])
    return models


def get_regressors(n_features):
    k = min(n_features, 30)
    models = {
        "ElasticNet": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=k)),
            ("clf", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_regression, k=k)),
            ("clf", RandomForestRegressor(n_estimators=200, max_depth=8,
                                          random_state=42, n_jobs=-1)),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_regression, k=k)),
            ("clf", XGBRegressor(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=42, n_jobs=-1)),
        ])
    return models


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_classification(X_df, y, models, outcome_name, feature_set_name):
    """5-fold repeated stratified CV for classification."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {
        "AUC": "roc_auc",
        "F1": make_scorer(f1_score, zero_division=0),
    }
    results = []
    for model_name, pipe in models.items():
        try:
            scores = cross_validate(pipe, X_df.values, y, cv=cv, scoring=scoring,
                                    n_jobs=-1, error_score="raise")
            results.append({
                "Outcome": outcome_name,
                "Features": feature_set_name,
                "Model": model_name,
                "AUC_mean": scores["test_AUC"].mean(),
                "AUC_std": scores["test_AUC"].std(),
                "F1_mean": scores["test_F1"].mean(),
                "F1_std": scores["test_F1"].std(),
                "n_samples": len(y),
                "n_features_input": X_df.shape[1],
                "prevalence": y.mean(),
            })
        except Exception as e:
            print(f"  WARN: {model_name} failed on {outcome_name}/{feature_set_name}: {e}")
            results.append({
                "Outcome": outcome_name, "Features": feature_set_name,
                "Model": model_name, "AUC_mean": np.nan, "AUC_std": np.nan,
                "F1_mean": np.nan, "F1_std": np.nan,
                "n_samples": len(y), "n_features_input": X_df.shape[1],
                "prevalence": y.mean(),
            })
    return results


def evaluate_regression(X_df, y, models, outcome_name, feature_set_name):
    """5-fold repeated CV for regression."""
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {
        "R2": "r2",
        "MAE": "neg_mean_absolute_error",
    }
    results = []
    for model_name, pipe in models.items():
        try:
            scores = cross_validate(pipe, X_df.values, y, cv=cv, scoring=scoring,
                                    n_jobs=-1, error_score="raise")
            results.append({
                "Outcome": outcome_name,
                "Features": feature_set_name,
                "Model": model_name,
                "R2_mean": scores["test_R2"].mean(),
                "R2_std": scores["test_R2"].std(),
                "MAE_mean": -scores["test_MAE"].mean(),
                "MAE_std": scores["test_MAE"].std(),
                "n_samples": len(y),
                "n_features_input": X_df.shape[1],
            })
        except Exception as e:
            print(f"  WARN: {model_name} failed on {outcome_name}/{feature_set_name}: {e}")
            results.append({
                "Outcome": outcome_name, "Features": feature_set_name,
                "Model": model_name, "R2_mean": np.nan, "R2_std": np.nan,
                "MAE_mean": np.nan, "MAE_std": np.nan,
                "n_samples": len(y), "n_features_input": X_df.shape[1],
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Run pipeline
# ═══════════════════════════════════════════════════════════════════════════

all_clf_results = []
all_reg_results = []

print("\n" + "=" * 70)
print("BINARY CLASSIFICATION")
print("=" * 70)

for outcome_col, outcome_name in BINARY_OUTCOMES.items():
    print(f"\n-- {outcome_name} ({outcome_col}) --")
    for fs_name, fs_cols in FEATURE_SETS.items():
        X_df, y = prepare_data(df, fs_cols, outcome_col, demographics=True)
        if len(y) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
            print(f"  {fs_name}: skipped (n={len(y)}, pos={y.sum()})")
            continue
        print(f"  {fs_name}: n={len(y)}, features={X_df.shape[1]}, prevalence={y.mean():.2%}")
        models = get_classifiers(X_df.shape[1])
        results = evaluate_classification(X_df, y, models, outcome_name, fs_name)
        for r in results:
            print(f"    {r['Model']:25s}  AUC={r['AUC_mean']:.3f}+/-{r['AUC_std']:.3f}  "
                  f"F1={r['F1_mean']:.3f}+/-{r['F1_std']:.3f}")
        all_clf_results.extend(results)

print("\n" + "=" * 70)
print("CONTINUOUS REGRESSION")
print("=" * 70)

for outcome_col, outcome_name in CONTINUOUS_OUTCOMES.items():
    print(f"\n-- {outcome_name} ({outcome_col}) --")
    for fs_name, fs_cols in FEATURE_SETS.items():
        X_df, y = prepare_data(df, fs_cols, outcome_col, demographics=True)
        if len(y) < 50:
            print(f"  {fs_name}: skipped (n={len(y)})")
            continue
        print(f"  {fs_name}: n={len(y)}, features={X_df.shape[1]}")
        models = get_regressors(X_df.shape[1])
        results = evaluate_regression(X_df, y, models, outcome_name, fs_name)
        for r in results:
            print(f"    {r['Model']:25s}  R2={r['R2_mean']:.3f}+/-{r['R2_std']:.3f}  "
                  f"MAE={r['MAE_mean']:.3f}+/-{r['MAE_std']:.3f}")
        all_reg_results.extend(results)

# ── Save results ──────────────────────────────────────────────────────────
clf_df = pd.DataFrame(all_clf_results)
reg_df = pd.DataFrame(all_reg_results)
clf_df.to_csv("output/results_classification.csv", index=False)
reg_df.to_csv("output/results_regression.csv", index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)
print(f"  Classification: output/results_classification.csv ({len(clf_df)} rows)")
print(f"  Regression:     output/results_regression.csv ({len(reg_df)} rows)")

# ── Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLASSIFICATION SUMMARY (best model per outcome × feature set)")
print("=" * 70)
if len(clf_df) > 0:
    best_clf = clf_df.loc[clf_df.groupby(["Outcome", "Features"])["AUC_mean"].idxmax()]
    print(best_clf[["Outcome", "Features", "Model", "AUC_mean", "AUC_std",
                     "F1_mean", "n_samples"]].to_string(index=False))

print("\n" + "=" * 70)
print("REGRESSION SUMMARY (best model per outcome × feature set)")
print("=" * 70)
if len(reg_df) > 0:
    best_reg = reg_df.loc[reg_df.groupby(["Outcome", "Features"])["R2_mean"].idxmax()]
    print(best_reg[["Outcome", "Features", "Model", "R2_mean", "R2_std",
                     "MAE_mean", "n_samples"]].to_string(index=False))
