"""
Nested cross-validation runner + Nadeau-Bengio comparison + Holm correction.

The outer CV `random_state` is fixed at 42 for the entire project. Per-fold
scores are persisted so future experiments can run paired comparisons against
the baseline without re-running it.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from scipy import stats

from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold, RepeatedKFold,
    StratifiedKFold, KFold,
    cross_validate, GridSearchCV,
)
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score

from core.pipeline import (
    build_pipeline_and_grid, get_clf_models, get_reg_models,
    DEFAULT_SELECTION_STRATEGIES,
)

OUTER_SEED = 42
N_SPLITS = 5
N_REPEATS = 3


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

def run_nested_cv(X, y, task_type, outcome_name, feature_set_name,
                  selection_strategies=None, feature_names=None, verbose=True):
    """
    Returns a list of dicts, one per (selection x model) cell.
    Each dict carries `_fold_AP` (clf) or `_fold_R2` (reg) for paired tests.
    """
    if selection_strategies is None:
        selection_strategies = DEFAULT_SELECTION_STRATEGIES

    if task_type == "classification":
        outer_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                           random_state=OUTER_SEED)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SEED)
        scoring = {"AUC": "roc_auc",
                   "AP": "average_precision",
                   "BalAcc": make_scorer(balanced_accuracy_score),
                   "F1": make_scorer(f1_score, zero_division=0)}
        inner_scoring = "average_precision"
        models = get_clf_models()
        primary = "AP"
    else:
        outer_cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=OUTER_SEED)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=OUTER_SEED)
        scoring = {"R2": "r2", "MAE": "neg_mean_absolute_error"}
        inner_scoring = "r2"
        models = get_reg_models()
        primary = "R2"

    results = []
    for sel_name in selection_strategies:
        for model_name, (model_inst, params) in models.items():
            try:
                pipe, grid = build_pipeline_and_grid(
                    clone(model_inst), params, sel_name, task_type,
                    feature_names=feature_names)
                gs = GridSearchCV(
                    pipe, grid, cv=inner_cv, scoring=inner_scoring,
                    refit=True, n_jobs=-1, error_score=np.nan,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_validate(
                        gs, X, y, cv=outer_cv, scoring=scoring,
                        n_jobs=1, return_estimator=False, error_score=np.nan,
                    )

                row = {
                    "Outcome": outcome_name,
                    "Features": feature_set_name,
                    "Selection": sel_name,
                    "Model": model_name,
                    "n_samples": int(len(y)),
                    "n_features_input": int(X.shape[1]),
                }

                if task_type == "classification":
                    row.update({
                        "AUC_mean": float(np.nanmean(scores["test_AUC"])),
                        "AUC_std":  float(np.nanstd(scores["test_AUC"])),
                        "AP_mean":  float(np.nanmean(scores["test_AP"])),
                        "AP_std":   float(np.nanstd(scores["test_AP"])),
                        "BalAcc_mean": float(np.nanmean(scores["test_BalAcc"])),
                        "F1_mean":  float(np.nanmean(scores["test_F1"])),
                        "prevalence": float(y.mean()),
                        "_fold_AP": np.asarray(scores["test_AP"]),
                        "_fold_AUC": np.asarray(scores["test_AUC"]),
                    })
                    if verbose:
                        print(f"    [{sel_name:14s}] {model_name:22s}  "
                              f"AUC={row['AUC_mean']:.3f}  AP={row['AP_mean']:.3f}  "
                              f"BalAcc={row['BalAcc_mean']:.3f}")
                else:
                    row.update({
                        "R2_mean":  float(np.nanmean(scores["test_R2"])),
                        "R2_std":   float(np.nanstd(scores["test_R2"])),
                        "MAE_mean": float(-np.nanmean(scores["test_MAE"])),
                        "_fold_R2": np.asarray(scores["test_R2"]),
                    })
                    if verbose:
                        print(f"    [{sel_name:14s}] {model_name:22s}  "
                              f"R2={row['R2_mean']:.3f} +/- {row['R2_std']:.3f}  "
                              f"MAE={row['MAE_mean']:.3f}")

                results.append(row)
            except Exception as e:
                if verbose:
                    print(f"    [{sel_name:14s}] {model_name:22s}  FAILED: {e}")
                row = {"Outcome": outcome_name, "Features": feature_set_name,
                       "Selection": sel_name, "Model": model_name,
                       "n_samples": int(len(y)), "n_features_input": int(X.shape[1]),
                       "error": str(e)}
                results.append(row)
    return results


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def corrected_repeated_cv_test(scores_a, scores_b, n_splits=N_SPLITS,
                               n_repeats=N_REPEATS, alpha=0.05):
    """Nadeau & Bengio (2003) corrected paired t-test for repeated k-fold CV."""
    d = np.asarray(scores_a) - np.asarray(scores_b)
    # Drop fold pairs where either side is nan (CV failure for one config)
    finite = np.isfinite(d)
    d = d[finite]
    n = len(d)
    if n < 2:
        return {"t_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "n_folds": n}
    d_bar = d.mean()
    s2_d = d.var(ddof=1)
    correction = 1.0 / n + 1.0 / (n_splits - 1)
    se = np.sqrt(correction * s2_d)
    if se < 1e-12:
        return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": float(d_bar),
                "ci_lower": float(d_bar), "ci_upper": float(d_bar), "n_folds": n}
    t_stat = d_bar / se
    df = n - 1
    p_value = 2.0 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return {"t_stat": float(t_stat), "p_value": float(p_value),
            "mean_diff": float(d_bar),
            "ci_lower": float(d_bar - t_crit * se),
            "ci_upper": float(d_bar + t_crit * se),
            "n_folds": int(n)}


def best_per_feature_set(rows, primary_metric):
    """Group rows by (Outcome, Features), keep the highest-mean configuration."""
    mean_key = f"{primary_metric}_mean"
    best = {}
    for r in rows:
        if r.get(mean_key) is None or (isinstance(r.get(mean_key), float)
                                       and not np.isfinite(r[mean_key])):
            continue
        key = (r["Outcome"], r["Features"])
        if key not in best or r[mean_key] > best[key][mean_key]:
            best[key] = r
    return best


def pairwise_comparisons(rows, primary_metric, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    fold_key = f"_fold_{primary_metric}"
    mean_key = f"{primary_metric}_mean"
    by_outcome = {}
    for r in rows:
        if r.get(fold_key) is None or (isinstance(r.get(mean_key), float)
                                        and not np.isfinite(r.get(mean_key, np.nan))):
            continue
        by_outcome.setdefault(r["Outcome"], []).append(r)

    comparisons = []
    for outcome, outcome_rows in by_outcome.items():
        best = best_per_feature_set(outcome_rows, primary_metric)
        # Restrict to this outcome
        best = {k: v for k, v in best.items() if k[0] == outcome}
        feature_sets = sorted([k[1] for k in best.keys()])
        for fs_a, fs_b in combinations(feature_sets, 2):
            ra = best[(outcome, fs_a)]; rb = best[(outcome, fs_b)]
            res = corrected_repeated_cv_test(
                ra[fold_key], rb[fold_key], n_splits, n_repeats)
            comparisons.append({
                "Outcome": outcome,
                "Feature_Set_A": fs_a, "Model_A": f"{ra['Model']} [{ra['Selection']}]",
                f"{primary_metric}_A": ra[mean_key],
                "Feature_Set_B": fs_b, "Model_B": f"{rb['Model']} [{rb['Selection']}]",
                f"{primary_metric}_B": rb[mean_key],
                f"delta_{primary_metric}": res["mean_diff"],
                "CI_lower": res["ci_lower"], "CI_upper": res["ci_upper"],
                "t_stat": res["t_stat"], "p_value": res["p_value"],
                "n_folds": res["n_folds"],
            })
    return comparisons


def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (adjusted_p, reject_flags)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); adj.fill(np.nan)
    running_max = 0.0
    for rank, i in enumerate(order):
        scaled = (m - rank) * p[i]
        running_max = max(running_max, scaled)
        adj[i] = min(running_max, 1.0)
    reject = adj < alpha
    return adj, reject


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results, run_dir, primary_metric_clf="AP", primary_metric_reg="R2"):
    """Persist per-cell results, fold scores (npz), and a summary JSON."""
    os.makedirs(run_dir, exist_ok=True)

    # Per-fold score arrays - npz keyed by (outcome|features|selection|model|metric)
    fold_arrays = {}
    rows_for_csv = []
    for r in results:
        key_base = "|".join([r["Outcome"], r["Features"],
                              r["Selection"], r["Model"]])
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        # Pull out fold arrays
        for fk in [k for k in r.keys() if k.startswith("_fold_")]:
            arr = r[fk]
            if arr is not None:
                fold_arrays[f"{key_base}|{fk[1:]}"] = np.asarray(arr)
        rows_for_csv.append(clean)

    pd.DataFrame(rows_for_csv).to_csv(
        os.path.join(run_dir, "results.csv"), index=False)
    if fold_arrays:
        np.savez_compressed(os.path.join(run_dir, "fold_scores.npz"), **fold_arrays)

    return rows_for_csv
