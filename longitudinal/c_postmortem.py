"""
Analysis C driver — nested CV over 36 (outcome x cohort x rung) cells.

Reads the frozen preregistration at
`runs/longitudinal/c_postmortem/preregistration.md`. Writes all outputs
under `runs/longitudinal/c_postmortem/`:

  per_fold_scores.csv   per-cell, per-model AP/AUC plus per-fold arrays
  descriptive.csv       AP mean/std per (outcome, cohort, rung), winner model
  contrasts.csv         Nadeau-Bengio results for both contrasts + Holm x6
  summary.json          headline numbers (best cell per contrast + Holm)
  feature_selection_sensitivity.csv
                        SelectKBest re-run of the +Gait Bout rungs (post-hoc)

Model grid (per preregistration, ~104 candidates x 36 cells ~= 170k fits):
  LR-l2, LR-l1, LR-elasticnet, SVC-RBF, RF, HistGB
  x  Block PCA variance_retained in {0.80, 0.90}

Two primary contrasts (each Holm x6 across outcomes):
  Cohort-D:  (+ Gait Bout) vs Demographics
  Cohort-8:  (+ Gait Bout) vs (+ 8ft)
"""
import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
)

from core.pipeline import build_pipeline_and_grid
from core.cv import corrected_repeated_cv_test, holm_correction
from longitudinal.c_common import (
    load_approved_cohort, build_cohorts, rungs_for_cohort,
    OUTCOMES, OUTER_SEED, N_SPLITS, N_REPEATS,
    prepare_xy, ensure_run_dir,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model grid (preregistration)
# ---------------------------------------------------------------------------

def get_models():
    """Return {name: (estimator, param_grid)}. Matches preregistration."""
    return {
        "LR-l2": (
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                               class_weight="balanced", random_state=42),
            {"model__C": list(np.logspace(-3, 2, 8))},
        ),
        "LR-l1": (
            LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                               class_weight="balanced", random_state=42),
            {"model__C": list(np.logspace(-3, 2, 8))},
        ),
        "LR-elasticnet": (
            LogisticRegression(penalty="elasticnet", solver="saga",
                               max_iter=5000, class_weight="balanced",
                               random_state=42, l1_ratio=0.5),
            {"model__C": [0.1, 1.0, 10.0],
             "model__l1_ratio": [0.25, 0.5, 0.75]},
        ),
        "SVC-RBF": (
            SVC(kernel="rbf", class_weight="balanced", probability=False,
                random_state=42),
            {"model__C": [0.1, 1.0, 10.0],
             "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
        ),
        "RF": (
            RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=42, n_jobs=1),
            {"model__max_depth": [None, 5],
             "model__min_samples_leaf": [1, 3],
             "model__max_features": ["sqrt", 0.3]},
        ),
        "HistGB": (
            HistGradientBoostingClassifier(random_state=42, max_iter=300,
                                           class_weight="balanced"),
            {"model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1]},
        ),
    }


SCORING = {"AP": "average_precision", "AUC": "roc_auc"}


# ---------------------------------------------------------------------------
# Nested-CV runner (one cell = one (outcome, cohort, rung) triple)
# ---------------------------------------------------------------------------

def run_one_cell(X, y, feature_names, cell_label,
                 selection_strategy="Block PCA", verbose=True):
    """
    Nested CV: outer RepeatedStratifiedKFold (5x3, seed=42), inner
    StratifiedKFold (3, seed=42). Each model in the registry is wrapped in
    GridSearchCV over its own params + Block PCA variance_retained.

    Returns a list of dicts (one per model). Each dict carries per-fold AP
    and AUC arrays under `_fold_AP` and `_fold_AUC` for later pairing.
    """
    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=OUTER_SEED)
    inner_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=OUTER_SEED)

    rows = []
    for model_name, (inst, params) in get_models().items():
        t0 = time.time()
        try:
            pipe, grid = build_pipeline_and_grid(
                clone(inst), params, selection_strategy, "classification",
                feature_names=feature_names)
            gs = GridSearchCV(
                pipe, grid, cv=inner_cv, scoring="average_precision",
                refit=True, n_jobs=-1, error_score=np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_validate(
                    gs, X, y, cv=outer_cv, scoring=SCORING,
                    n_jobs=1, return_estimator=False, error_score=np.nan)
            row = {
                "cell": cell_label,
                "model": model_name,
                "selection": selection_strategy,
                "n": int(len(y)),
                "n_raw_features": int(X.shape[1]),
                "prevalence": float(y.mean()),
                "AP_mean": float(np.nanmean(scores["test_AP"])),
                "AP_std":  float(np.nanstd(scores["test_AP"])),
                "AUC_mean": float(np.nanmean(scores["test_AUC"])),
                "AUC_std":  float(np.nanstd(scores["test_AUC"])),
                "elapsed_sec": round(time.time() - t0, 1),
                "_fold_AP":  np.asarray(scores["test_AP"]),
                "_fold_AUC": np.asarray(scores["test_AUC"]),
            }
            if verbose:
                print(f"    {model_name:14s} AP={row['AP_mean']:.3f}"
                      f"+/-{row['AP_std']:.3f}  AUC={row['AUC_mean']:.3f}"
                      f"  ({row['elapsed_sec']:.0f}s)")
        except Exception as exc:
            row = {"cell": cell_label, "model": model_name,
                   "selection": selection_strategy, "n": int(len(y)),
                   "n_raw_features": int(X.shape[1]),
                   "prevalence": float(y.mean()),
                   "error": str(exc), "elapsed_sec": round(time.time() - t0, 1),
                   "_fold_AP": None, "_fold_AUC": None}
            if verbose:
                print(f"    {model_name:14s} FAILED: {exc}")
        rows.append(row)
    return rows


def best_model_row(cell_rows):
    """Return row with max AP_mean (ignore failures)."""
    valid = [r for r in cell_rows
             if r.get("AP_mean") is not None
             and np.isfinite(r.get("AP_mean", np.nan))
             and r.get("_fold_AP") is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r["AP_mean"])


# ---------------------------------------------------------------------------
# Contrasts + Holm correction
# ---------------------------------------------------------------------------

CONTRAST_DEFS = {
    # name: (cohort, rung_a, rung_b) — "A vs B", delta = A - B (positive => A wins)
    "cohort_d_gait_vs_demo": ("cohort_d", "+ Gait Bout", "Demographics"),
    "cohort_8_gait_vs_8ft":  ("cohort_8", "+ Gait Bout", "+ 8ft"),
}

SENSITIVITY_CONTRASTS = {
    # Reported without Holm (sensitivity only; not in the co-primary family).
    "cohort_8_combined_vs_8ft": ("cohort_8", "+ Gait Bout + 8ft", "+ 8ft"),
    "cohort_8_gait_vs_demo":    ("cohort_8", "+ Gait Bout", "Demographics"),
}


def run_contrast(best_per_rung, cohort, rung_a, rung_b, outcome_key):
    ra = best_per_rung.get((cohort, rung_a, outcome_key))
    rb = best_per_rung.get((cohort, rung_b, outcome_key))
    if ra is None or rb is None:
        return {"n_folds": 0, "mean_diff": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "t_stat": np.nan,
                "model_a": None, "model_b": None,
                "ap_a": np.nan, "ap_b": np.nan, "note": "missing_cell"}
    res = corrected_repeated_cv_test(
        ra["_fold_AP"], rb["_fold_AP"],
        n_splits=N_SPLITS, n_repeats=N_REPEATS)
    return {
        **res,
        "model_a": ra["model"], "model_b": rb["model"],
        "ap_a": ra["AP_mean"], "ap_b": rb["AP_mean"],
        "note": "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = ensure_run_dir()
    t_start = time.time()

    df = load_approved_cohort()
    cohorts = build_cohorts(df)

    # Plan the cells.
    cells = []  # (cohort_key, rung_name, outcome_key)
    for ck, cdf in cohorts.items():
        rungs = rungs_for_cohort(cdf, ck)
        for rk in rungs.keys():
            for okey in OUTCOMES.keys():
                cells.append((ck, rk, okey))
    print(f"[driver] {len(cells)} cells to run "
          f"(6 outcomes x [Cohort-D: 2 rungs + Cohort-8: 4 rungs])")

    all_rows = []                  # flat per-model rows
    best_per_rung = {}             # (cohort, rung, outcome) -> best-model row

    for i, (ck, rk, okey) in enumerate(cells, 1):
        cdf = cohorts[ck]
        rungs = rungs_for_cohort(cdf, ck)
        fcols = rungs[rk]
        X, y, feature_names = prepare_xy(cdf, fcols, okey)
        pos = int(y.sum()); neg = int(len(y) - pos)
        cell_label = f"{ck}|{rk}|{okey}"
        print(f"\n[{i}/{len(cells)}] {cell_label}  "
              f"n={len(y)} pos={pos} neg={neg} raw_feats={X.shape[1]}")
        cell_rows = run_one_cell(X, y, feature_names, cell_label)
        for r in cell_rows:
            r["cohort"] = ck; r["rung"] = rk; r["outcome"] = okey
        all_rows.extend(cell_rows)

        best = best_model_row(cell_rows)
        if best is not None:
            best_per_rung[(ck, rk, okey)] = best

    # ---- Persist per-fold scores + descriptive table ----
    # Save fold arrays separately as npz to keep CSV readable.
    fold_arrays = {}
    csv_rows = []
    for r in all_rows:
        key_base = f"{r['cohort']}|{r['rung']}|{r['outcome']}|{r['model']}"
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        csv_rows.append(clean)
        for fk in ("_fold_AP", "_fold_AUC"):
            arr = r.get(fk)
            if arr is not None:
                fold_arrays[f"{key_base}|{fk[1:]}"] = np.asarray(arr)

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(out_dir, "per_fold_scores.csv"), index=False)
    if fold_arrays:
        np.savez_compressed(
            os.path.join(out_dir, "fold_scores.npz"), **fold_arrays)
    print(f"\n[save] per_fold_scores.csv ({len(csv_rows)} rows) + fold_scores.npz")

    # Descriptive: winner model per cell.
    desc = []
    for (ck, rk, okey), r in best_per_rung.items():
        desc.append({
            "cohort": ck, "rung": rk, "outcome": okey,
            "winner_model": r["model"],
            "AP_mean": r["AP_mean"], "AP_std": r["AP_std"],
            "AUC_mean": r["AUC_mean"], "AUC_std": r["AUC_std"],
            "n": r["n"], "prevalence": r["prevalence"],
        })
    desc_df = pd.DataFrame(desc)
    if not desc_df.empty:
        desc_df = desc_df.sort_values(["cohort", "outcome", "rung"])
    desc_df.to_csv(os.path.join(out_dir, "descriptive.csv"), index=False)

    # ---- Primary contrasts: Holm x6 per contrast family ----
    contrast_rows = []
    holm_families = {}
    outcome_keys = list(OUTCOMES.keys())

    for cname, (cohort, ra, rb) in CONTRAST_DEFS.items():
        fam = []
        for okey in outcome_keys:
            res = run_contrast(best_per_rung, cohort, ra, rb, okey)
            res.update({"contrast": cname, "cohort": cohort,
                        "rung_a": ra, "rung_b": rb, "outcome": okey,
                        "family": "primary"})
            fam.append(res)
        ps_full = np.array([r["p_value"] for r in fam], dtype=float)
        # Holm over valid p-values only; pad non-finite as 1.0 reject=False
        valid_mask = np.isfinite(ps_full)
        adj = np.ones_like(ps_full)
        reject = np.zeros_like(ps_full, dtype=bool)
        if valid_mask.any():
            a, r_ = holm_correction(ps_full[valid_mask], alpha=0.05)
            adj[valid_mask] = a; reject[valid_mask] = r_
        for r, a, e in zip(fam, adj, reject):
            r["p_holm"] = float(a); r["reject_holm"] = bool(e)
        holm_families[cname] = fam
        contrast_rows.extend(fam)

    for cname, (cohort, ra, rb) in SENSITIVITY_CONTRASTS.items():
        for okey in outcome_keys:
            res = run_contrast(best_per_rung, cohort, ra, rb, okey)
            res.update({"contrast": cname, "cohort": cohort,
                        "rung_a": ra, "rung_b": rb, "outcome": okey,
                        "family": "sensitivity",
                        "p_holm": np.nan, "reject_holm": False})
            contrast_rows.append(res)

    pd.DataFrame(contrast_rows).to_csv(
        os.path.join(out_dir, "contrasts.csv"), index=False)
    print(f"[save] contrasts.csv ({len(contrast_rows)} rows; "
          f"primary families Holm-corrected)")

    # ---- Headline summary ----
    def _headline_block(fam):
        return [{"outcome": r["outcome"],
                 "ap_gait": r["ap_a"], "ap_comparator": r["ap_b"],
                 "delta": r["mean_diff"],
                 "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                 "p_raw": r["p_value"],
                 "p_holm": r["p_holm"],
                 "reject_holm": r["reject_holm"],
                 "model_gait": r["model_a"], "model_comparator": r["model_b"]}
                for r in fam]

    summary = {
        "preregistration": "runs/longitudinal/c_postmortem/preregistration.md",
        "frozen_on": "2026-04-23",
        "n_cells": len(cells),
        "n_approved": int(len(df)),
        "n_cohort_d": int(len(cohorts["cohort_d"])),
        "n_cohort_8": int(len(cohorts["cohort_8"])),
        "primary_contrasts": {
            cname: _headline_block(fam)
            for cname, fam in holm_families.items()
        },
        "elapsed_sec": round(time.time() - t_start, 1),
        "stat_test": "Nadeau & Bengio (2003) corrected repeated-kfold t-test "
                     "(core.cv.corrected_repeated_cv_test)",
        "multiplicity": "Holm-Bonferroni step-down within each primary "
                        "contrast family (m=6 outcomes).",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] summary.json")

    # ---- Post-hoc feature-selection sensitivity ----
    # Rerun just the +Gait Bout rungs (both cohorts) with SelectKBest to
    # see whether the Block-PCA backbone is the source of any signal.
    print("\n[sensitivity] Re-running + Gait Bout rungs with SelectKBest...")
    sens_rows = []
    for ck in ("cohort_d", "cohort_8"):
        cdf = cohorts[ck]
        rungs = rungs_for_cohort(cdf, ck)
        fcols = rungs["+ Gait Bout"]
        for okey in outcome_keys:
            X, y, fn = prepare_xy(cdf, fcols, okey)
            cell_label = f"{ck}|+ Gait Bout|{okey}|SelectKBest"
            print(f"  [{ck}|{okey}] n={len(y)} raw={X.shape[1]}")
            cell_rows = run_one_cell(X, y, fn, cell_label,
                                     selection_strategy="SelectKBest",
                                     verbose=False)
            best = best_model_row(cell_rows)
            pca_best = best_per_rung.get((ck, "+ Gait Bout", okey))
            if best is None or pca_best is None:
                continue
            # Paired contrast against the Block-PCA winner.
            res = corrected_repeated_cv_test(
                best["_fold_AP"], pca_best["_fold_AP"],
                n_splits=N_SPLITS, n_repeats=N_REPEATS)
            sens_rows.append({
                "cohort": ck, "outcome": okey,
                "skb_model": best["model"], "skb_ap": best["AP_mean"],
                "pca_model": pca_best["model"], "pca_ap": pca_best["AP_mean"],
                "delta_skb_minus_pca": res["mean_diff"],
                "ci_lower": res["ci_lower"], "ci_upper": res["ci_upper"],
                "p_raw": res["p_value"],
                "note": "Holm not applied (post-hoc sensitivity)",
            })
    pd.DataFrame(sens_rows).to_csv(
        os.path.join(out_dir, "feature_selection_sensitivity.csv"), index=False)
    print(f"[save] feature_selection_sensitivity.csv ({len(sens_rows)} rows)")

    print(f"\n[done] total wall-clock {summary['elapsed_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
