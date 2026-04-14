"""
Baseline run: dev set only, 7 preregistered ladder rungs x 4 outcomes
(2 co-primary + 2 secondary). Writes runs/baseline/.

Lean strategy set (No Selection, SelectKBest, L1) + 3 models per task.
Wider strategy / model sweeps live in subsequent experiments.

This is the number to beat. Re-running it produces identical results
(seeds fixed). Do not edit after writing the first results.
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core.data import (
    load_dev, prepare_data, feature_sets,
    BINARY_OUTCOMES, CONTINUOUS_OUTCOMES,
    CO_PRIMARY, SECONDARY, EXPLORATORY,
)
from core.cv import (
    run_nested_cv, pairwise_comparisons, holm_correction,
    save_results, best_per_feature_set,
)

REPO = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(REPO, "runs", "baseline")
os.makedirs(RUN_DIR, exist_ok=True)

# Lean strategy set for the baseline; experiments add more.
BASELINE_STRATEGIES = ["No Selection", "SelectKBest", "L1-based"]

# Outcomes evaluated in the baseline (co-primary + secondary).
# Exploratory mUPDRS-derived outcomes are skipped; they can be added in
# experiments if the headline result is established.
TARGET_OUTCOMES = list(CO_PRIMARY.keys()) + list(SECONDARY.keys())


def main():
    t0 = time.time()
    df = load_dev()
    print(f"[load] dev set: {df.shape[0]} subjects, {df.shape[1]} columns")
    fs_dict = feature_sets(df)
    print(f"[features] {len(fs_dict)} ladder rungs:")
    for name, cols in fs_dict.items():
        print(f"  {name}: {len(cols)} sensor cols (+ 3 demographics)")

    # Per-outcome n
    per_outcome_n = {}
    for outcome in TARGET_OUTCOMES:
        n = int(df[outcome].notna().sum())
        per_outcome_n[outcome] = n
        print(f"[outcome] {outcome}: n={n}")
    with open(os.path.join(RUN_DIR, "n_per_outcome.json"), "w") as f:
        json.dump(per_outcome_n, f, indent=2)

    all_results = []

    for outcome in TARGET_OUTCOMES:
        if outcome in CO_PRIMARY:
            task = CO_PRIMARY[outcome]
            tier = "co-primary"
        elif outcome in SECONDARY:
            task = SECONDARY[outcome]
            tier = "secondary"
        else:
            continue
        outcome_label = (BINARY_OUTCOMES.get(outcome)
                         or CONTINUOUS_OUTCOMES.get(outcome) or outcome)

        print(f"\n{'='*72}\n  {outcome_label} ({outcome}) [{tier} | {task}]\n{'='*72}")

        for fs_name, fs_cols in fs_dict.items():
            X, y, feat_names = prepare_data(df, fs_cols, outcome, demographics=True)
            # Sanity: balance / sample size
            if task == "classification":
                pos = int(y.sum()); neg = int(len(y) - pos)
                if len(y) < 50 or pos < 10 or neg < 10:
                    print(f"  {fs_name}: skipped (n={len(y)}, pos={pos})")
                    continue
                print(f"\n  {fs_name}: n={len(y)}, raw_features={X.shape[1]}, "
                      f"prevalence={y.mean():.2%}")
            else:
                if len(y) < 50:
                    print(f"  {fs_name}: skipped (n={len(y)})")
                    continue
                print(f"\n  {fs_name}: n={len(y)}, raw_features={X.shape[1]}")

            # Demographics-only and +8ft only have 3-4 features, single strategy.
            if len(fs_cols) <= 1:
                strategies = ["No Selection"]
            else:
                strategies = BASELINE_STRATEGIES

            res = run_nested_cv(X, y, task, outcome_label, fs_name,
                                selection_strategies=strategies,
                                feature_names=feat_names, verbose=True)
            for r in res:
                r["_outcome_col"] = outcome
                r["_task_type"] = task
                r["_tier"] = tier
            all_results.extend(res)

    # Persist
    rows = save_results(all_results, RUN_DIR)
    print(f"\n[save] runs/baseline/results.csv ({len(rows)} rows)")

    # Pairwise comparisons within each outcome
    clf_rows = [r for r in all_results if r.get("_task_type") == "classification"]
    reg_rows = [r for r in all_results if r.get("_task_type") == "regression"]

    clf_comp = pairwise_comparisons(clf_rows, "AP")
    reg_comp = pairwise_comparisons(reg_rows, "R2")

    # Holm correction within each outcome (across pairwise comparisons)
    def _apply_holm(comp_rows):
        out = []
        by_outcome = {}
        for r in comp_rows:
            by_outcome.setdefault(r["Outcome"], []).append(r)
        for outcome, rows in by_outcome.items():
            ps = [r["p_value"] for r in rows]
            adj, rej = holm_correction(ps, alpha=0.05)
            for r, a, e in zip(rows, adj, rej):
                r["p_holm"] = float(a)
                r["reject_holm"] = bool(e)
                out.append(r)
        return out

    clf_comp = _apply_holm(clf_comp)
    reg_comp = _apply_holm(reg_comp)

    pd.DataFrame(clf_comp).to_csv(
        os.path.join(RUN_DIR, "comparisons_classification.csv"), index=False)
    pd.DataFrame(reg_comp).to_csv(
        os.path.join(RUN_DIR, "comparisons_regression.csv"), index=False)

    # ---- Headline numbers for the preregistered primary claim ----
    headline = {}
    for outcome, task in CO_PRIMARY.items():
        outcome_label = (BINARY_OUTCOMES.get(outcome)
                         or CONTINUOUS_OUTCOMES.get(outcome) or outcome)
        comp = clf_comp if task == "classification" else reg_comp
        primary_metric = "AP" if task == "classification" else "R2"
        # Find +Gait Bout vs +8ft Gait Speed comparison
        for r in comp:
            if r["Outcome"] != outcome_label:
                continue
            sets = (r["Feature_Set_A"], r["Feature_Set_B"])
            if "+ Gait Bout" in sets and "+ 8ft Gait Speed" in sets:
                # Want delta = GaitBout - 8ft (positive = gait bout wins)
                if r["Feature_Set_A"] == "+ Gait Bout":
                    delta = r[f"delta_{primary_metric}"]
                    ci_lo, ci_hi = r["CI_lower"], r["CI_upper"]
                else:
                    delta = -r[f"delta_{primary_metric}"]
                    ci_lo, ci_hi = -r["CI_upper"], -r["CI_lower"]
                headline[outcome] = {
                    "metric": primary_metric,
                    "delta": delta,
                    "ci_lower": ci_lo, "ci_upper": ci_hi,
                    "p_value_NB": r["p_value"],
                    "p_holm": r["p_holm"],
                    "reject_holm": r["reject_holm"],
                    "model_A": r["Model_A"], "model_B": r["Model_B"],
                    "value_GaitBout": (r["AP_A"] if "AP_A" in r and r["Feature_Set_A"]=="+ Gait Bout"
                                       else r.get("R2_A") if r["Feature_Set_A"]=="+ Gait Bout"
                                       else r.get("AP_B", r.get("R2_B"))),
                    "value_8ft": (r["AP_B"] if "AP_B" in r and r["Feature_Set_B"]=="+ 8ft Gait Speed"
                                   else r.get("R2_B") if r["Feature_Set_B"]=="+ 8ft Gait Speed"
                                   else r.get("AP_A", r.get("R2_A"))),
                }

    # Apply Holm-Bonferroni across the 2 co-primary tests
    if len(headline) == 2:
        ps = [v["p_value_NB"] for v in headline.values()]
        adj, rej = holm_correction(ps, alpha=0.05)
        for (k, v), a, e in zip(headline.items(), adj, rej):
            v["p_holm_coprimary"] = float(a)
            v["reject_holm_coprimary"] = bool(e)

    summary = {
        "run": "baseline",
        "n_dev": int(df.shape[0]),
        "outcomes": TARGET_OUTCOMES,
        "n_per_outcome": per_outcome_n,
        "feature_sets": list(fs_dict.keys()),
        "selection_strategies": BASELINE_STRATEGIES,
        "headline_coprimary": headline,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] {summary['elapsed_sec']:.0f}s elapsed")
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
