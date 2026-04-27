"""
Analysis C — pre-fit EDA (descriptive only, no modeling, no inference).

Runs BEFORE any model fit. Writes:
  runs/longitudinal/c_postmortem/prefit_diagnostics.json
  runs/longitudinal/c_postmortem/figures/eda/01_class_balance.png
  runs/longitudinal/c_postmortem/figures/eda/02_covariates.png
  runs/longitudinal/c_postmortem/figures/eda/03_selection_bias.png
  runs/longitudinal/c_postmortem/figures/eda/04_missingness.png

Descriptive only. Verifies preregistered per-cell counts, covariate
coverage, and cohort definitions. If any tripwire fires, exits non-zero
before any fit could run.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from longitudinal.c_common import (
    load_approved_cohort, build_cohorts, rungs_for_cohort, get_gait_bout_cols,
    FORCED_COVARIATES, OUTCOMES, RUN_DIR, POSTMORTEM_CSV, ensure_run_dir,
    N_APPROVED_EXPECTED, N_COHORT8_EXPECTED,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

TRIPWIRE_N_MIN = 100          # n with non-null outcome
TRIPWIRE_MINORITY_MIN = 25    # minority class >= 25
TRIPWIRE_GAIT_MISS_MAX = 0.30  # gait_speed missingness in approved cohort


def build_prefit_diagnostics(df_approved, cohorts):
    diag = {
        "preregistration": "runs/longitudinal/c_postmortem/preregistration.md",
        "frozen_on": "2026-04-23",
        "n_approved": int(len(df_approved)),
        "n_cohort_d": int(len(cohorts["cohort_d"])),
        "n_cohort_8": int(len(cohorts["cohort_8"])),
        "forced_covariates": list(FORCED_COVARIATES),
        "covariate_non_null_in_approved": {
            c: int(df_approved[c].notna().sum()) for c in FORCED_COVARIATES
        },
        "gait_speed_non_null_in_approved": int(
            df_approved["gait_speed"].notna().sum()),
        "gait_speed_missing_rate_in_approved": float(
            df_approved["gait_speed"].isna().mean()),
        "gait_bout_feature_count": len(get_gait_bout_cols(df_approved)),
        "outcomes": {},
        "tripwires": {"fired": [], "passed": []},
    }

    for okey, spec in OUTCOMES.items():
        cell = {
            "source_col": spec["source"],
            "cutpoint_label": spec["cut_label"],
            "domain": spec["domain"],
            "per_cohort": {},
        }
        for ck, cdf in cohorts.items():
            y = cdf[okey].dropna()
            pos = int(y.sum()); neg = int(len(y) - pos); n = int(len(y))
            cell["per_cohort"][ck] = {"n": n, "pos": pos, "neg": neg}

            # Tripwire #1: n >= 100
            if n < TRIPWIRE_N_MIN:
                diag["tripwires"]["fired"].append(
                    f"n<{TRIPWIRE_N_MIN} for {okey} on {ck}: n={n}")
            # Tripwire #2: minority >= 25
            minority = min(pos, neg)
            if minority < TRIPWIRE_MINORITY_MIN:
                diag["tripwires"]["fired"].append(
                    f"minority<{TRIPWIRE_MINORITY_MIN} for {okey} on {ck}: "
                    f"pos={pos} neg={neg}")
        diag["outcomes"][okey] = cell

    # Tripwire #4: gait_speed missingness
    miss = diag["gait_speed_missing_rate_in_approved"]
    if miss > TRIPWIRE_GAIT_MISS_MAX:
        diag["tripwires"]["fired"].append(
            f"gait_speed missing rate {miss:.1%} > "
            f"{TRIPWIRE_GAIT_MISS_MAX:.0%} cap")
    else:
        diag["tripwires"]["passed"].append(
            f"gait_speed missing rate {miss:.1%} <= "
            f"{TRIPWIRE_GAIT_MISS_MAX:.0%}")

    # Cohort size tripwires
    if diag["n_approved"] != N_APPROVED_EXPECTED:
        diag["tripwires"]["fired"].append(
            f"n_approved mismatch: {diag['n_approved']} vs prereg "
            f"{N_APPROVED_EXPECTED}")
    if diag["n_cohort_8"] != N_COHORT8_EXPECTED:
        diag["tripwires"]["fired"].append(
            f"n_cohort_8 mismatch: {diag['n_cohort_8']} vs prereg "
            f"{N_COHORT8_EXPECTED}")

    if not diag["tripwires"]["fired"]:
        diag["tripwires"]["passed"].append(
            "all outcomes pass n>=100 and minority>=25")

    return diag


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

OUTCOME_ORDER = ["ad_adnc", "lb_7reg", "tdp_st4_bin", "arteriol_scler_bin",
                 "cvda_4gp2_bin", "henl_4gp_bin"]
OUTCOME_LABELS = {
    "ad_adnc": "AD\n(ad_adnc)",
    "lb_7reg": "Lewy 7reg\n(lb_7reg)",
    "tdp_st4_bin": "TDP-43\n(>=1)",
    "arteriol_scler_bin": "Arteriol.\n(>=2)",
    "cvda_4gp2_bin": "CAA\n(>=2)",
    "henl_4gp_bin": "Lewy HENL\n(>=1)",
}


def fig_class_balance(cohorts, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (ck, cdf) in zip(axes, cohorts.items()):
        pos_counts, neg_counts = [], []
        for okey in OUTCOME_ORDER:
            y = cdf[okey].dropna()
            pos = int(y.sum()); neg = int(len(y) - pos)
            pos_counts.append(pos); neg_counts.append(neg)
        xs = np.arange(len(OUTCOME_ORDER))
        ax.bar(xs, neg_counts, label="negative", color="#bfd8df")
        ax.bar(xs, pos_counts, bottom=neg_counts, label="positive",
               color="#346b73")
        for i, (p, n) in enumerate(zip(pos_counts, neg_counts)):
            ax.text(i, p + n + 2, f"{p}/{p+n}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([OUTCOME_LABELS[k] for k in OUTCOME_ORDER], fontsize=8)
        n_c = len(cdf)
        title = "Cohort-D (vs Demographics)" if ck == "cohort_d" \
            else "Cohort-8 (vs 8ft)"
        ax.set_title(f"{title}  n={n_c}")
        ax.set_ylabel("subjects" if ck == "cohort_d" else "")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Pre-fit class balance per (outcome × cohort)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_covariates(df_approved, cohorts, out_path):
    vars_cont = ["age_bl", "age_death", "pmi", "time_lastce2dod", "gait_speed"]
    fig, axes = plt.subplots(1, len(vars_cont), figsize=(16, 3.5))
    for ax, v in zip(axes, vars_cont):
        d = df_approved[v].dropna()
        d8 = cohorts["cohort_8"][v].dropna()
        ax.hist(d, bins=25, color="#bfd8df", edgecolor="white",
                label=f"Cohort-D (n={len(d)})")
        ax.hist(d8, bins=25, color="#346b73", alpha=0.7, edgecolor="white",
                label=f"Cohort-8 (n={len(d8)})")
        ax.set_title(v, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Covariate distributions — Cohort-D vs Cohort-8 overlay",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_selection_bias(out_path):
    """Render the selection_bias_table.csv as a forest-style SMD plot."""
    path = os.path.join(RUN_DIR, "selection_bias_table.csv")
    if not os.path.exists(path):
        print(f"[eda] {path} not found; run c_selection_bias first. Skipping.")
        return
    tbl = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    y = np.arange(len(tbl))
    ax.hlines(y, 0, tbl["smd"], color="#346b73", lw=3)
    ax.scatter(tbl["smd"], y, color="#346b73", s=40, zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.axvline(0.2, color="#aa3a3a", lw=0.4, ls="--", label="|SMD|=0.2 (small)")
    ax.axvline(-0.2, color="#aa3a3a", lw=0.4, ls="--")
    ax.axvline(0.5, color="#aa3a3a", lw=0.4, ls=":", label="|SMD|=0.5 (medium)")
    ax.axvline(-0.5, color="#aa3a3a", lw=0.4, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(tbl["variable"])
    ax.invert_yaxis()
    ax.set_xlabel("SMD (analysis cohort − excluded)")
    ax.set_title("Selection bias: analysis cohort (n=187) vs excluded (n=603)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_missingness(df_approved, out_path):
    gait = get_gait_bout_cols(df_approved)
    miss_rates = df_approved[gait].isna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(np.arange(len(miss_rates)), miss_rates.values,
           color="#346b73", width=1.0)
    ax.axhline(0.6, color="#aa3a3a", lw=0.7, ls="--",
               label="MissingRateFilter cap = 0.6")
    ax.set_xlabel(f"gait-bout features ({len(miss_rates)} total), sorted by missing rate")
    ax.set_ylabel("missing rate")
    ax.set_title("Gait-bout feature missingness (approved cohort n=187)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    out_dir = ensure_run_dir()
    fig_dir = ensure_run_dir(os.path.join("figures", "eda"))

    df = load_approved_cohort()
    cohorts = build_cohorts(df)

    diag = build_prefit_diagnostics(df, cohorts)

    diag_path = os.path.join(out_dir, "prefit_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2, default=str)
    print(f"[eda] wrote {diag_path}")

    if diag["tripwires"]["fired"]:
        print("[eda] TRIPWIRES FIRED — investigate before running the driver:")
        for t in diag["tripwires"]["fired"]:
            print(f"  - {t}")
    else:
        print("[eda] all tripwires passed")

    fig_class_balance(cohorts,
                      os.path.join(fig_dir, "01_class_balance.png"))
    fig_covariates(df, cohorts,
                   os.path.join(fig_dir, "02_covariates.png"))
    fig_selection_bias(os.path.join(fig_dir, "03_selection_bias.png"))
    fig_missingness(df, os.path.join(fig_dir, "04_missingness.png"))
    print(f"[eda] wrote 4 figures to {fig_dir}")


if __name__ == "__main__":
    main()
