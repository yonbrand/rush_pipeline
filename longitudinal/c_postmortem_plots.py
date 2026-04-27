"""
Analysis C — post-fit figures. Run AFTER c_postmortem.py finishes.

Reads:
  runs/longitudinal/c_postmortem/contrasts.csv
  runs/longitudinal/c_postmortem/descriptive.csv
  runs/longitudinal/c_postmortem/feature_selection_sensitivity.csv

Writes:
  runs/longitudinal/c_postmortem/figures/post/05_forest_cohort_d.png
  runs/longitudinal/c_postmortem/figures/post/05_forest_cohort_8.png
  runs/longitudinal/c_postmortem/figures/post/06_ap_per_rung.png
  runs/longitudinal/c_postmortem/figures/post/07_winner_heatmap.png
  runs/longitudinal/c_postmortem/figures/post/08_skb_sensitivity.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from longitudinal.c_common import RUN_DIR, OUTCOMES, ensure_run_dir


OUTCOME_ORDER = ["ad_adnc", "lb_7reg", "tdp_st4_bin", "arteriol_scler_bin",
                 "cvda_4gp2_bin", "henl_4gp_bin"]
OUTCOME_LABELS = {
    "ad_adnc": "AD (ad_adnc)",
    "lb_7reg": "Lewy 7reg",
    "tdp_st4_bin": "TDP-43 (>=1)",
    "arteriol_scler_bin": "Arteriolosclerosis (>=2)",
    "cvda_4gp2_bin": "CAA (>=2)",
    "henl_4gp_bin": "Lewy HENL (>=1)",
}


# ---------------------------------------------------------------------------
# Forest plot (primary contrasts)
# ---------------------------------------------------------------------------

def fig_forest(contrasts, contrast_name, title, out_path):
    sub = contrasts[contrasts["contrast"] == contrast_name].copy()
    if sub.empty:
        print(f"[plots] no rows for {contrast_name}; skip")
        return
    sub["outcome"] = pd.Categorical(sub["outcome"],
                                     categories=OUTCOME_ORDER, ordered=True)
    sub = sub.sort_values("outcome")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(sub))
    # Point + CI
    for i, (_, row) in enumerate(sub.iterrows()):
        if pd.isna(row["mean_diff"]):
            continue
        color = "#346b73" if bool(row.get("reject_holm", False)) else "#7a8a8d"
        ax.hlines(i, row["ci_lower"], row["ci_upper"], color=color, lw=2.0)
        ax.plot(row["mean_diff"], i, "o", color=color, ms=8, zorder=3)
        # p-value annotation
        pstr = "p<.001" if row["p_value"] < 1e-3 else f"p={row['p_value']:.3f}"
        star = " *" if bool(row.get("reject_holm", False)) else ""
        ax.text(0.99, i, f"  Δ={row['mean_diff']:+.3f}  {pstr}{star}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([OUTCOME_LABELS[o] for o in sub["outcome"]])
    ax.invert_yaxis()
    ax.set_xlabel("ΔAP  (positive => Gait Bout wins)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# AP per rung (absolute values, lets reader see ceiling)
# ---------------------------------------------------------------------------

def fig_ap_per_rung(desc, out_path):
    # Order rungs consistently per cohort.
    rung_order = {
        "cohort_d": ["Demographics", "+ Gait Bout"],
        "cohort_8": ["Demographics", "+ 8ft", "+ Gait Bout", "+ Gait Bout + 8ft"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ck in zip(axes, ("cohort_d", "cohort_8")):
        sub = desc[desc["cohort"] == ck].copy()
        rungs = rung_order[ck]
        x = np.arange(len(OUTCOME_ORDER))
        width = 0.8 / len(rungs)
        palette = ["#cbd5dc", "#8ba7b0", "#4f8591", "#20626d"][:len(rungs)]
        for j, rung in enumerate(rungs):
            vals = []
            errs = []
            for okey in OUTCOME_ORDER:
                r = sub[(sub["rung"] == rung) & (sub["outcome"] == okey)]
                vals.append(float(r["AP_mean"].iloc[0]) if len(r) else np.nan)
                errs.append(float(r["AP_std"].iloc[0]) if len(r) else 0.0)
            offsets = x + (j - (len(rungs) - 1) / 2) * width
            ax.bar(offsets, vals, width=width, color=palette[j],
                   edgecolor="white", label=rung, yerr=errs, capsize=2,
                   error_kw={"elinewidth": 0.7, "ecolor": "#555"})
        ax.set_xticks(x)
        ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AP (mean over 15 outer folds)")
        ax.set_title(f"{'Cohort-D (n=187)' if ck=='cohort_d' else 'Cohort-8 (n=146)'}",
                     fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Absolute AP per (outcome x rung) — winner model per cell",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Winner-model heatmap
# ---------------------------------------------------------------------------

def fig_winner_heatmap(desc, out_path):
    rung_order = (["Demographics", "+ Gait Bout"]
                  + ["+ 8ft", "+ Gait Bout + 8ft"])
    # Show cohort_d and cohort_8 side by side; fall back to NaN where not present.
    rows = [(ck, rk) for ck in ("cohort_d", "cohort_8") for rk in rung_order
            if (ck, rk) in {(r["cohort"], r["rung"]) for _, r in desc.iterrows()}]
    models = sorted(desc["winner_model"].dropna().unique().tolist())
    if not models or not rows:
        print("[plots] no data for winner heatmap; skip")
        return
    grid = np.full((len(rows), len(OUTCOME_ORDER)), "", dtype=object)
    for (ck, rk) in rows:
        for oi, okey in enumerate(OUTCOME_ORDER):
            r = desc[(desc["cohort"] == ck) & (desc["rung"] == rk)
                     & (desc["outcome"] == okey)]
            if len(r):
                grid[rows.index((ck, rk)), oi] = str(r["winner_model"].iloc[0])

    # Encode model -> int for colormap
    model_to_code = {m: i for i, m in enumerate(models)}
    arr = np.full_like(grid, fill_value=-1, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] in model_to_code:
                arr[i, j] = model_to_code[grid[i, j]]
    masked = np.ma.masked_where(arr < 0, arr)

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(rows)))
    cmap = plt.get_cmap("tab10", len(models))
    im = ax.imshow(masked, aspect="auto", cmap=cmap,
                    vmin=-0.5, vmax=len(models) - 0.5)
    # text
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                ax.text(j, i, grid[i, j], ha="center", va="center",
                        fontsize=7, color="white")
    ax.set_xticks(range(len(OUTCOME_ORDER)))
    ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOME_ORDER],
                       rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{ck} | {rk}" for (ck, rk) in rows], fontsize=8)
    ax.set_title("Winner model per cell (best AP)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SelectKBest sensitivity forest
# ---------------------------------------------------------------------------

def fig_skb_sensitivity(sens, out_path):
    if sens.empty:
        print("[plots] no sensitivity rows; skip")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, ck in zip(axes, ("cohort_d", "cohort_8")):
        sub = sens[sens["cohort"] == ck].copy()
        sub["outcome"] = pd.Categorical(sub["outcome"],
                                         categories=OUTCOME_ORDER, ordered=True)
        sub = sub.sort_values("outcome")
        y = np.arange(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.hlines(i, row["ci_lower"], row["ci_upper"],
                      color="#346b73", lw=2.0)
            ax.plot(row["delta_skb_minus_pca"], i, "o",
                    color="#346b73", ms=7, zorder=3)
            ax.text(0.99, i,
                     f"  SKB={row['skb_ap']:.2f} / PCA={row['pca_ap']:.2f}",
                     transform=ax.get_yaxis_transform(),
                     ha="right", va="center", fontsize=7)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([OUTCOME_LABELS[o] for o in sub["outcome"]],
                           fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ(SelectKBest − Block PCA) AP")
        ax.set_title(f"{'Cohort-D' if ck=='cohort_d' else 'Cohort-8'}  + Gait Bout",
                     fontsize=10)
    fig.suptitle("Post-hoc feature-selection sensitivity (no Holm)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    fig_dir = ensure_run_dir(os.path.join("figures", "post"))

    contrasts_path = os.path.join(RUN_DIR, "contrasts.csv")
    desc_path = os.path.join(RUN_DIR, "descriptive.csv")
    sens_path = os.path.join(RUN_DIR, "feature_selection_sensitivity.csv")

    if not all(os.path.exists(p) for p in (contrasts_path, desc_path)):
        raise SystemExit(f"Run c_postmortem.py first. Missing: "
                         f"{[p for p in (contrasts_path, desc_path) if not os.path.exists(p)]}")

    contrasts = pd.read_csv(contrasts_path)
    desc = pd.read_csv(desc_path)

    fig_forest(contrasts, "cohort_d_gait_vs_demo",
               "Cohort-D: + Gait Bout vs Demographics  (Holm x6)",
               os.path.join(fig_dir, "05_forest_cohort_d.png"))
    fig_forest(contrasts, "cohort_8_gait_vs_8ft",
               "Cohort-8: + Gait Bout vs + 8ft  (Holm x6)",
               os.path.join(fig_dir, "05_forest_cohort_8.png"))
    fig_ap_per_rung(desc, os.path.join(fig_dir, "06_ap_per_rung.png"))
    fig_winner_heatmap(desc, os.path.join(fig_dir, "07_winner_heatmap.png"))

    if os.path.exists(sens_path):
        sens = pd.read_csv(sens_path)
        fig_skb_sensitivity(sens, os.path.join(fig_dir, "08_skb_sensitivity.png"))
    else:
        print("[plots] no feature_selection_sensitivity.csv; skipped fig 08")

    print(f"[plots] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
