"""
Analysis C — selection-bias helper (`selection_bias_table.csv`).

Compares the analysis cohort (n=187 approved autopsy) to the excluded
subjects (n=603: alive or unapproved) on the 6 variables listed in the
preregistration: age_bl, msex, educ, gait_speed, cogn_global,
mobility_disability_binary.

Writes `runs/longitudinal/c_postmortem/selection_bias_table.csv`:
  variable | n_analysis | n_excluded | mean_analysis | mean_excluded |
  sd_analysis | sd_excluded | smd | test | p_value

SMD = (mean_analysis − mean_excluded) / pooled_sd for continuous;
Cohen's h for binary (small effect: |h|<0.2).
Tests: t-test (continuous), chi-square (binary).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from longitudinal.c_common import (
    POSTMORTEM_CSV, RUN_DIR, ID_COL, ensure_run_dir,
)


BIAS_VARIABLES = [
    ("age_bl", "continuous"),
    ("msex", "binary"),
    ("educ", "continuous"),
    ("gait_speed", "continuous"),
    ("cogn_global", "continuous"),
    ("mobility_disability_binary", "binary"),
]


def _pooled_sd(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    return np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                   / (nx + ny - 2))


def _smd_continuous(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s = _pooled_sd(x, y)
    if not np.isfinite(s) or s < 1e-12:
        return np.nan
    return (x.mean() - y.mean()) / s


def _cohens_h(p1, p2):
    def _phi(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return 2 * np.arcsin(np.sqrt(p))
    return _phi(p1) - _phi(p2)


def compute_selection_bias_table():
    df = pd.read_csv(POSTMORTEM_CSV)

    analysis = df.loc[df["npath_approved"] == 1].copy()
    excluded = df.loc[df["npath_approved"] != 1].copy()
    if len(analysis) != 187:
        raise RuntimeError(f"Analysis cohort size: expected 187, got {len(analysis)}")

    rows = []
    for var, kind in BIAS_VARIABLES:
        if var not in df.columns:
            rows.append({"variable": var, "test": "MISSING_COLUMN"})
            continue

        a = analysis[var].dropna()
        e = excluded[var].dropna()

        row = {
            "variable": var,
            "kind": kind,
            "n_analysis": int(len(a)),
            "n_excluded": int(len(e)),
        }
        if kind == "continuous":
            row["mean_analysis"] = float(a.mean()) if len(a) else np.nan
            row["sd_analysis"]   = float(a.std(ddof=1)) if len(a) > 1 else np.nan
            row["mean_excluded"] = float(e.mean()) if len(e) else np.nan
            row["sd_excluded"]   = float(e.std(ddof=1)) if len(e) > 1 else np.nan
            if len(a) > 1 and len(e) > 1:
                t, p = stats.ttest_ind(a, e, equal_var=False, nan_policy="omit")
                row["test"] = "welch_t"
                row["t_stat"] = float(t)
                row["p_value"] = float(p)
                row["smd"] = float(_smd_continuous(a, e))
            else:
                row["test"] = "INSUFFICIENT_N"
        else:  # binary
            # Compute class-1 rate, chi-square, Cohen's h.
            a_bin = a.astype(int); e_bin = e.astype(int)
            if a_bin.nunique() < 2 or e_bin.nunique() < 2:
                row["test"] = "CONSTANT_CATEGORY"
                continue
            p1 = float(a_bin.mean()); p2 = float(e_bin.mean())
            row["mean_analysis"] = p1     # interpret as pos rate
            row["mean_excluded"] = p2
            row["sd_analysis"]   = float(np.sqrt(p1 * (1 - p1)))
            row["sd_excluded"]   = float(np.sqrt(p2 * (1 - p2)))
            # 2x2 contingency
            counts = pd.crosstab(
                pd.Series(np.concatenate([a_bin, e_bin])),
                pd.Series(["analysis"] * len(a_bin) + ["excluded"] * len(e_bin)),
            )
            chi2, p, dof, _ = stats.chi2_contingency(counts)
            row["test"] = "chi2"
            row["t_stat"] = float(chi2)
            row["p_value"] = float(p)
            row["smd"] = float(_cohens_h(p1, p2))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    out_dir = ensure_run_dir()
    tbl = compute_selection_bias_table()
    path = os.path.join(out_dir, "selection_bias_table.csv")
    tbl.to_csv(path, index=False)
    print(f"[selection_bias] wrote {path} ({len(tbl)} rows)")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
