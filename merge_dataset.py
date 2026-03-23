"""
Merge gait metrics (subject_summary.csv) with clinical/demographic data
from wrist_sensor_metadata_Yonatan.xlsx.

Outputs (controlled by flags below):
  - output/merged_gait_clinical_abl.csv   (ABL only — one visit per subject)
  - output/merged_gait_clinical_lv.csv    (LV only — last visit per subject)
  - output/merged_gait_clinical_allvisits.csv  (all valid visits)
  - output/merged_gait_clinical_postmortem.csv (ABL + postmortem indices)
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# FLAGS — toggle which datasets to produce
# ═══════════════════════════════════════════════════════════════════════════
EXPORT_ABL = True           # Analytic baseline (first visit) — for cross-sectional
EXPORT_LV = True            # Last visit — for future analyses
EXPORT_ALL_VISITS = True    # All valid visits — for longitudinal / mixed models
EXPORT_POSTMORTEM = True    # ABL + postmortem pathology indices

# ── Load data ──────────────────────────────────────────────────────────────
gait = pd.read_csv("output/subject_summary.csv")
excel = "output/wrist_sensor_metadata_Yonatan.xlsx"
abl = pd.read_excel(excel, sheet_name="ABL")
lv = pd.read_excel(excel, sheet_name="LV")
allv = pd.read_excel(excel, sheet_name="Data from all valid cycles")
postmortem = pd.read_excel(excel, sheet_name="Postmortem Indices")

print(f"Gait data: {gait.shape[0]} rows, {gait.shape[1]} columns")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: derive binary outcomes + add feature bucket info
# ═══════════════════════════════════════════════════════════════════════════

def derive_binary_outcomes(merged):
    """Create binary columns from multi-class outcomes."""
    # Mobility disability: rosbsum > 0
    merged["mobility_disability_binary"] = (merged["rosbsum"] > 0).astype(float)
    merged.loc[merged["rosbsum"].isna(), "mobility_disability_binary"] = np.nan

    # Falls binary: falls > 0 (from full 'falls' variable if present)
    if "falls" in merged.columns:
        merged["falls_binary"] = (merged["falls"] > 0).astype(float)
        merged.loc[merged["falls"].isna(), "falls_binary"] = np.nan
    elif "falls_binary" not in merged.columns:
        merged["falls_binary"] = np.nan

    # Cognitive impairment: dcfdx == 1 -> NCI (0), dcfdx > 1 -> impaired (1)
    if "dcfdx" in merged.columns:
        merged["cognitive_impairment"] = (merged["dcfdx"] > 1).astype(float)
        merged.loc[merged["dcfdx"].isna(), "cognitive_impairment"] = np.nan
    elif "dcfdx_3gp" in merged.columns:
        merged["cognitive_impairment"] = (merged["dcfdx_3gp"] > 1).astype(float)
        merged.loc[merged["dcfdx_3gp"].isna(), "cognitive_impairment"] = np.nan
    else:
        merged["cognitive_impairment"] = np.nan

    # Age at visit
    if "age_bl" in merged.columns:
        merged["age_at_visit"] = merged["age_bl"] + merged["fu_year"]

    return merged


def print_outcome_summary(merged, label):
    """Print outcome availability for a merged dataset."""
    outcomes = {
        "parkinsonism_yn": "binary",
        "mobility_disability_binary": "binary",
        "falls_binary": "binary",
        "cognitive_impairment": "binary",
        "parksc": "continuous",
        "motor10": "continuous",
        "cogn_global": "continuous",
    }
    print(f"\n  Outcome availability ({label}):")
    for col, dtype in outcomes.items():
        if col not in merged.columns:
            print(f"    {col:35s}  NOT AVAILABLE")
            continue
        n = merged[col].notna().sum()
        if dtype == "binary" and n > 0:
            pos = (merged[col] == 1).sum()
            print(f"    {col:35s}  n={n:5d}  pos={pos:4d}  ({100*pos/n:.1f}%)")
        elif dtype == "continuous" and n > 0:
            print(f"    {col:35s}  n={n:5d}  mean={merged[col].mean():.3f}  std={merged[col].std():.3f}")
        else:
            print(f"    {col:35s}  n=0")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ABL — Analytic Baseline (one visit per subject, full clinical)
# ═══════════════════════════════════════════════════════════════════════════
if EXPORT_ABL:
    merged_abl = gait.merge(abl, on=["projid", "fu_year"], how="inner")
    merged_abl = derive_binary_outcomes(merged_abl)

    print(f"\n[ABL] Merged: {merged_abl.shape[0]} rows, {merged_abl.shape[1]} cols")
    print(f"  Unique subjects: {merged_abl['projid'].nunique()}")
    print_outcome_summary(merged_abl, "ABL")

    merged_abl.to_csv("output/merged_gait_clinical_abl.csv", index=False)
    print(f"  -> Saved: output/merged_gait_clinical_abl.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LV — Last Visit (one visit per subject, full clinical)
# ═══════════════════════════════════════════════════════════════════════════
if EXPORT_LV:
    merged_lv = gait.merge(lv, on=["projid", "fu_year"], how="inner")
    merged_lv = derive_binary_outcomes(merged_lv)

    # Add demographics from ABL (age_bl, msex, educ, race7 only in ABL)
    demo_cols = ["projid", "age_bl", "msex", "educ", "race7"]
    demo = abl[demo_cols].drop_duplicates(subset="projid")
    merged_lv = merged_lv.merge(demo, on="projid", how="left")
    merged_lv["age_at_visit"] = merged_lv["age_bl"] + merged_lv["fu_year"]

    print(f"\n[LV] Merged: {merged_lv.shape[0]} rows, {merged_lv.shape[1]} cols")
    print(f"  Unique subjects: {merged_lv['projid'].nunique()}")
    print_outcome_summary(merged_lv, "LV")

    merged_lv.to_csv("output/merged_gait_clinical_lv.csv", index=False)
    print(f"  -> Saved: output/merged_gait_clinical_lv.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 3. All Valid Visits (multiple visits per subject, limited outcomes)
# ═══════════════════════════════════════════════════════════════════════════
if EXPORT_ALL_VISITS:
    allv_renamed = allv.rename(columns={
        "parkinsonism_YN": "parkinsonism_yn",
        "falls_yn": "falls_binary",
        "dcfdx_3gp": "dcfdx_3gp",
    }).drop_duplicates(subset=["projid", "fu_year"])

    merged_allv = gait.merge(allv_renamed, on=["projid", "fu_year"], how="inner")
    merged_allv = derive_binary_outcomes(merged_allv)

    # Add demographics from ABL
    demo_cols = ["projid", "age_bl", "msex", "educ", "race7"]
    demo = abl[demo_cols].drop_duplicates(subset="projid")
    merged_allv = merged_allv.merge(demo, on="projid", how="left")
    merged_allv["age_at_visit"] = merged_allv["age_bl"] + merged_allv["fu_year"]

    print(f"\n[ALL VISITS] Merged: {merged_allv.shape[0]} rows, {merged_allv.shape[1]} cols")
    print(f"  Unique subjects: {merged_allv['projid'].nunique()}")
    print(f"  Note: limited outcomes (dcfdx_3gp, parkinsonism_yn, rosbsum, falls_binary)")
    print_outcome_summary(merged_allv, "All Visits")

    merged_allv.to_csv("output/merged_gait_clinical_allvisits.csv", index=False)
    print(f"  -> Saved: output/merged_gait_clinical_allvisits.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Postmortem — ABL gait + postmortem pathology indices
# ═══════════════════════════════════════════════════════════════════════════
if EXPORT_POSTMORTEM:
    # Start from ABL merge, then add postmortem by projid
    merged_pm = gait.merge(abl, on=["projid", "fu_year"], how="inner")
    merged_pm = merged_pm.merge(postmortem.drop(columns=["study"], errors="ignore"),
                                on="projid", how="inner")
    merged_pm = derive_binary_outcomes(merged_pm)

    print(f"\n[POSTMORTEM] Merged: {merged_pm.shape[0]} rows, {merged_pm.shape[1]} cols")
    print(f"  Unique subjects: {merged_pm['projid'].nunique()}")
    pm_cols = [c for c in postmortem.columns if c not in ["projid", "study"]]
    for c in pm_cols:
        n = merged_pm[c].notna().sum()
        print(f"    {c:25s}  n={n}")

    merged_pm.to_csv("output/merged_gait_clinical_postmortem.csv", index=False)
    print(f"  -> Saved: output/merged_gait_clinical_postmortem.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Feature bucket summary
# ═══════════════════════════════════════════════════════════════════════════
daily_pa_cols = [c for c in gait.columns if c.startswith("daily_")]
id_cols = ["sub_id", "projid", "fu_year", "wear_days"]
gait_bout_cols = [c for c in gait.columns
                  if c not in id_cols and c not in daily_pa_cols]

print(f"\nFeature buckets (from gait data):")
print(f"  Daily PA variables: {len(daily_pa_cols)}")
print(f"  Gait bout metrics:  {len(gait_bout_cols)}")
