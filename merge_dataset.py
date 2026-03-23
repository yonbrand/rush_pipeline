"""
Merge gait metrics (subject_summary.csv) with clinical/demographic data
from wrist_sensor_metadata_Yonatan.xlsx.

Strategy:
- Use "Data from all valid cycles" sheet for visit-level outcomes (projid + fu_year)
- Stack ABL + LV sheets for additional clinical vars (parksc, motor10, cogn_global, dcfdx full)
- Pull demographics (age_bl, msex, educ, race7) from ABL sheet (one per subject)

Output: output/merged_gait_clinical.csv
"""

import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
gait = pd.read_csv("output/subject_summary.csv")
excel = "output/wrist_sensor_metadata_Yonatan.xlsx"
abl = pd.read_excel(excel, sheet_name="ABL")
lv = pd.read_excel(excel, sheet_name="LV")
allv = pd.read_excel(excel, sheet_name="Data from all valid cycles")

print(f"Gait data: {gait.shape[0]} rows, {gait.shape[1]} columns")

# ── Demographics from ABL (one row per subject) ───────────────────────────
demo_cols = ["projid", "age_bl", "msex", "educ", "race7", "study"]
demographics = abl[demo_cols].drop_duplicates(subset="projid")
print(f"Demographics: {demographics.shape[0]} unique subjects")

# ── Clinical outcomes from 'all valid cycles' (visit-level) ───────────────
# Standardize column names to match ABL/LV conventions
allv_renamed = allv.rename(columns={
    "parkinsonism_YN": "parkinsonism_yn",
    "falls_yn": "falls_binary",
    "dcfdx_3gp": "dcfdx_3gp",
})
allv_cols = ["projid", "fu_year", "parkinsonism_yn", "dcfdx_3gp",
             "rosbsum", "falls_binary", "iadlsum", "katzsum"]
allv_subset = allv_renamed[allv_cols].drop_duplicates(subset=["projid", "fu_year"])

# ── Stack ABL + LV for full clinical variables ────────────────────────────
clinical_cols = ["projid", "fu_year", "dcfdx", "dementia", "cogn_global",
                 "cogn_ep", "cogn_po", "cogn_ps", "cogn_se", "cogn_wo",
                 "cts_mmse30", "parksc", "motor10", "parkinsonism_yn",
                 "rosbsum", "falls", "gait_speed", "gaitsc", "bradysc",
                 "rigidsc", "tremsc", "r_depres", "bmi",
                 "med_con_sum_cum", "vasc_risks_sum"]
abl_clinical = abl[[c for c in clinical_cols if c in abl.columns]]
lv_clinical = lv[[c for c in clinical_cols if c in lv.columns]]
stacked_clinical = pd.concat([abl_clinical, lv_clinical], ignore_index=True)
stacked_clinical = stacked_clinical.drop_duplicates(subset=["projid", "fu_year"])
print(f"Stacked ABL+LV clinical: {stacked_clinical.shape[0]} visit-rows")

# ── Merge: gait ← allv (visit-level) ← stacked_clinical ← demographics ──
merged = gait.merge(allv_subset, on=["projid", "fu_year"], how="left")
print(f"After allv merge: {merged.shape[0]} rows, "
      f"{merged['parkinsonism_yn'].notna().sum()} with parkinsonism_yn")

# Add full clinical from ABL+LV (avoid duplicating columns already present)
clinical_extra_cols = [c for c in stacked_clinical.columns
                       if c not in allv_subset.columns or c in ["projid", "fu_year"]]
merged = merged.merge(stacked_clinical[clinical_extra_cols],
                      on=["projid", "fu_year"], how="left")
print(f"After clinical merge: {merged.shape[0]} rows, "
      f"{merged['parksc'].notna().sum()} with parksc")

# Add demographics
merged = merged.merge(demographics, on="projid", how="left")
print(f"After demographics merge: {merged.shape[0]} rows, "
      f"{merged['age_bl'].notna().sum()} with age_bl")

# ── Create derived binary outcomes ────────────────────────────────────────
# Mobility disability: rosbsum > 0 → 1 (any disability), else 0
merged["mobility_disability_binary"] = (merged["rosbsum"] > 0).astype(float)
merged.loc[merged["rosbsum"].isna(), "mobility_disability_binary"] = np.nan

# Falls binary: falls > 0 → 1, else 0 (from the full 'falls' variable)
merged["falls_binary_from_falls"] = (merged["falls"] > 0).astype(float)
merged.loc[merged["falls"].isna(), "falls_binary_from_falls"] = np.nan
# Also keep the pre-computed falls_binary from all-valid-cycles if available
# Use falls_binary (from allv) as primary; fill gaps with derived
merged["falls_binary_final"] = merged["falls_binary"].fillna(
    merged["falls_binary_from_falls"]
)

# Cognitive impairment binary: dcfdx == 1 → NCI (0), dcfdx > 1 → impaired (1)
merged["cognitive_impairment_binary"] = (merged["dcfdx"] > 1).astype(float)
merged.loc[merged["dcfdx"].isna(), "cognitive_impairment_binary"] = np.nan
# Also from dcfdx_3gp: 1 → NCI, 2/3 → impaired
merged["cognitive_impairment_from_3gp"] = (merged["dcfdx_3gp"] > 1).astype(float)
merged.loc[merged["dcfdx_3gp"].isna(), "cognitive_impairment_from_3gp"] = np.nan
# Final: use dcfdx-based if available, else dcfdx_3gp
merged["cognitive_impairment_final"] = merged["cognitive_impairment_binary"].fillna(
    merged["cognitive_impairment_from_3gp"]
)

# ── Define feature buckets ────────────────────────────────────────────────
id_cols = ["sub_id", "projid", "fu_year", "wear_days", "study"]
daily_pa_cols = [c for c in gait.columns if c.startswith("daily_")]
gait_bout_cols = [c for c in gait.columns
                  if c not in id_cols and c not in daily_pa_cols
                  and c not in ["sub_id", "projid", "fu_year", "wear_days"]]

print(f"\nFeature buckets:")
print(f"  Daily PA variables: {len(daily_pa_cols)}")
print(f"  Gait bout metrics:  {len(gait_bout_cols)}")

# ── Save ──────────────────────────────────────────────────────────────────
merged.to_csv("output/merged_gait_clinical.csv", index=False)
print(f"\nSaved: output/merged_gait_clinical.csv ({merged.shape[0]} rows, {merged.shape[1]} cols)")

# ── Summary of outcome availability ───────────────────────────────────────
outcomes = {
    "parkinsonism_yn": "binary",
    "mobility_disability_binary": "binary",
    "falls_binary_final": "binary",
    "cognitive_impairment_final": "binary",
    "parksc": "continuous",
    "motor10": "continuous",
    "cogn_global": "continuous",
}
print("\nOutcome availability:")
for col, dtype in outcomes.items():
    n = merged[col].notna().sum()
    if dtype == "binary":
        pos = (merged[col] == 1).sum()
        print(f"  {col:35s}  n={n:5d}  pos={pos:4d}  ({100*pos/n:.1f}%)" if n > 0 else f"  {col:35s}  n=0")
    else:
        print(f"  {col:35s}  n={n:5d}  mean={merged[col].mean():.3f}  std={merged[col].std():.3f}")
