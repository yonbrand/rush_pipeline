"""
Refactored Merge Script for RUSH Gait + Clinical Data
----------------------------------------------------
- Robust path handling
- Automatic directory creation
- Cleaner structure
- Reusable functions
- Easier to maintain / extend
"""

import os
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = "N:/Gait-Neurodynamics by Names/Yonatan/RUSH/rush_pipeline"
INPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_DIR = os.path.join(INPUT_DIR, "tables")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "gait": os.path.join(INPUT_DIR, "subject_summary.csv"),
    "excel": os.path.join(INPUT_DIR, "wrist_sensor_metadata_Yonatan.xlsx"),
}

EXPORT_FLAGS = {
    "ABL": True,
    "LV": True,
    "ALL_VISITS": True,
    "POSTMORTEM": True,
}

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    gait = pd.read_csv(FILES["gait"])
    excel = FILES["excel"]

    sheets = {
        "abl": pd.read_excel(excel, sheet_name="ABL"),
        "lv": pd.read_excel(excel, sheet_name="LV"),
        "allv": pd.read_excel(excel, sheet_name="Data from all valid cycles"),
        "postmortem": pd.read_excel(excel, sheet_name="Postmortem Indices"),
    }

    print(f"Gait data: {gait.shape}")
    return gait, sheets

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def derive_binary_outcomes(df):
    df = df.copy()

    df["mobility_disability_binary"] = (df["rosbsum"] > 0).astype(float)
    df.loc[df["rosbsum"].isna(), "mobility_disability_binary"] = np.nan

    if "falls" in df.columns:
        df["falls_binary"] = (df["falls"] > 0).astype(float)
        df.loc[df["falls"].isna(), "falls_binary"] = np.nan
    elif "falls_binary" not in df.columns:
        df["falls_binary"] = np.nan

    if "dcfdx" in df.columns:
        df["cognitive_impairment"] = (df["dcfdx"] > 1).astype(float)
        df.loc[df["dcfdx"].isna(), "cognitive_impairment"] = np.nan
    elif "dcfdx_3gp" in df.columns:
        df["cognitive_impairment"] = (df["dcfdx_3gp"] > 1).astype(float)
        df.loc[df["dcfdx_3gp"].isna(), "cognitive_impairment"] = np.nan
    else:
        df["cognitive_impairment"] = np.nan

    if "age_bl" in df.columns and "fu_year" in df.columns:
        df["age_at_visit"] = df["age_bl"] + df["fu_year"]

    return df

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def save(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def print_summary(df, label):
    print(f"\n[{label}] shape={df.shape}, subjects={df['projid'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════
# MERGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def merge_abl(gait, abl):
    df = gait.merge(abl, on=["projid", "fu_year"], how="inner")
    df = derive_binary_outcomes(df)
    print_summary(df, "ABL")
    save(df, "merged_gait_clinical_abl.csv")


def merge_lv(gait, lv, abl):
    df = gait.merge(lv, on=["projid", "fu_year"], how="inner")

    demo = abl[["projid", "age_bl", "msex", "educ", "race7"]].drop_duplicates("projid")
    df = df.merge(demo, on="projid", how="left")

    df = derive_binary_outcomes(df)
    print_summary(df, "LV")
    save(df, "merged_gait_clinical_lv.csv")


def merge_all_visits(gait, allv, abl):
    allv = allv.rename(columns={
        "parkinsonism_YN": "parkinsonism_yn",
        "falls_yn": "falls_binary",
    }).drop_duplicates(["projid", "fu_year"])

    df = gait.merge(allv, on=["projid", "fu_year"], how="inner")

    demo = abl[["projid", "age_bl", "msex", "educ", "race7"]].drop_duplicates("projid")
    df = df.merge(demo, on="projid", how="left")

    df = derive_binary_outcomes(df)
    print_summary(df, "ALL_VISITS")
    save(df, "merged_gait_clinical_allvisits.csv")


def merge_postmortem(gait, abl, postmortem):
    df = gait.merge(abl, on=["projid", "fu_year"], how="inner")
    df = df.merge(postmortem.drop(columns=["study"], errors="ignore"), on="projid", how="inner")

    df = derive_binary_outcomes(df)
    print_summary(df, "POSTMORTEM")
    save(df, "merged_gait_clinical_postmortem.csv")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    gait, sheets = load_data()

    if EXPORT_FLAGS["ABL"]:
        merge_abl(gait, sheets["abl"])

    if EXPORT_FLAGS["LV"]:
        merge_lv(gait, sheets["lv"], sheets["abl"])

    if EXPORT_FLAGS["ALL_VISITS"]:
        merge_all_visits(gait, sheets["allv"], sheets["abl"])

    if EXPORT_FLAGS["POSTMORTEM"]:
        merge_postmortem(gait, sheets["abl"], sheets["postmortem"])

    print("\nDone.")


if __name__ == "__main__":
    main()
