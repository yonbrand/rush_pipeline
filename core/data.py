"""
Data loading + canonical feature-set definitions for the autoresearch project.

Only this module knows where the merged CSV lives. Only this module knows the
dev/lockbox split files. The dev loader explicitly refuses to return lockbox
subjects; the lockbox loader is gated and only `final_evaluation.py` may call it.
"""
import os
import json
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "outputs", "tables", "merged_gait_clinical_abl.csv")
DEV_IDS = os.path.join(REPO, "dev_ids.csv")
LOCKBOX_IDS = os.path.join(REPO, "lockbox_ids.csv")

ID_COL = "projid"

# ---------------------------------------------------------------------------
# Feature-set definitions (FROZEN after first successful baseline run).
# Demographics are appended to every set by `prepare_data(..., demographics=True)`.
# ---------------------------------------------------------------------------

DEMOGRAPHIC_COLS = ["age_at_visit", "msex", "educ"]

# Columns that are never features (IDs, outcomes, redundant slopes, etc.)
EXCLUDE_FROM_FEATURES = set([
    "sub_id", "projid", "fu_year", "wear_days", "study", "device",
    "age_bl", "msex", "educ", "race7", "age_at_visit",
    # Outcomes (all 7) and close cousins
    "parkinsonism_yn", "dcfdx", "dementia", "cpd_ever", "cogdx",
    "rosbsum", "falls", "falls_binary", "mobility_disability_binary",
    "cognitive_impairment",
    "cogn_global", "cogn_ep", "cogn_po", "cogn_ps", "cogn_se", "cogn_wo",
    "cogng_demog_slope", "time_lastce2dod", "age_death",
    "cts_mmse30", "parksc", "motor10", "gait_speed", "scaled_to",
    "pert_noact_avg", "tactivity_acth_avgnew", "tactivity_d_avgnew",
    "gaitsc", "bradysc", "rigidsc", "tremsc",
    # Other clinical scales (potential confounds / partial outcomes)
    "r_depres", "bmi", "med_con_sum_cum", "vasc_risks_sum",
    "iadlsum", "katzsum", "late_life_cogact_freq", "phys5itemsum",
    "late_life_soc_act", "park_rx",
    "motor10_demog_slope", "sqrt_parksc_demog_slope",
    "motor_dexterity", "motor_gait", "motor_handstreng",
    # Old rest-activity proxies (replaced by rar_ columns)
    "is", "iv", "kar", "kra",
])


def _classify_columns(df):
    """Return (gait_bout_cols, daily_pa_cols, sleep_cols)."""
    daily_pa_cols = [
        c for c in df.columns
        if c.startswith("daily_pa_mean_") or c.startswith("daily_pa_std_")
        or c.startswith("tdpa_")
    ]
    sleep_cols = [
        c for c in df.columns
        if (c.startswith("sleep_") or c.startswith("rar_"))
        and c != "sleep_n_nights"
    ]
    gait_bout_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and c not in daily_pa_cols
        and c not in sleep_cols
        and "_prob_bin" not in c   # prob_bin == freq_bin / sum (perfect collinear)
    ]
    return gait_bout_cols, daily_pa_cols, sleep_cols


def feature_sets(df):
    """
    Return the 7 preregistered ladder rungs as {name: [cols]}.
    Demographics are NOT included here — append at prepare_data time.
    """
    gait, pa, sleep = _classify_columns(df)
    return {
        "Demographics only": [],
        "+ 8ft Gait Speed": ["gait_speed"],
        "+ Daily PA": pa,
        "+ Sleep / RAR": sleep,
        "+ Gait Bout": gait,
        "Full Sensor (Gait + PA + Sleep)": gait + pa + sleep,
        "+ Gait Bout + 8ft": gait + ["gait_speed"],
    }


def load_dev():
    """Load the development subset only. Refuses to leak lockbox IDs."""
    df = pd.read_csv(CSV)
    dev = pd.read_csv(DEV_IDS)[ID_COL].tolist()
    sub = df[df[ID_COL].isin(dev)].reset_index(drop=True)
    if not os.path.exists(LOCKBOX_IDS):
        raise RuntimeError("lockbox_ids.csv missing - run make_split.py first")
    lock = pd.read_csv(LOCKBOX_IDS)[ID_COL].tolist()
    overlap = set(sub[ID_COL]).intersection(lock)
    if overlap:
        raise RuntimeError(f"Lockbox leakage: {len(overlap)} subjects appear in dev")
    return sub


def prepare_data(df, feature_cols, outcome_col, demographics=True):
    """Subset cols, drop rows with missing outcome, return (X, y, feature_names)."""
    feats = list(feature_cols) + (DEMOGRAPHIC_COLS if demographics else [])
    if not feats:
        raise ValueError("Empty feature list")
    cols = feats + [outcome_col]
    sub = df[cols].dropna(subset=[outcome_col])
    y = sub[outcome_col].values
    X = sub[feats].values.astype(float)
    return X, y, feats


# ---------------------------------------------------------------------------
# Outcome registry
# ---------------------------------------------------------------------------

BINARY_OUTCOMES = {
    "mobility_disability_binary": "Mobility Disability",
    "cognitive_impairment": "Cognitive Impairment",
    "falls_binary": "Falls",
    "parkinsonism_yn": "Parkinsonism (mUPDRS-derived; exploratory)",
}
CONTINUOUS_OUTCOMES = {
    "cogn_global": "Global Cognition",
    "motor10": "Motor Composite (mUPDRS-derived; exploratory)",
    "parksc": "Parkinsonism Score (mUPDRS-derived; exploratory)",
}

# Co-primary, secondary, exploratory designations (from preregistration)
CO_PRIMARY = {
    "cogn_global": "regression",                        # R^2
    "mobility_disability_binary": "classification",     # AP
}
SECONDARY = {
    "falls_binary": "classification",
    "cognitive_impairment": "classification",
}
EXPLORATORY = {
    "parksc": "regression",
    "motor10": "regression",
    "parkinsonism_yn": "classification",
}


# ---------------------------------------------------------------------------
# Block-PCA domain map (used by some selection strategies)
# ---------------------------------------------------------------------------

GAIT_DOMAINS = {
    "Speed":              ["bout_speed"],
    "Gait Length":        ["bout_gait_length_indirect", "bout_gait_length"],
    "Cadence":            ["bout_cadence"],
    "Regularity":         ["bout_regularity_eldernet", "bout_regularity_sp"],
    "Gait Quantity":      ["bout_duration", "bout_total", "daily_n_bouts",
                           "daily_step", "daily_walking", "n_bouts"],
    "Bout Intensity":     ["bout_pa_amplitude", "bout_pa_variability"],
    "Within-Bout Var":    ["var_var"],
    "Spectral/Complexity":["bout_entropy", "bout_dom", "bout_psd_amp",
                           "bout_psd_width", "bout_psd_slope"],
    "Day-to-Day":         ["stability_", "dist_"],
    "Temporal Pattern":   ["tod_"],
}
