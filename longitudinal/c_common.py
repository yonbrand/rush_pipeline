"""
Shared utilities for Analysis C (postmortem neuropathology).

Implements the frozen preregistration at
`runs/longitudinal/c_postmortem/preregistration.md` (2026-04-23):
cohort construction (Cohort-D n=187, Cohort-8 n=146), outcome binarization
at preregistered cutpoints, rung definitions per cohort, and the forced
covariate list. Any change here must be logged in `deviations.md`.
"""
import os
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTMORTEM_CSV = os.path.join(REPO, "outputs", "tables",
                              "merged_gait_clinical_postmortem.csv")
RUN_DIR = os.path.join(REPO, "runs", "longitudinal", "c_postmortem")

ID_COL = "projid"

# Forced covariates — appended to every rung (per preregistration §Forced covariates).
FORCED_COVARIATES = ["age_bl", "msex", "educ", "age_death", "pmi",
                     "time_lastce2dod"]

# Expected approved cohort size (frozen).
N_APPROVED_EXPECTED = 187
N_COHORT8_EXPECTED = 146

# Outer CV (matches cross-sectional + longitudinal A/B).
OUTER_SEED = 42
N_SPLITS = 5
N_REPEATS = 3

# Outcome registry: key -> (source_col, cutpoint_fn, label_if_pos, domain).
# Cutpoint functions map a pandas Series of raw values to a {0,1} Series,
# leaving NaN where source is NaN.
def _native(s):
    return s

def _ge(thresh):
    def _cut(s):
        return (s >= thresh).astype("float").where(s.notna())
    return _cut


OUTCOMES = {
    "ad_adnc": {
        "source": "ad_adnc",
        "cutpoint": _native,
        "cut_label": "native 0/1",
        "domain": "Alzheimer's disease neuropath",
    },
    "lb_7reg": {
        "source": "lb_7reg",
        "cutpoint": _native,
        "cut_label": "native 0/1",
        "domain": "Lewy body (7-region)",
    },
    "tdp_st4_bin": {
        "source": "tdp_st4",
        "cutpoint": _ge(1),
        "cut_label": ">=1 vs 0 (any-vs-none)",
        "domain": "TDP-43 staging",
    },
    "arteriol_scler_bin": {
        "source": "arteriol_scler",
        "cutpoint": _ge(2),
        "cut_label": ">=2 vs <2 (moderate-severe, MAP/ROS convention)",
        "domain": "Vascular - arteriolosclerosis",
    },
    "cvda_4gp2_bin": {
        "source": "cvda_4gp2",
        "cutpoint": _ge(2),
        "cut_label": ">=2 vs <2 (moderate-severe CAA, MAP/ROS convention)",
        "domain": "Vascular - cerebrovascular/amyloid angiopathy",
    },
    "henl_4gp_bin": {
        "source": "henl_4gp",
        "cutpoint": _ge(1),
        "cut_label": ">=1 vs 0 (any-vs-none; grade>=2 n=10, too sparse)",
        "domain": "Lewy body (HENL regions)",
    },
}

# Columns that are never features (IDs, outcomes, redundant slopes, forced cov,
# raw ordinal sources). Anything not here and not in FORCED_COVARIATES is eligible
# for the gait-bout feature set.
_OUTCOME_COLS = {o["source"] for o in OUTCOMES.values()} | set(OUTCOMES.keys())

_EXCLUDE_FROM_FEATURES = _OUTCOME_COLS | set(FORCED_COVARIATES) | {
    "sub_id", "projid", "fu_year", "wear_days", "study", "device",
    "age_at_visit", "race7",
    # 8ft comparator — lives in its own rung (+ 8ft), never in the gait-bout set.
    "gait_speed",
    # Other clinical / pathology columns — not features.
    "npath_approved",
    "path_pd_modsev", "tangsqrt", "amylsqrt",
    # Other pathology scales in postmortem CSV (prevent circular leakage:
    # these are themselves neuropathology measurements, not gait features).
    "caa_4gp", "caa_3gp", "lb_any",
    "ci_num2_gct", "ci_num2_mct",   # chronic infarct (gross/macro cortical)
    "hip_scl_mid",                   # hippocampal sclerosis (midline)
    # Daily PA / sleep / RAR — out of scope for Analysis C (gait-bout only).
    # Kept narrow; anything starting with daily_pa_ or sleep_ or rar_ is
    # filtered below.
    # Redundant 8ft comparator container name.
    "scaled_to",
    # Clinical composites / candidate confounds.
    "cogn_global", "cogn_ep", "cogn_po", "cogn_ps", "cogn_se", "cogn_wo",
    "cts_mmse30", "parksc", "motor10", "gaitsc", "bradysc", "rigidsc",
    "tremsc", "rosbsum", "falls", "falls_binary",
    "mobility_disability_binary", "cognitive_impairment",
    "dcfdx", "dementia", "cpd_ever", "cogdx", "parkinsonism_yn",
    "iadlsum", "katzsum",
    "r_depres", "bmi", "med_con_sum_cum", "vasc_risks_sum",
    "late_life_cogact_freq", "phys5itemsum", "late_life_soc_act", "park_rx",
    "motor_dexterity", "motor_gait", "motor_handstreng",
    "pert_noact_avg", "tactivity_acth_avgnew", "tactivity_d_avgnew",
    "cogng_demog_slope", "motor10_demog_slope", "sqrt_parksc_demog_slope",
    # Legacy rest-activity proxies.
    "is", "iv", "kar", "kra",
}


def load_approved_cohort():
    """
    Load n=187 postmortem approved cohort with binarized outcomes.

    Returns the DataFrame with:
    - Only rows where `npath_approved == 1` (drops 603 others).
    - One row per `projid` (asserts no duplicates).
    - Added `_bin` columns for the 4 ordinal outcomes (native binaries
      left untouched).
    """
    df = pd.read_csv(POSTMORTEM_CSV)
    approved = df.loc[df["npath_approved"] == 1].copy().reset_index(drop=True)

    # Safety: one row per projid.
    if approved[ID_COL].duplicated().any():
        raise RuntimeError(
            f"Duplicate projid in approved cohort: "
            f"{int(approved[ID_COL].duplicated().sum())} extras.")

    if len(approved) != N_APPROVED_EXPECTED:
        raise RuntimeError(
            f"Approved cohort size mismatch: got {len(approved)}, "
            f"preregistration expects {N_APPROVED_EXPECTED}.")

    # Forced covariates must be fully non-null in the approved cohort
    # (preregistration §Forced covariates: all 187/187).
    for c in FORCED_COVARIATES:
        n = int(approved[c].notna().sum())
        if n != N_APPROVED_EXPECTED:
            raise RuntimeError(
                f"Forced covariate {c} has {n}/{N_APPROVED_EXPECTED} non-null; "
                f"preregistration requires full coverage.")

    # Add binarized outcome columns.
    for key, spec in OUTCOMES.items():
        if spec["source"] == key:
            continue  # native binary — no new column needed
        approved[key] = spec["cutpoint"](approved[spec["source"]])

    return approved


def build_cohorts(df_approved):
    """
    Return {'cohort_d': df_187, 'cohort_8': df_146}.

    Cohort-D: all approved (gait_speed may be NaN; gait_speed not used here).
    Cohort-8: approved with non-null gait_speed. Paired across all 4 rungs
              in Cohort-8 so contrasts are apples-to-apples on identical
              subjects.
    """
    cohort_d = df_approved.copy()
    cohort_8 = df_approved.loc[df_approved["gait_speed"].notna()].copy()
    cohort_8 = cohort_8.reset_index(drop=True)
    if len(cohort_8) != N_COHORT8_EXPECTED:
        raise RuntimeError(
            f"Cohort-8 size mismatch: got {len(cohort_8)}, "
            f"preregistration expects {N_COHORT8_EXPECTED}.")
    return {"cohort_d": cohort_d, "cohort_8": cohort_8}


def get_gait_bout_cols(df):
    """
    Gait-bout feature columns: everything not explicitly excluded and not
    a daily-PA / sleep / RAR column. Returns sorted list.
    """
    exclude = set(_EXCLUDE_FROM_FEATURES)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("daily_pa_") or c.startswith("tdpa_"):
            continue
        if c.startswith("sleep_") or c.startswith("rar_"):
            continue
        if "_prob_bin" in c:  # perfect collinear with freq_bin
            continue
        cols.append(c)
    return sorted(cols)


def rungs_for_cohort(df, cohort_key):
    """
    Return {rung_name: feature_col_list} for the given cohort.

    Cohort-D uses 2 rungs (primary contrast is #3 vs #1):
      1. Demographics
      3. + Gait Bout

    Cohort-8 uses 4 rungs (primary contrast is #3 vs #2; secondary is #4 vs #2):
      1. Demographics
      2. + 8ft
      3. + Gait Bout
      4. + Gait Bout + 8ft
    """
    gait = get_gait_bout_cols(df)
    if cohort_key == "cohort_d":
        return {
            "Demographics": list(FORCED_COVARIATES),
            "+ Gait Bout": list(FORCED_COVARIATES) + gait,
        }
    if cohort_key == "cohort_8":
        return {
            "Demographics": list(FORCED_COVARIATES),
            "+ 8ft": list(FORCED_COVARIATES) + ["gait_speed"],
            "+ Gait Bout": list(FORCED_COVARIATES) + gait,
            "+ Gait Bout + 8ft": list(FORCED_COVARIATES) + gait + ["gait_speed"],
        }
    raise ValueError(f"Unknown cohort: {cohort_key}")


def prepare_xy(df, feature_cols, outcome_key):
    """
    Given a cohort DataFrame and a rung's feature cols, return (X, y,
    feature_names) with rows missing the outcome dropped. Outcome cutpoints
    were applied already in load_approved_cohort, so outcome_key is the
    (possibly binarized) column name directly.
    """
    cols = list(feature_cols) + [outcome_key]
    sub = df[cols].dropna(subset=[outcome_key])
    y = sub[outcome_key].values.astype(int)
    X = sub[feature_cols].values.astype(float)
    return X, y, list(feature_cols)


def ensure_run_dir(subdir=None):
    """Create RUN_DIR (and optional subdir) and return its path."""
    path = RUN_DIR if subdir is None else os.path.join(RUN_DIR, subdir)
    os.makedirs(path, exist_ok=True)
    return path
