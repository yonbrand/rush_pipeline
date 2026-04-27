"""
Create the stratified lockbox / dev split for the Rush wrist-gait analysis.

Run exactly once. Writes:
  - lockbox_ids.csv  (15% of subjects, never read during iteration)
  - dev_ids.csv      (the other 85%, used for all model development)
  - split_balance.json (post-hoc balance check)

Stratification: mobility_disability_binary (with an explicit 'missing'
stratum so the 5 NaN subjects are not silently lost).

Random seed: 20260414 (fixed in preregistration.md).
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(REPO, "outputs", "tables", "merged_gait_clinical_abl.csv")

SEED = 20260414
LOCKBOX_FRAC = 0.15
ID_COL = "projid"
STRAT = "mobility_disability_binary"

df = pd.read_csv(CSV)
print(f"Loaded {df.shape[0]} subjects, {df.shape[1]} columns")

assert df[ID_COL].is_unique, f"{ID_COL} not unique"

strat = df[STRAT].astype("Int64").astype(str).fillna("missing").values
print(f"Stratification counts on '{STRAT}':")
for v, c in pd.Series(strat).value_counts().items():
    print(f"  {v}: {c}")

ids = df[ID_COL].values
dev_ids, lock_ids = train_test_split(
    ids, test_size=LOCKBOX_FRAC, random_state=SEED, stratify=strat,
)

assert len(set(dev_ids).intersection(lock_ids)) == 0, "leakage between dev and lockbox"
assert len(dev_ids) + len(lock_ids) == len(ids), "subject count mismatch"

pd.DataFrame({ID_COL: sorted(dev_ids)}).to_csv(os.path.join(REPO, "dev_ids.csv"), index=False)
pd.DataFrame({ID_COL: sorted(lock_ids)}).to_csv(os.path.join(REPO, "lockbox_ids.csv"), index=False)
print(f"\nWrote {len(dev_ids)} dev subjects, {len(lock_ids)} lockbox subjects")

# ── Post-hoc balance check ───────────────────────────────────────────────────
dev = df[df[ID_COL].isin(dev_ids)]
lock = df[df[ID_COL].isin(lock_ids)]

balance = {"seed": SEED, "n_dev": int(len(dev)), "n_lockbox": int(len(lock))}

for col in [STRAT, "cognitive_impairment", "falls_binary", "parkinsonism_yn"]:
    if col not in df.columns:
        continue
    balance[f"{col}_dev_mean"] = float(dev[col].dropna().mean())
    balance[f"{col}_lock_mean"] = float(lock[col].dropna().mean())
    balance[f"{col}_dev_n"] = int(dev[col].notna().sum())
    balance[f"{col}_lock_n"] = int(lock[col].notna().sum())

# cogn_global quartiles (using full-sample edges)
edges = np.nanquantile(df["cogn_global"].values, [0.25, 0.5, 0.75])
def qbin(x):
    if pd.isna(x): return "missing"
    if x <= edges[0]: return "Q1"
    if x <= edges[1]: return "Q2"
    if x <= edges[2]: return "Q3"
    return "Q4"

balance["cogn_global_quartile_dev"] = dict(dev["cogn_global"].apply(qbin).value_counts())
balance["cogn_global_quartile_lock"] = dict(lock["cogn_global"].apply(qbin).value_counts())
balance["age_dev_mean"] = float(dev["age_at_visit"].mean())
balance["age_lock_mean"] = float(lock["age_at_visit"].mean())
balance["msex_dev_mean"] = float(dev["msex"].mean())
balance["msex_lock_mean"] = float(lock["msex"].mean())

print("\n-- Balance check --")
print(json.dumps(balance, indent=2, default=str))

with open(os.path.join(REPO, "split_balance.json"), "w") as f:
    json.dump(balance, f, indent=2, default=str)
print("\nWrote split_balance.json")
