# Pre-registration — Daily-Living Wrist Gait vs PA vs Clinic Gait Speed

**Date written:** 2026-04-14
**Author:** Autoresearch agent (Claude Code) acting on behalf of the project owner.
**Status:** FROZEN. This document is committed once and never edited. Any post-hoc deviation must be reported in the manuscript under "Deviations from preregistration" with the original wording preserved here.

---

## 1. Scientific question

Do daily-living gait *quality* metrics extracted from a ~10-day wrist accelerometer recording carry clinical information that (a) the traditional 8-foot in-clinic gait speed test does not, and (b) cannot be recovered from broad physical-activity summaries alone?

## 2. Data

- Source: `outputs/tables/merged_gait_clinical_abl.csv` (790 subjects × 681 columns), one row per subject at analytic baseline (ABL) visit.
- Cohort: Rush Memory and Aging Project (MAP) / Religious Orders Study (ROS).
- Sensor: wrist-worn accelerometer, ~10 days continuous recording.
- Subject ID: `projid`.

## 3. Co-primary outcomes

1. **`cogn_global`** — global cognition composite (continuous, regression). Primary metric: **R²** (mean across outer folds). Coverage 787/790. Not derived from the 8-ft walk test, so the comparison "Gait Bout vs 8-ft Speed" is not circular.
2. **`mobility_disability_binary`** — self-reported mobility disability (binary, classification). Primary metric: **average precision (AP)**. Coverage 785/790; positive class ≈ 46% (well-balanced). Not derived from the 8-ft walk test.

These two outcomes are conceptually independent (cognitive vs functional axis); a result that holds across both is a substantively stronger claim than either alone.

## 4. Primary claim and statistical test

For each co-primary outcome, the primary comparison is:

> **Gait Bout features (+ demographics) vs 8-ft Gait Speed (+ demographics)**, evaluated with the same 5-fold × 3-repeat outer CV (random_state = 42), best (model × selection) per feature set, paired per-fold scores, **Nadeau–Bengio (2003) corrected repeated-k-fold t-test**.

The two co-primary tests are corrected for multiplicity using **Holm–Bonferroni at family-wise α = 0.05**. The strongest version of the primary claim requires *both* corrected p-values < 0.05 with positive effect-size point estimates (ΔR² > 0 and ΔAP > 0). Either one passing alone is reported as a weaker, still-publishable finding.

## 5. Incremental value ladder (conceptual spine of the paper)

The following nested feature sets are all evaluated under the identical CV protocol so the reader sees what each layer adds. Demographics (`age_at_visit`, `msex`, `educ`) are included in every set so all comparisons are apples-to-apples:

1. **Demographics only** — floor.
2. **Demographics + 8-ft gait speed** — current standard-of-care benchmark.
3. **Demographics + Daily PA** — broad activity volume only.
4. **Demographics + Sleep / RAR** — sleep architecture + rest-activity rhythm only.
5. **Demographics + Gait Bout** — daily-living gait *quality* (the hero set).
6. **Demographics + Gait Bout + Daily PA + Sleep** — full sensor stack.
7. **Demographics + Gait Bout + 8-ft speed** — does the clinic test add anything *on top of* daily-living gait?

## 6. Secondary (confirmatory) outcomes

- **`falls_binary`** (binary, AP).
- **`cognitive_impairment`** (binary, AP).

Holm–Bonferroni correction is applied across these two secondary tests.

## 7. Exploratory outcomes (with circularity caveat)

`parksc`, `motor10`, `parkinsonism_yn` — all derived from the modified UPDRS motor exam, which includes the 8-ft walk test as a scoring component. Any "Gait Bout vs 8-ft Speed" comparison on these outcomes is contaminated. Reported descriptively only, in a clearly labelled exploratory / sensitivity section. Never used as evidence for the primary claim.

## 8. Held-out lockbox

- **Stratified 15% of subjects**, written to `lockbox_ids.csv` on first run.
- Stratification variable: `mobility_disability_binary` (the co-primary binary outcome with the fullest n and most balanced distribution).
- Random seed: `20260414`.
- Subjects with missing `mobility_disability_binary` are stratified by an "missing" stratum so they are not silently excluded from one side of the split.
- Balance is verified post-hoc on `cogn_global` (quartiles) and `cognitive_impairment`. Imbalance is logged but the split is not redrawn.
- The lockbox is **never read** during iteration. Only `final_evaluation.py` opens it, and only **once**, after the agent declares itself done.

## 9. Hard methodological rules

These are non-negotiable; any idea that violates them is rejected without running.

1. All data-dependent preprocessing (imputation, scaling, variance filter, correlation filter, feature selection, PCA, SHAP-based selection, target encoding) is implemented as an sklearn-compatible transformer and lives **inside** the CV `Pipeline`. Re-fit per training fold.
2. Hyperparameters and selection `k` are tuned in the inner CV loop only. Outer-loop scores are reported untouched.
3. The outer CV `random_state` is **fixed at 42** for the full project. Fold pairing across feature sets is required for the Nadeau–Bengio test.
4. The lockbox is sacred (see §8).
5. Co-primary outcomes are fixed at `cogn_global` and `mobility_disability_binary`. They cannot change after seeing results.
6. Subjects may be dropped only for pre-specified reasons (missing outcome; pre-specified wear-time / bout-count quality threshold). Every exclusion is logged in `exclusions.md` with count and reason.
7. Feature-set membership is **frozen after the first successful baseline run**. Demographics appear in every set. Within the Gait Bout set, new features may be engineered, but a feature cannot migrate between groups.
8. Every reported number carries an uncertainty (SD across outer folds at minimum; bootstrap or Nadeau–Bengio CIs for headline contrasts).
9. Holm–Bonferroni correction is applied within each outcome's pairwise comparison family and across the secondary outcomes.
10. No peeking at lockbox data, labels, or derived statistics during iteration.

## 10. Stop criteria

Iteration stops when **any** of the following is true:

1. ≥ 15 experiments run AND top-3 configurations within 0.005 R² (cogn_global) of each other.
2. Both co-primary tests pass at family-wise α = 0.05 (Holm-corrected) AND robust under ≥ 2 sensitivity analyses.
3. Improvement menu exhausted; further ideas would require new data or violate a rigor rule.
4. **Hard ceiling: 50 experiments**, regardless of outcome.

On stop, freeze the current best configuration. Run `final_evaluation.py` exactly once. Report lockbox numbers alongside dev numbers.

## 11. Software & seeds

- Python; numpy, scipy, pandas, scikit-learn, xgboost, matplotlib. Versions pinned in `code/REPRODUCE.md` after freeze.
- Random seed for outer CV split: 42 (fixed).
- Random seed for lockbox split: 20260414 (fixed).
- Random seed for stochastic models (RF, XGB, L1 selectors): 42 (fixed).

---

End of preregistration. Do not edit.
