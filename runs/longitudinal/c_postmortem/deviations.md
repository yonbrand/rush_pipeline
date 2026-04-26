# Analysis C — Deviations log

This file records every documented change from (a) the original longitudinal brief (Analysis C sub-section, which specified 2 outcomes, no Holm, 217-subject cohort, LR-l2 only, dev/lockbox split) and (b) the v1 post-council draft plan. All entries are dated and user-approved. Logged **before** any fit was run on the Analysis C cohort.

Freeze point: `preregistration.md` (FROZEN 2026-04-23). Any further change to the analysis plan must be appended below with date and rationale.

---

## 2026-04-23 — Deviations from original brief (Analysis C sub-section)

### 1. Outcome family expanded from 2 → 6

- **Original brief:** `ad_adnc`, `henl_4gp`.
- **Preregistration:** `ad_adnc`, `lb_7reg`, `tdp_st4_bin`, `arteriol_scler_bin`, `cvda_4gp2_bin`, `henl_4gp_bin`.
- **Why:** Pre-fit data inspection (2026-04-23) found 4 additional pathology outcomes with approved n ≥ 186 spanning 3 new neuropath domains (Lewy 7-region, TDP-43, vascular/arteriolosclerosis, vascular/CAA). Original 2-outcome set systematically under-samples the pathology domains the paper claims to probe.
- **Decided before any fit.** Confirmed via column-level `npath_approved==1` non-null counts on `merged_gait_clinical_postmortem.csv`.

### 2. Ordinal outcomes binarized at preregistered cutpoints

- **v1 draft:** proposed cumulative-logit via statsmodels OrderedModel for `tdp_st4`, `arteriol_scler`, `cvda_4gp2`.
- **Preregistration:** all 4 ordinal outcomes binarized.
  - `tdp_st4_bin`: ≥1 vs 0 (any-vs-none)
  - `arteriol_scler_bin`: ≥2 vs <2 (moderate-severe per MAP/ROS convention)
  - `cvda_4gp2_bin`: ≥2 vs <2 (moderate-severe CAA per MAP/ROS convention)
  - `henl_4gp_bin`: ≥1 vs 0 (any-vs-none; grade ≥2 has n=10, too sparse for higher cut)
- **Why:** Council (GPT, HIGH) flagged OrderedModel fragility at n≈190 — no regularization support, PO-assumption risk, class dropout across folds. Binarization lets the entire family share a single `LogisticRegression`-backed inner-CV grid and keeps inference uniform.

### 3. Holm×6 multiplicity applied within each contrast

- **Original brief:** "no Holm — this is supplementary."
- **Preregistration:** Holm×6 applied **within each of the two contrast families independently** (Contrast 1: vs Demographics on Cohort-D; Contrast 2: vs +8ft on Cohort-8). Raw p-values also reported per outcome.
- **Why:** Belt-and-suspenders defensibility when reporting 6 outcomes across 4 domains. A single Holm×12 would over-penalize by mixing unrelated comparisons on different cohorts. Both Holm-corrected and raw p-values exposed so reviewers can read either lens.

### 4. Cohort size: n=187, not 217

- **v1 draft:** cited n=217 for the approved cohort.
- **Correction:** n=187 with `npath_approved==1`. n=217 was `npath_approved` non-null count (includes rows where the column exists but equals 0 or is otherwise not an approved autopsy).
- **Why:** Data inspection error in v1. Corrected via `df[df['npath_approved'] == 1].shape[0]` on `merged_gait_clinical_postmortem.csv`.

### 5. `time_lastce2dod` promoted to forced covariate

- **v1 draft:** mentioned in `limitations.md` only.
- **Preregistration:** forced into every rung (Demographics, +8ft, +Gait Bout, +Gait Bout+8ft).
- **Why:** Council (GPT, MED) flagged time-to-death confounding. Subjects who died sooner after their last clinical evaluation have shorter "decay interval" between measurement and autopsy — a confound that is cheap to adjust for directly rather than stratify. 187/187 non-null, no imputation needed.

### 6. Nadeau-Bengio corrected t replaces paired bootstrap on fold deltas

- **v1 draft:** proposed paired bootstrap (2000 resamples) on 15 outer-fold ΔAP values.
- **Preregistration:** Nadeau-Bengio (2003) corrected repeated-CV t-test via `core.cv.corrected_repeated_kfold_cv_test`.
- **Why:** Council (GPT, HIGH) flagged the paired bootstrap as statistically invalid — bootstrapping 15 fold-level deltas ignores the correlation structure induced by shared training data across folds, underestimating variance. Nadeau-Bengio is the repo-standard corrected-variance test used in cross-sectional, Analysis A, and Analysis B v2.

### 7. Added `(+Gait Bout) vs Demographics` as a co-primary contrast

- **v1 draft:** single primary contrast `(+Gait Bout) vs (+8ft)`.
- **Preregistration:** two co-primary contrasts — `(+Gait Bout) vs Demographics` on Cohort-D (n=187) **and** `(+Gait Bout) vs (+8ft)` on Cohort-8 (n=146). Holm×6 applied within each contrast family independently.
- **Why:** Council split (Claude: "vs Demographics is the right question — does sensor tell us anything the medical record can't"; Gemini: "vs 8ft is the right question — does sensor beat the existing clinical tool"). Both are valid, non-overlapping questions. Making them co-primary with two parallel Holm corrections preserves the integrity of each without collapsing them into a choice. Cohort-per-contrast rules (see preregistration) keep each contrast paired on identical subjects.

### 8. No lockbox split within Analysis C

- **v1 draft:** proposed holding out a 29-subject lockbox (158 dev / 29 lockbox).
- **Preregistration:** nested CV on the **full n=187 approved cohort**. No lockbox split.
- **Why:** User-directed (2026-04-23). At n=29 a held-out slice gives no honest error estimate — the confidence interval on lockbox AP would be wider than the effect size we're probing. Forfeits 18% of an already-small cohort. Matches Analysis A's posture (nested CV only, no lockbox). Orthogonal to the cross-sectional lockbox's purpose (held out for cross-sectional co-primaries, already evaluated and closed).

### 9. Model + hyperparameter search folded into inner CV

- **v1 draft:** deterministic `LogisticRegression(penalty='l2')` with `C ∈ logspace(-3, 2, 12)` as the only tunable. No Selection feature strategy.
- **Preregistration:** inner 3-fold `GridSearchCV` searches over LR-l2, LR-l1, LR-elasticnet, SVC-RBF (see entry #10 for RF and entry #11 for HistGB), plus BlockPCA `var_kept` ∈ {0.80, 0.90}.
- **Why:** Postmortem neuropathology is a different domain from cognition/motor outcomes that the "LR-l2 winner" was tuned on in cross-sectional and longitudinal A/B. Locking in LR-l2 a priori is an under-powered prior. Model selection occurs strictly inside the inner fold per outer fold — one prediction vector per outer fold reaches Nadeau-Bengio, so no inferential DoF added. User-approved.

### 10. RandomForest added to the inner-CV grid

- **Cross-sectional / Analyses A, B posture:** "no tree ensembles — prior evidence: don't help here."
- **Preregistration:** `RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42)` added to the inner-CV grid with `max_depth ∈ {None, 5}`, `min_samples_leaf ∈ {1, 3}`, `max_features ∈ {'sqrt', 0.3}`.
- **Why:** Postmortem outcomes may carry non-linear interactions (e.g., age × mobility × pathology-domain burden) that linear models can't capture. RF handles these without the convergence fragility of classical gradient boosting at n≈120 training. User-directed addition. Boosting (classical GB, XGBoost, LightGBM) still excluded to keep grid defensible; HistGB added under entry #11 as the sole boosting family.

### 11. Grid widened and HistGradientBoosting added

- **Preregistration:**
  - LR-l2 and LR-l1: `C ∈ logspace(-3, 2, 8)` (was 5 points).
  - SVC-RBF: `γ ∈ {scale, 0.003, 0.01, 0.03, 0.1}` (was 3 values).
  - `HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=42)` added with `max_depth ∈ {3, 5}`, `learning_rate ∈ {0.05, 0.1}` — 4 configs.
- **Why:** User-directed widening of the hyperparameter surface (compute cost explicitly OK'd as secondary). HistGB chosen over XGBoost/LightGBM because it's scikit-learn native (no dependency), histogram-based and fast at small n, and well-regularized by default. Classical `GradientBoostingClassifier` excluded because HistGB already probes the boosting question better at this sample size. XGBoost/LightGBM add dependencies without probing a meaningfully different hypothesis.

### 12. Post-hoc feature-selection sensitivity analysis added

- **Preregistration:** after the primary nested-CV run, for each (outcome × rung × cohort) cell, prepend `SelectKBest(f_classif, k=20)` to the mode inner-CV winner, rerun the 5×3 outer StratifiedKFold, report `ΔAP_select` mean ± 95% Nadeau-Bengio CI in `feature_selection_sensitivity.csv`. Non-inferential — not part of any Holm-corrected test.
- **Why:** The primary grid already probes sparsity implicitly (LR-l1, elasticnet, RF `max_features`), and cross-sectional evidence favored No Selection at this n. Embedding SelectKBest as a grid-level ON/OFF flag would double compute and invite selective-reporting reads of the primary table. A post-hoc sensitivity answers the question without polluting the primary inference. User-approved 2026-04-23.

### 13. `selection_bias_table.csv` promoted to preregistered deliverable

- **v1 draft:** treated as reviewer-response artifact.
- **Preregistration:** preregistered deliverable, committed before any fit. Compares analysis cohort (n=187) vs excluded subjects (alive-or-unapproved, n=603) on age_bl, msex, educ, gait_speed, cogn_global, mobility_disability_binary. Standardized mean differences + chi-sq/t-test p-values.
- **Why:** Council (Gemini, MED) flagged selection bias specific to died+consented-to-autopsy cohorts. Committing the quantitative comparison as a preregistered artifact — not a defensive addendum — signals the analysis owned the bias question from the start.

---

## Post-freeze changes

### 2026-04-23 — Pre-fit data-integrity corrections (caught in `longitudinal/c_common.py` smoke test, BEFORE any model fit)

Three column-level corrections to the gait-bout feature set. All discovered by the common-module smoke test that enumerates rungs and checks for leakage. No inferential code had run at this point.

1. **`gait_speed` explicitly excluded from the gait-bout feature set.** The preregistration implies this (gait_speed is the +8ft comparator, lives in its own rung), but the classifier in `core/data.py::_classify_columns` does not exclude it by default. Without this fix, `gait_speed` would have appeared in both the `+8ft` rung and the `+Gait Bout` / `+Gait Bout + 8ft` rungs, invalidating the Contrast-2 `+Gait Bout vs +8ft` comparison. Corrected; not a change to the inferential plan, just a textual gap in the preregistration.

2. **Neuropathology columns excluded to prevent circular prediction:** `ci_num2_gct`, `ci_num2_mct` (chronic infarct gross/macroscopic cortical), `hip_scl_mid` (hippocampal sclerosis, midline). These are themselves pathology measurements in the postmortem CSV and would be circular as predictors of pathology outcomes. The preregistration's circularity check explicitly covered `gait_speed` but did not enumerate these three. Excluding them matches the spirit of the check (non-circular predictors only). 440 gait-bout features remain after this filter.

3. **Preregistered feature count "~195" was stale.** The actual gait-bout feature count after excluding the 3 pathology cols in #2 is 440 on the postmortem CSV (442 on the cross-sectional ABL CSV under the same classifier). The "~195" estimate in the preregistration was inherited from an older version of the cross-sectional codebase. Factual correction, not a methodological change — the preregistered pipeline (BlockPCA per GAIT_DOMAIN with `var_kept` ∈ {0.80, 0.90}) collapses these 440 raw features into ~30–50 principal components before any estimator sees them.

**Compute-estimate correction:** the preregistration section "## Compute" computed ~56k fits as `104 candidates × 3 inner × 15 outer × 6 outcomes × 2 cohorts = 56,160`. This undercounts cells. Cohort-D runs 2 rungs (Demographics, +Gait Bout) per outcome = 12 cells; Cohort-8 runs 4 rungs (Dem, +8ft, +Gait Bout, +Gait Bout+8ft) per outcome = 24 cells. Correct total: 36 cells × 15 outer × 3 inner × 104 candidates ≈ 168k fits. No inferential change; just a compute-estimate revision. Expected wall clock: ~10–20 hr serial, ~2–5 hr with `n_jobs=-1` on 6–8 cores.
