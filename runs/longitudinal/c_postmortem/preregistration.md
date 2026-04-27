# Analysis C — Postmortem neuropathology — Preregistration

**Status:** FROZEN 2026-04-23. User-approved after `/council plan` round 1 and two rounds of user-directed revision (nested-CV-only cohort posture; expanded model + hyperparameter grid + post-hoc feature-selection sensitivity). Any change after this point must be logged in `deviations.md` with date and rationale.

## Changes vs v1 (what the council pushed back on + user direction)

1. **Added `deviations.md` as a deliverable** (logs the 2→6 outcome-family expansion before any fit).
2. **Dropped the ordinal model branch.** All 4 ordinal outcomes binarized at preregistered cutpoints. Uniform `LogisticRegression` across all 6 outcomes.
3. **`time_lastce2dod` now a forced covariate** in every rung (was limitations.md only).
4. **`selection_bias_table.csv` is a preregistered deliverable** (not a reviewer-response artifact).
5. **Nadeau–Bengio corrected t replaces paired bootstrap** on 15 fold deltas (matches cross-sectional + A + B v2 convention).
6. **Added `(+Gait Bout) vs Demographics` as a co-primary** alongside `(+Gait Bout) vs (+8ft)`. Two parallel Holm×6 families.
7. **Honest sample-size correction.** Approved n=187 (not 217 as v1 claimed — I confused `npath_approved non-null` with `npath_approved==1`).
8. **No lockbox split within Analysis C.** n=29 is too small to give useful held-out estimates; holding it out just forfeits 18% of the sample. Match Analysis A's posture (nested CV only, no lockbox). Full n=187 approved cohort enters the analysis.
9. **Model + hyperparameter search inside inner CV.** Replaced the deterministic LR-l2 pipeline with an inner-CV grid over LR-l2, LR-l1, LR-elasticnet, SVC-RBF, plus PCA `var_kept` ∈ {0.80, 0.90}. Model selection happens strictly inside the inner fold per outer fold — one prediction vector per outer fold, no inferential DoF added. Rationale: postmortem outcomes are a different domain than the cognition/motor outcomes the LR-l2 prior was tuned on.
10. **RandomForest added to the inner-CV grid.** Explicit departure from the cross-sectional "no tree ensembles" posture. Rationale: postmortem outcomes may carry non-linear interactions (e.g., age × mobility × pathology) that linear models can't capture; RF handles these without the convergence fragility of GB at n≈120. User-approved 2026-04-23.
11. **Grid widened and HistGradientBoosting added.** LR-l2 and LR-l1 `C` grids expanded to `logspace(-3, 2, 8)`; SVC `γ` grid widened to 5 values. `HistGradientBoostingClassifier` (4 configs) added as the sole boosting family — native scikit-learn, histogram-based, regularized, fast at small n. XGBoost/LightGBM/classical GB still excluded. Rationale: probe the linear/kernel hyperparameter surface more densely and cover one well-regularized boosting family; user explicitly OK'd compute cost as secondary. User-approved 2026-04-23.
12. **Post-hoc feature-selection sensitivity analysis added.** `SelectKBest(f_classif, k=20)` prepended to the inner-CV winner per cell, rerun on outer CV, reported as supplementary `feature_selection_sensitivity.csv`. Non-inferential, not part of any Holm-corrected test. Rationale: probe whether explicit prefilter matters without inflating primary grid or inviting selective-reporting reads. User-approved 2026-04-23.

---

## Question

Do wrist-sensor daily-living gait-bout features, measured at baseline, explain variance in postmortem neuropathology beyond (a) demographics alone and (b) the 8-ft walk test, across multiple pathology domains (AD, Lewy body, TDP-43, vascular)?

## Data

- Source: `outputs/tables/merged_gait_clinical_postmortem.csv` (790 rows; 187 with `npath_approved==1`).
- **Analysis cohort:** n=187 approved subjects. **No lockbox split** (see Change #8). Matches Analysis A's posture.
- First sensor visit per subject (one row per projid; StratifiedKFold applies — no GroupKFold machinery needed since no within-subject clustering).
- 8ft comparator: `gait_speed` from ABL (146/187 non-null = 78% coverage).

## Forced covariates (every rung)

- `age_bl` (baseline age, 187/187)
- `msex` (sex, 187/187)
- `educ` (education, 187/187)
- `age_death` (187/187)
- `pmi` (post-mortem interval, 187/187 — confirmed, no missingness, no sensitivity needed)
- **`time_lastce2dod`** (time from last clinical evaluation to death, 187/187)

## Outcome family (6 — all binary, preregistered cutpoints)

Exact counts from the n=187 approved cohort:

| Outcome | Raw type | Cutpoint | Cohort-D pos / neg (n) | Cohort-8 pos / neg (n) | Domain |
|---|---|---|---|---|---|
| `ad_adnc` | binary | native 0/1 | 120 / 67 (187) | 95 / 51 (146) | Alzheimer's disease neuropath |
| `lb_7reg` | binary | native 0/1 | 40 / 147 (187) | 30 / 116 (146) | Lewy body (7-region) |
| `tdp_st4_bin` | ordinal 0-3 | **≥1 vs 0** | 101 / 86 (187) | 82 / 64 (146) | TDP-43 staging |
| `arteriol_scler_bin` | ordinal 0-3 | **≥2 vs <2** (moderate-severe, MAP/ROS convention) | 67 / 119 (186) | 51 / 94 (145) | Vascular — arteriolosclerosis |
| `cvda_4gp2_bin` | ordinal 0-3 | **≥2 vs <2** (moderate-severe CAA, MAP/ROS convention) | 48 / 139 (187) | 33 / 113 (146) | Vascular — cerebrovascular/amyloid angiopathy |
| `henl_4gp_bin` | ordinal 0-3 | **≥1 vs 0** (any-vs-none; grade ≥2 has n=10, too sparse for higher cut) | 48 / 138 (186) | 38 / 107 (145) | Lewy body (HENL regions) |

All outcomes clear the minority-class tripwire (≥25). All outcomes clear the dev-n tripwire (≥100). Final counts frozen in `prefit_diagnostics.json` before any fit.

Dropped (pre-checked, no power): `path_pd_modsev` (8 events), `tangsqrt` (n=72), `amylsqrt` (n=20).

## Rungs (4, nested)

1. **Demographics** — age_bl, msex, educ, age_death, pmi, time_lastce2dod
2. **+ 8ft** — Demographics + `gait_speed`
3. **+ Gait Bout** — Demographics + wrist gait-bout features (~195 cols)
4. **+ Gait Bout + 8ft** — Demographics + Gait Bout + `gait_speed`

## Pipeline (uniform skeleton; model + hyperparameters searched inside inner CV)

Preprocessing skeleton fixed across all configs:
`SimpleImputer(median) → StandardScaler → BlockPCA(per GAIT_DOMAIN, demographics passed through) → CorrelationFilter(|r|>0.95) → {estimator}`

Feature-selection strategy: No Selection (inherited winner from cross-sectional + longitudinal A/B — this is feature selection, orthogonal to model selection below).

**Model + hyperparameter grid (searched inside inner 3-fold GridSearchCV, scoring=AP):**

| Estimator | Hyperparameters searched |
|---|---|
| LogisticRegression (penalty=l2) | C ∈ logspace(-3, 2, 8) |
| LogisticRegression (penalty=l1, solver=liblinear) | C ∈ logspace(-3, 2, 8) |
| LogisticRegression (penalty=elasticnet, solver=saga, max_iter=5000) | C ∈ {0.1, 1, 10}; l1_ratio ∈ {0.25, 0.5, 0.75} |
| SVC (kernel=rbf, probability=True) | C ∈ {0.1, 1, 10}; γ ∈ {scale, 0.003, 0.01, 0.03, 0.1} |
| RandomForestClassifier (n_estimators=300, class_weight='balanced_subsample', random_state=42) | max_depth ∈ {None, 5}; min_samples_leaf ∈ {1, 3}; max_features ∈ {'sqrt', 0.3} |
| HistGradientBoostingClassifier (max_iter=300, early_stopping=False, random_state=42) | max_depth ∈ {3, 5}; learning_rate ∈ {0.05, 0.1} |

Additional preprocessing hyperparameter in the same grid:
- BlockPCA `var_kept` ∈ {0.80, 0.90}

Total candidates per inner CV: ~104 (8 LR-l2 + 8 LR-l1 + 9 elasticnet + 15 SVC + 8 RF + 4 HistGB, × 2 PCA settings). Inner 3-fold GridSearchCV selects a single winner per outer fold; the outer fold sees **one prediction vector** (the inner-CV winner). Nadeau-Bengio + Holm apply unchanged — no added degrees of freedom at the inference layer.

**Outer CV:** StratifiedKFold 5×3 repeats (seed=42) on the full n=187 approved cohort. No groups (one row per subject). Stratification: binary outcome label per fold.

**Per-outcome winner reporting:** `summary.json` logs the mode + frequency of the inner-CV winner (estimator + key hyperparameters) across the 15 outer folds per (outcome × rung × cohort). If winners are unstable across folds, that's reported honestly — no post-hoc "best model" cherry-picking.

**Rationale for expanding beyond LR-l2:** Cross-sectional + longitudinal A/B inherited-winner posture was chosen for cognition/motor outcomes. Postmortem neuropathology outcomes differ (different signal structure, different class balance, different domain). Locking in LR-l2 a priori is an under-powered prior; folding model selection into the inner CV is the statistically clean way to let the data speak without inflating analytic flexibility.

## Cohort-per-contrast rules

To handle `gait_speed` 22% missingness without unfair imputation:

- **Cohort-D (for vs-Demographics contrast):** n=187 (186 for `arteriol_scler` and `henl_4gp`), all approved subjects. `gait_speed` not used in this cohort.
- **Cohort-8 (for vs-8ft contrast):** n=146 (145 for the two outcomes above), approved subjects with non-null `gait_speed`. Used for all 4 rungs in this contrast (including Demographics and +Gait Bout), so contrasts are paired on identical subjects.

The `(+Gait Bout) vs Demographics` and `(+Gait Bout) vs (+8ft)` contrasts therefore live in different (but explicitly documented) cohorts. Holm is applied within each contrast's 6-outcome family independently.

## Primary tests (two co-primary contrasts)

**Per outcome, both contrasts:**
- **Contrast 1:** `AP(+Gait Bout on Cohort-D)` vs `AP(Demographics on Cohort-D)`
- **Contrast 2:** `AP(+Gait Bout on Cohort-8)` vs `AP(+8ft on Cohort-8)`

**Inference:** Nadeau–Bengio corrected repeated-CV t-test (same estimator as `core.cv.corrected_repeated_kfold_cv_test` used in cross-sectional + longitudinal A/B), one-sided (gait bout > comparator), α=0.05.

**Multiplicity:** **Holm×6** within each contrast family independently (two parallel Holm corrections, one per contrast). Raw p-values also reported per outcome. Rationale: Contrast 1 and Contrast 2 test different incremental-value hypotheses on different cohorts; a single Holm×12 would over-penalize by mixing unrelated comparisons. Both Holm×6 results reported per outcome with no re-framing based on which rejects.

## Secondary / descriptive

- `(+Gait Bout + 8ft) vs (+8ft)` on Cohort-8 — does adding sensor features on top of 8ft help?
- Absolute AP per rung (no comparison) — for the manuscript supplementary table.
- All metrics: AP (primary), AUC, Brier, balanced accuracy (descriptive only).

## Feature-selection sensitivity analysis (post-hoc, non-inferential)

For each (outcome × rung × cohort) cell, after the primary nested-CV run:

1. Take the mode inner-CV winner (estimator + hyperparameters) from the primary grid.
2. Prepend `SelectKBest(f_classif, k=20)` to that pipeline (fit inside each outer training fold — no leakage).
3. Rerun the same 5×3 outer StratifiedKFold and record AP per fold.
4. Compare to the primary-grid AP: report `ΔAP_select = AP(with SelectKBest) − AP(primary)` per fold, mean ± 95% Nadeau-Bengio CI.

**Purpose:** probe whether an explicit prefilter on top of the inner-CV winner shifts performance meaningfully. Reported as supplementary table only; **not** part of any primary or Holm-corrected test. Deliverable: `feature_selection_sensitivity.csv` (cell-level) and narrative sub-section in `limitations.md`.

Rationale for keeping this post-hoc: cross-sectional evidence favored No Selection at this n; embedding SelectKBest as a grid-level flag would double compute and invite reviewers to read selective reporting in the primary table. A non-inferential sensitivity is the defensible compromise.

## Preregistered deliverables (committed BEFORE any fit)

Under `runs/longitudinal/c_postmortem/`:

1. **`preregistration.md`** — this document, once frozen (remove DRAFT marker, lock heading, commit).
2. **`deviations.md`** — dated 2026-04-23, entries:
   - Original brief spec'd 2 outcomes (`ad_adnc`, `henl_4gp`). Data inspection 2026-04-23 found 4 additional outcomes with approved n ≥ 186 across 3 new pathology domains. Expansion decided before any fit.
   - Brief said "no Holm". v2 plan introduces Holm×6 within each contrast as belt-and-suspenders; raw also reported. Rationale for Holm: defensibility when reporting 6 outcomes.
   - v1 draft cited n=217; correct approved n=187. Corrected here.
   - v1 draft proposed holding out a 29-subject lockbox. Deviation: Analysis C uses nested CV on the full n=187 with no lockbox split. Matches Analysis A posture. Rationale: at n=29 a held-out slice gives no honest error estimate, just forfeits 18% of the cohort. User-approved deviation 2026-04-23.
   - v1 draft locked the estimator to LogisticRegression(l2) only (inherited cross-sectional winner). v2 expands the inner-CV grid to include LR-l1, LR-elasticnet, and SVC-RBF, plus PCA var_kept ∈ {0.80, 0.90}. Rationale: postmortem neuropathology is a different domain from cognition/motor; inherited winner may not transfer. Model selection occurs strictly inside the inner CV per outer fold — no added inferential DoF. User-approved deviation 2026-04-23.
3. **`prefit_diagnostics.json`** — per-outcome dev n, pos/neg after binarization, covariate non-null rates, gait_speed coverage in Cohort-8.
4. **`selection_bias_table.csv`** — comparison of analysis cohort (n=187) vs excluded subjects (alive-or-unapproved, n=603) on: age_bl, msex, educ, gait_speed, cogn_global, mobility_disability_binary. Standardized mean differences + p-values (chi-sq for categorical, t-test for continuous). **Committed as a deliverable, not a reviewer-response artifact.**
5. **`per_fold_scores.csv`** — all 15 outer-fold AP per (outcome × rung × cohort).
6. **`descriptive.csv`** — mean ± std per rung per outcome, Nadeau-Bengio 95% CI on Δ.
7. **`summary.json`** — per-outcome (Δ_mean, Δ_CI, p_raw, p_holm_C1, p_holm_C2, reject flags).
8. **`limitations.md`** — selection bias quantitative summary (pulls from #4), time-to-death distribution, PMI missingness sensitivity, circularity check (none of the 6 outcomes derive from gait_speed — clean), feature-selection sensitivity narrative (pulls from #9).
9. **`feature_selection_sensitivity.csv`** — post-hoc `SelectKBest(f_classif, k=20)` sensitivity per (outcome × rung × cohort); `ΔAP_select` mean ± 95% CI vs primary-grid AP. Non-inferential, supplementary table only.

## PMI missingness sensitivity (Fix #4 addendum)

187/187 approved subjects have `pmi` — no missingness. Sensitivity analysis NOT required.

## Tripwires (frozen, preregistered)

1. n with non-null outcome < 100 → drop outcome from family, report null.
2. Pos or neg class < 25 after binarization → drop outcome from family (all 6 currently clear; minority = 40 for lb_7reg in Cohort-D).
3. Logistic non-convergence > 30% of folds → report null for that cell.
4. Gait_speed missingness > 30% → trigger Cohort-8 sensitivity analysis (currently at 22%, under the tripwire).

## Lockbox

No lockbox split within Analysis C. See Change #8 and deviations.md. The cross-sectional lockbox was never "reserved for Analysis C" — it was held out for cross-sectional co-primaries (already evaluated, both null). Using those 29 subjects in Analysis C's nested CV is orthogonal to the cross-sectional lockbox's purpose and matches Analysis A's posture.

## Compute

Model search: ~104 candidates × 3 inner folds × 15 outer folds × 6 outcomes × 2 cohorts ≈ 56k fits. Linear/SVM/HistGB fits are sub- to few-seconds at n_train ≈ 120. RF (300 trees) and saga elasticnet dominate wall time. Estimated total: **10–16 hours** serial, **2–4 hours** with `n_jobs=-1` on outer folds (6–8 cores). Run overnight.

## Not in scope

- No XGBoost / LightGBM / classical `GradientBoostingClassifier`. `HistGradientBoostingClassifier` is the only boosting family included — it's scikit-learn native, histogram-based (fast at small n), and regularized. XGBoost/LightGBM would add dependencies without probing a meaningfully different hypothesis.
- No stacking / blending meta-models (DoF bomb at n≈120 training).
- No MLPs / KNN / Naive Bayes (no evidence they'd help at this n and dimension).
- No foundation-model features.
- No ordinal models (dropped per council fix #2).
- No headline paper claim — strictly supplementary.
- No lockbox touch.

## What the paper gets

Supplementary "Postmortem neuropathology" panel: one forest plot per contrast (vs Demographics, vs +8ft), 6 outcomes each, showing ΔAP ± 95% CI with Holm-corrected reject markers. Absolute AP table in supplement. Limitations sub-section pulling from `limitations.md` — honest about selection bias (quantified in table), honest about power (MDE on ΔAP ~ 0.10 at n=158).

---

## Post-freeze workflow

1. User eyeballs this draft, greenlights or requests revisions.
2. Rename to `preregistration.md`, commit, mark FROZEN.
3. Write `deviations.md` with today's date BEFORE any code runs.
4. Implement driver (`longitudinal/c_postmortem.py`) + `selection_bias.py` helper.
5. Run `/council review` on the driver diff before any fit.
6. Council APPROVED → run the driver. Council FIX → fix, round 2.
7. Write summary.md after run.
