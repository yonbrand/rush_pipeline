# Analysis C — Summary

Supplementary longitudinal analysis. Not a headline claim.
Run completed 2026-04-23, wall-clock 56 min on 40-core vast.ai CPU.

## Question

Do wrist-sensor daily-living gait-bout features explain variance in
postmortem neuropathology beyond (a) demographics alone, and (b) the
8-ft walk test, across 6 pathology outcomes spanning 4 neuropath
domains?

## Design (preregistered)

- **Cohort-D** (n=187, all approved autopsy): + Gait Bout vs Demographics.
- **Cohort-8** (n=146, approved + gait_speed non-null): + Gait Bout vs + 8ft.
- **6 outcomes**: AD composite (`ad_adnc`), Lewy 7-region (`lb_7reg`),
  TDP-43 ≥1 (`tdp_st4_bin`), arteriolosclerosis ≥2 (`arteriol_scler_bin`),
  CAA ≥2 (`cvda_4gp2_bin`), Lewy HENL ≥1 (`henl_4gp_bin`).
- **Pipeline**: SimpleImputer(median) → StandardScaler → BlockPCA
  (per-domain, variance ∈ {0.80, 0.90}) → model.
- **Models**: LR-l2, LR-l1, LR-elasticnet, SVC-RBF, Random Forest,
  HistGradientBoosting. ~104 candidates per cell. Nested CV:
  outer RepeatedStratifiedKFold 5×3 (seed=42), inner StratifiedKFold 3.
- **Statistical test**: Nadeau & Bengio (2003) corrected repeated-kfold
  paired t-test on per-fold ΔAP (`core.cv.corrected_repeated_cv_test`).
- **Multiplicity**: Holm-Bonferroni step-down, m=6 outcomes within each
  contrast family.
- **Primary metric**: AP (average precision). AUC reported alongside.
- **Forced covariates** (all rungs): age_bl, msex, educ, age_death, pmi,
  time_lastce2dod.

## Pre-fit diagnostics (see `prefit_diagnostics.json`)

- All 12 (outcome × cohort) cells pass n≥100 and minority≥25.
- gait_speed missingness in approved cohort: 21.9% (under 30% cap).
- All 6 forced covariates have 187/187 non-null coverage.
- Selection bias quantified in `selection_bias_table.csv` (see
  `limitations.md` §1).

## Results

### Primary contrast 1 — Cohort-D: + Gait Bout vs Demographics

Holm×6 within family. **0 of 6 outcomes reject.**

| outcome | AP Gait | AP Demo | ΔAP [95% CI] | p_raw | p_holm | reject |
|---|---:|---:|---|---:|---:|:---:|
| ad_adnc            | 0.706 | 0.752 | −0.047 [−0.157, +0.064] | 0.380 | 1.000 | no |
| lb_7reg            | 0.317 | 0.345 | −0.028 [−0.195, +0.140] | 0.728 | 1.000 | no |
| tdp_st4_bin        | 0.685 | 0.714 | −0.029 [−0.159, +0.100] | 0.637 | 1.000 | no |
| arteriol_scler_bin | 0.526 | 0.548 | −0.023 [−0.184, +0.138] | 0.767 | 1.000 | no |
| cvda_4gp2_bin      | 0.383 | 0.479 | −0.096 [−0.250, +0.059] | 0.204 | 1.000 | no |
| henl_4gp_bin       | 0.368 | 0.411 | −0.043 [−0.138, +0.051] | 0.345 | 1.000 | no |

All point estimates are negative (comparator weakly preferred); all CIs cross zero.
See `figures/post/05_forest_cohort_d.png`.

### Primary contrast 2 — Cohort-8: + Gait Bout vs + 8ft

Holm×6 within family. **0 of 6 outcomes reject.**

| outcome | AP Gait | AP 8ft | ΔAP [95% CI] | p_raw | p_holm | reject |
|---|---:|---:|---|---:|---:|:---:|
| ad_adnc            | 0.766 | 0.777 | −0.011 [−0.091, +0.069] | 0.773 | 1.000 | no |
| lb_7reg            | 0.359 | 0.354 | +0.005 [−0.224, +0.235] | 0.961 | 1.000 | no |
| tdp_st4_bin        | 0.666 | 0.705 | −0.039 [−0.207, +0.128] | 0.622 | 1.000 | no |
| arteriol_scler_bin | 0.507 | 0.540 | −0.033 [−0.216, +0.150] | 0.707 | 1.000 | no |
| cvda_4gp2_bin      | 0.428 | 0.427 | +0.000 [−0.175, +0.176] | 0.997 | 1.000 | no |
| henl_4gp_bin       | 0.380 | 0.420 | −0.039 [−0.207, +0.129] | 0.624 | 1.000 | no |

Point estimates near zero (max |Δ| = 0.039); all CIs cross zero.
See `figures/post/05_forest_cohort_8.png`.

### Sensitivity contrasts (no Holm; reported without family correction)

- Cohort-8: + Gait Bout + 8ft vs + 8ft — does adding the 8-ft walk to
  wrist features improve the combined model?
- Cohort-8: + Gait Bout vs Demographics — does wrist beat demographics
  on the smaller Cohort-8 (where 8-ft is available)?

See `contrasts.csv` rows with `family == "sensitivity"`.

### Post-hoc feature-selection sensitivity

Re-runs the + Gait Bout rungs with SelectKBest in place of Block PCA, to
check whether the PCA backbone is the source of any observed signal.

All 12 SKB−PCA ΔAP CIs cross zero (|Δ| ≤ 0.084; raw p ∈ [0.30, 0.98]).
The null result is not a PCA artefact — the alternate feature-selection
backbone does not reveal hidden signal.

See `figures/post/08_skb_sensitivity.png`.

### Sensitivity contrasts (Cohort-8; no Holm)

- **Combined (+ Gait Bout + 8ft) vs + 8ft alone**: all 6 ΔAP ∈ [−0.047, −0.005], all raw p ≥ 0.57. Adding wrist features to 8ft does not help.
- **+ Gait Bout vs Demographics (Cohort-8)**: all 6 ΔAP ∈ [−0.069, +0.040], all raw p ≥ 0.37.

## Headline sentence (for paper text)

In the autopsy subcohort (n=187 / n=146 with 8ft), a wrist-sensor
daily-living gait-bout feature set did **not** improve postmortem
neuropathology prediction beyond demographics alone or demographics
plus the 8-ft walk test; 0 of 6 outcomes rejected Holm×6 in either
primary family (max |ΔAP| = 0.096 in Cohort-D, 0.039 in Cohort-8; all
95% CIs crossed zero; all Holm-corrected p-values = 1.00). The result
is robust to replacing Block PCA with SelectKBest and is consistent
with this subcohort being the older, slower, more cognitively impaired
tail of the source population (selection bias detailed in
`limitations.md` §1), where demographics already explain most
pathology-relevant variance.

## Files

| path | content |
|---|---|
| `preregistration.md` | frozen design (2026-04-23) |
| `deviations.md` | logged deviations |
| `limitations.md` | selection bias, power, circularity |
| `selection_bias_table.csv` | n=187 vs n=603 on 6 variables |
| `prefit_diagnostics.json` | per-cell counts, tripwire status |
| `per_fold_scores.csv` | per-(cell, model) AP/AUC + timings |
| `fold_scores.npz` | per-(cell, model) fold arrays |
| `descriptive.csv` | winner model + AP per (cohort, rung, outcome) |
| `contrasts.csv` | Nadeau-Bengio results + Holm (primary) |
| `feature_selection_sensitivity.csv` | SelectKBest re-run |
| `summary.json` | headline + metadata |
| `figures/eda/01_class_balance.png` | pre-fit |
| `figures/eda/02_covariates.png` | pre-fit |
| `figures/eda/03_selection_bias.png` | pre-fit |
| `figures/eda/04_missingness.png` | pre-fit |
| `figures/post/05_forest_cohort_d.png` | primary contrast 1 forest |
| `figures/post/05_forest_cohort_8.png` | primary contrast 2 forest |
| `figures/post/06_ap_per_rung.png` | absolute AP per rung |
| `figures/post/07_winner_heatmap.png` | winner model per cell |
| `figures/post/08_skb_sensitivity.png` | SKB sensitivity forest |
