# Analysis C — Limitations

Frozen on 2026-04-23, written from preregistration + pre-fit diagnostics
alone. Any post-run nuance is added to `summary.md`, not here.

## 1. Selection bias (died + consented to autopsy)

The analysis cohort is **n=187** (approved autopsy), drawn from a source
population of n=790 with at least one wrist-sensor visit. The excluded
603 are either (a) alive, (b) deceased without autopsy consent, or
(c) autopsy records not yet approved. `selection_bias_table.csv`
quantifies the bias on 6 pre-specified variables. Key values:

| variable | n_analysis | n_excluded | SMD (analysis − excluded) |
|----------|-----------:|-----------:|--------------------------:|
| age_bl                        | 187 | 600 | **+0.57** (older) |
| gait_speed                    | 146 | 391 | **−0.65** (slower) |
| cogn_global                   | 187 | 570 | **−0.77** (lower cognition) |
| mobility_disability_binary    | 185 | 600 | **+0.83** (more disability) |
| educ                          | 187 | 600 | +0.11 |
| msex                          | 187 | 600 | −0.06 |

These are large effects (|SMD| > 0.5) on three of six variables. The
analysis cohort is older, slower, more cognitively impaired, and more
mobility-disabled than the excluded population. **Generalization is
limited to the subgroup most likely to die with approved autopsy** —
which is, by design, the tail most at risk of neuropathology. Effect
sizes reported here are representative of that tail, not of the full
MAP/ROS cohort.

## 2. Time between last sensor visit and death

Covariate `time_lastce2dod` is forced into every rung to partially
account for the decay interval between gait measurement and
autopsy-observed pathology. Residual confounding from time-to-death
remains; subjects who died sooner after their sensor visit have a
shorter accumulation window, and pathology could continue to
progress during that interval. Not corrected for via IPCW or
survival weighting — single-timepoint outcome is a snapshot at death.

## 3. Power and multiplicity

- **n per cell**: 145–187 (minority class 30–120).
- **Minimum detectable effect** (ΔAP, at alpha=.05 two-sided, power=.80)
  for n≈150: roughly 0.08 AP. Effects smaller than that will read as
  non-significant even if real.
- **Multiplicity posture**: Holm-Bonferroni correction within each of
  two primary contrast families (m=6 outcomes each). Raw p-values
  reported alongside. Two families correspond to two distinct primary
  questions (Gait Bout beats Demographics? beats 8ft?) and are not
  pooled under a single Holm.

## 4. Cross-sectional measurement; no time-varying sensor signal

First-visit sensor summary (as in cross-sectional + Analysis A/B). Does
not use longitudinal trajectory information. Any within-subject
time-varying signal that might predict proximity to pathology-driven
death is not captured.

## 5. Model-grid "winner" selection on outer-CV scores

For each (outcome × cohort × rung) cell, 6 models × Block-PCA variance
grid are evaluated via nested CV. The paired Nadeau-Bengio test uses
the best-AP model's per-fold array. Selecting the winner on outer-CV
AP and then testing on the same per-fold scores is a mild double-dip;
it is the convention in the project's cross-sectional and A/B
analyses. We mitigate by: (a) reporting Holm-corrected p-values, and
(b) publishing a post-hoc feature-selection sensitivity
(`feature_selection_sensitivity.csv`) that re-runs the +Gait Bout cells
with a disjoint strategy (SelectKBest instead of Block PCA) to check
whether the backbone drives signal.

## 6. Gait-bout feature set is wrist-derived daily-living, not gait-lab

The +Gait Bout rung comprises 440 features derived from wrist-worn
accelerometer daily-living bouts — cadence, regularity, speed
estimates, temporal patterns, day-to-day stability. These are
fundamentally different from the single 8-ft walk test (`gait_speed`),
which is a brief supervised assessment. The +8ft comparator uses a
single scalar; Block PCA reduces the 440 sensor features to ~30–50
orthogonal components before modeling. Comparisons are between
**different measurement modalities**, not within-modality refinements.

## 7. What we did NOT do

- No tree ensembles beyond Random Forest + HistGB.
- No foundation-model wrist features.
- No lockbox touch. Dev-only analysis. Per brief §C (supplementary),
  results are published regardless of direction.
- No ordinal cumulative-logit models. All 4 native-ordinal outcomes
  were binarized at preregistered cutpoints (see `preregistration.md`
  §Outcomes) for consistency with the cross-sectional and A/B logistic
  backbone and to maintain convergence with n≈190 and minority
  classes as small as 30.
- No external replication cohort. MAP/ROS only.

## 8. Circularity checklist (passes)

- None of the 6 outcome columns (ad_adnc, lb_7reg, tdp_st4,
  arteriol_scler, cvda_4gp2, henl_4gp) are derived from `gait_speed`
  or any wrist-sensor feature. `gait_speed` (8-ft) is a clinical
  measurement, independent of sensor.
- Other pathology scales (caa_4gp, lb_any, ci_num2_gct, ci_num2_mct,
  hip_scl_mid) are excluded from the feature set — see
  `longitudinal/c_common.py::_EXCLUDE_FROM_FEATURES`.
- Clinical composites (cogn_global, mobility_disability_binary,
  rosbsum, etc.) that might partially derive from or correlate
  mechanically with pathology are excluded from the feature set.
