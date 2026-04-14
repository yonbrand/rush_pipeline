# Autoresearch Brief — Daily-Living Wrist Gait vs PA vs Clinic Gait Speed

**Audience:** Claude Code, running as a background autoresearch agent.
**Owner:** (you)
**Mode:** Iterative, long-running, self-directed. Karpathy-style autoresearch: propose → run → log → decide → repeat, until a stop criterion is hit.

---

## 1. Scientific mission (read this first, re-read before every experiment)

We want to show — rigorously and reproducibly — that **daily-living gait quality metrics extracted from a ~10-day wrist accelerometer recording carry clinical information that (a) the traditional 8-foot gait speed test does not, and (b) cannot be recovered from broad physical-activity summaries alone.**

The deliverable is a Nature-style short paper. The agent's job is to maximize the *scientifically defensible* added value of the wrist gait feature set while maintaining methodological standards that would survive peer review at a top journal.

**"Better results" in this project means exactly one thing: a larger, more statistically credible gap between the gait-bout feature set and the two comparators (8-ft speed, broad PA), on outcomes that are not circular with the comparator.** Anything else — squeezing AUC on any single outcome, tuning until a p-value crosses 0.05 — is failure.

---

## 2. Inputs you already have

- `merged_gait_clinical_abl.csv` — 790 subjects × 681 columns, one row per subject at analytic baseline. Contains gait bout features, daily PA features, sleep / rest-activity features, demographics, 8-ft gait speed, and 7 outcomes.
- `prediction_pipeline.py` — current nested-CV pipeline. Treat this as the **baseline**, not as sacred. You may refactor it freely.
- `MODELING_OVERVIEW.md`, `PIPELINE_OVERVIEW.md` — context on features and methodology. Read both before your first experiment.

Outcomes available:
- **Binary:** `parkinsonism_yn`, `mobility_disability_binary`, `falls_binary`, `cognitive_impairment`
- **Continuous:** `parksc`, `motor10`, `cogn_global`

### Circularity warning — critical
`parksc`, `motor10`, and `parkinsonism_yn` are all derived from the modified UPDRS motor exam, which includes the 8-ft walk test as a scoring component. Any comparison of "gait bout features vs 8-ft speed" on these outcomes is contaminated and **cannot be a primary or secondary finding**. Report them as exploratory / sensitivity analyses with this caveat in the text. The two co-primary outcomes (`cogn_global`, `mobility_disability_binary`) and the two confirmatory secondary outcomes (`falls_binary`, `cognitive_impairment`) are all free of this concern.

---

## 3. Pre-registered primary analysis (DO NOT CHANGE AFTER SEEING RESULTS)

Before your first experiment, write the following to `preregistration.md` and commit it. Never edit it.

- **Co-primary outcomes:**
  1. `cogn_global` (continuous, regression, R²) — cognitive axis. Not circular with the 8-ft walk test, fullest n.
  2. `mobility_disability_binary` (binary, classification, AP) — motor / functional axis. Not circular (self-reported difficulty with mobility tasks, not a derived score using the walk test), near-complete coverage (785/790), well-balanced (≈46% positive).
  These two outcomes are conceptually independent (cognitive vs functional) and showing that daily-living gait predicts both is a substantively stronger claim than either alone.
- **Primary comparison:** Gait Bout (+ demographics) vs 8-ft Gait Speed (+ demographics), corrected repeated-k-fold t-test (Nadeau–Bengio) on paired outer-fold scores. Run identically for both co-primary outcomes.
- **Primary claim to be tested:** ΔR² (cogn_global) > 0 *and* ΔAP (mobility_disability_binary) > 0, with Holm–Bonferroni correction across the two co-primary tests at family-wise α = 0.05. The strongest version of the claim requires both to pass; either one passing alone is reported as a weaker but still publishable finding.
- **Incremental value ladder (the conceptual spine of the paper):** report the following nested feature sets, evaluated identically, so the reader sees what each layer adds on top of the previous one:
  1. **Demographics only** — `age_at_visit`, `msex`, `educ`. Sets the floor: how much of each outcome is explained by who the person is, before any movement data.
  2. **Demographics + 8-ft gait speed** — adds the traditional clinic walk test. The current standard of care benchmark.
  3. **Demographics + Daily PA** — adds broad wrist-derived activity volume metrics (no gait quality).
  4. **Demographics + Sleep / RAR** — adds sleep architecture and rest-activity rhythm metrics.
  5. **Demographics + Gait Bout** — adds daily-living gait *quality* features (the paper's hero set).
  6. **Demographics + Gait Bout + Daily PA + Sleep** — full sensor stack.
  7. **Demographics + Gait Bout + 8-ft speed** — does the clinic test add anything *on top of* daily-living gait?
  Demographics are included in every set so all comparisons are apples-to-apples — any difference between sets is attributable to the sensor / clinical features, not to age/sex/education leaking into one set and not another.
- **Secondary (confirmatory) outcomes:** `falls_binary`, `cognitive_impairment` (both binary, AP), with Holm–Bonferroni correction across the two.
- **Exploratory only (explicitly flagged in paper with circularity caveat):** `parksc`, `motor10`, `parkinsonism_yn`. All three use the modified UPDRS motor exam, which includes the 8-ft walk test as a component, so any "gait bout vs 8-ft speed" comparison on these outcomes is contaminated. Report descriptively only; never as evidence for the primary claim.
- **Held-out lockbox:** a stratified 15% of subjects, split with `random_state=20260414`, written to `lockbox_ids.csv` on first run. Stratification is on `mobility_disability_binary` (the co-primary binary outcome with the fullest n and most balanced distribution); after the split, verify by inspection that the lockbox and dev sets are also reasonably balanced on `cogn_global` quartiles and on `cognitive_impairment` — if any subset is grossly imbalanced, log it and proceed (do not re-split to fish for a nicer one). **Never touched during iteration.** Only the final frozen best configuration is evaluated on it, exactly once, at the end.

If during iteration you find yourself wanting to change any of the above — stop and log it as a limitation instead.

---

## 4. Hard methodological rules (non-negotiable)

These override any "improvement" you think of. If an idea violates one of them, don't run it; log it as rejected and move on.

1. **All data-dependent preprocessing happens inside CV folds.** Imputation, scaling, variance filtering, correlation pruning, feature selection, PCA, target encoding — anything that estimates statistics from the data — goes inside the sklearn `Pipeline` so it is re-fit on each training fold. Never fit on the full dataset.
2. **Nested CV stays nested.** Hyperparameters and selection `k` are tuned in the inner loop only. Outer-loop scores are reported untouched.
3. **The outer CV seed is fixed once** (`random_state=42`, matching the current pipeline) so fold scores are paired across feature sets and the Nadeau–Bengio test is valid. Do not re-seed to get nicer numbers.
4. **Lockbox is sacred.** The 15% held-out set is loaded only by `final_evaluation.py`, which runs exactly once after the agent declares itself done. Not used for model selection, not used for early stopping, not used for figure generation during iteration.
5. **No outcome shopping.** The co-primary outcomes are `cogn_global` and `mobility_disability_binary`. You may report others, but the headline claim is fixed to those two with Holm–Bonferroni correction.
6. **No subject dropping for convenience.** You may drop subjects only for pre-specified reasons (missing outcome, insufficient valid wear time if a criterion is defined before seeing results). Document every exclusion with a count and reason in `exclusions.md`.
7. **Feature set definitions are frozen after the first successful baseline run.** Specifically, the membership of "Demographics", "8-ft Gait Speed", "Daily PA", "Sleep / RAR", and "Gait Bout" feature groups is locked. Demographics (`age_at_visit`, `msex`, `educ`) appear in every comparison set as the common baseline. You may *engineer new features within* the Gait Bout set, but a feature cannot migrate between groups to make a comparison look better, and demographics cannot be removed from any set.
8. **Every reported number has an uncertainty.** Mean ± SD across outer folds at minimum; bootstrap or Nadeau–Bengio CIs for headline comparisons.
9. **Multiple testing correction** on all pairwise feature-set comparisons within an outcome (Holm–Bonferroni).
10. **No peeking.** You do not look at lockbox predictions, lockbox labels, or lockbox-derived statistics at any point during iteration. If you need to sanity-check data, use the development set only.

---

## 5. Baseline — run this first, exactly once

Before any improvements:

1. Load `merged_gait_clinical_abl.csv`.
2. Create the stratified lockbox split (15%, seed 20260414, stratified on `cognitive_impairment` as a reasonable proxy). Save `lockbox_ids.csv` and `dev_ids.csv`.
3. Run `prediction_pipeline.py` **on the development set only**, unchanged except for (a) reading the dev subset and (b) writing outputs to `runs/baseline/`.
4. Record the baseline R²(gait bout, cogn_global), ΔR² vs 8-ft speed, and corrected p-value. This is the number to beat.
5. Write `runs/baseline/summary.json` with the full results table.

If the baseline crashes or produces obviously broken numbers, fix the bug and re-run before starting any improvement experiments. Do not stack fixes on top of a broken baseline.

---

## 6. Improvement menu (what you are allowed to try)

This is the idea space. Work through it roughly in order of expected value-per-effort. Every experiment must be logged (section 7). You are not required to try all of these; you are required to stop when the stop criterion (section 8) is hit.

### 6a. Data quality & preprocessing
- Smarter imputation: iterative imputer / KNN imputer vs median, inside the fold. Compare on the primary outcome.
- Wear-time / bout-count quality filter on subjects (pre-specify a minimum, e.g., ≥ N valid days or ≥ N bouts) *before* seeing results. Document in `exclusions.md`.
- Winsorization of extreme gait feature values at the 1st/99th percentile, fit inside the fold.
- Better handling of the `prob_bin_*` / `freq_bin_*` histogram duplication already noted in MODELING_OVERVIEW.

### 6b. Feature engineering within the Gait Bout set
- Ratios and contrasts that encode clinically meaningful asymmetries (e.g., slow-bout speed vs fast-bout speed, short-bout regularity vs long-bout regularity).
- Robust summary statistics: trimmed means, IQRs, MAD, in addition to median/CV.
- Percentile features (e.g., 10th, 25th, 75th, 90th of within-subject bout speed distribution) — these are often more informative than the mean for frailty-type outcomes.
- Day-to-day stability features: within-subject between-day SD of each gait metric.
- Time-of-day modulation: amplitude/phase of diurnal gait speed.
- *Do not* mix in PA or sleep features here. Those belong to their own feature sets.

### 6c. Modeling
- Gradient boosting with proper early stopping on an inner-fold validation split (not the outer test fold).
- Calibrated classifiers (`CalibratedClassifierCV`) for the binary outcomes — often improves AP meaningfully for imbalanced classes.
- Stacking: base learners (LR, RF, XGB) → meta-learner (logistic / ridge). Fit stacking with `cross_val_predict` inside the training fold to avoid leakage.
- Per-outcome hyperparameter grids tuned to the sample size (bigger regularization for smaller-n outcomes).

### 6d. Feature selection
- Add SHAP-based selection as an additional strategy (fit base model in inner fold, rank by mean |SHAP|, select top-k).
- Report which features the best configuration keeps, aggregated across outer folds (selection frequency).

### 6e. Statistical framing that strengthens the headline
- **Incremental / partial contribution:** fit a model with the 8-ft speed feature set, then measure whether adding gait bout features *to that baseline* significantly increases R². This is the "added clinical value" framing reviewers want.
- **Nested likelihood-ratio-style comparison** via nested linear models on the continuous outcomes (for interpretability alongside the ML results).
- **DeLong test** for AUC comparisons on binary outcomes, as a complement to Nadeau–Bengio on AP.
- **Bootstrap CIs** on ΔR² / ΔAP (n ≥ 1000 resamples of outer-fold scores).

### 6f. Robustness / sensitivity analyses (expected by reviewers)
- Subgroup results by age tertile and sex.
- Leave-one-site-out or leave-one-batch-out if a site/batch column exists.
- Re-run with a more conservative missing-rate threshold.
- Re-run excluding subjects with very short recordings.
- Permutation test: shuffle outcome and re-run the full pipeline a handful of times, confirm null R² ~ 0.

### 6g. Explicitly forbidden
- Changing the primary outcome after seeing results.
- Changing the outer CV seed after seeing results.
- Training on the lockbox.
- Using the lockbox for early stopping or model selection.
- Engineering features using the outcome variable outside a CV fold.
- Reporting only the best-of-many runs without multiplicity correction.
- Dropping subjects post-hoc because they are outliers in predictions.
- Using `parksc`, `motor10`, or `parkinsonism_yn` as the headline comparison to 8-ft speed (all three are mUPDRS-derived and contaminated).

---

## 7. Autoresearch loop protocol

Work in iterations. Each iteration is one experiment. Each experiment follows this template exactly and writes one line to `experiments.jsonl`:

```json
{
  "id": "exp_0007",
  "timestamp": "2026-04-14T12:34:56Z",
  "hypothesis": "Per-subject percentile features (p10, p25, p75, p90) of bout speed will outperform median/CV alone on cogn_global because frailty manifests in the slow tail.",
  "change": "Added 4 percentile features per bout metric to the Gait Bout feature set, computed per subject from the raw bout table. No other changes.",
  "parent_run": "exp_0004",
  "primary_metric": {"outcome": "cogn_global", "feature_set": "gait_bout", "r2_mean": 0.182, "r2_sd": 0.041},
  "delta_vs_parent": {"r2_mean": +0.014, "nadeau_bengio_p": 0.11},
  "delta_vs_8ft_speed": {"r2_mean": +0.061, "nadeau_bengio_p": 0.018, "corrected_95ci": [0.012, 0.110]},
  "decision": "KEEP. Improves primary comparison materially. Move forward.",
  "notes": "Selection frequency of new p10 bout_speed feature: 14/15 outer folds.",
  "artifacts": ["runs/exp_0007/results.csv", "runs/exp_0007/summary.json"]
}
```

Rules for the loop:
- One variable changed per experiment where feasible. Bundled changes only when the individual components have already been vetted.
- Every experiment writes: full results CSV, `summary.json`, any figures, and its JSONL line.
- A decision is one of: `KEEP`, `REJECT`, `INCONCLUSIVE_RERUN`, `PARKED`.
- The "current best configuration" is tracked in `runs/current_best.json` and is updated only on `KEEP`.
- If an experiment's primary comparison worsens, do not silently discard it — log it as `REJECT` with the numbers. Negative results inform the next hypothesis.
- Maintain a short `research_log.md` (human-readable) that narrates the reasoning across experiments. Append, never rewrite.

### Background running
- Run experiments sequentially in a loop inside a long-lived session.
- After each experiment, update `dashboard.md` with a ranked table of all experiments by primary-comparison ΔR² and p-value.
- If an experiment is expected to take > 30 minutes, print a progress line every few minutes.
- On crash: log the traceback to `crashes/exp_XXXX.log`, mark the experiment `INCONCLUSIVE_RERUN`, and continue with the next idea — do not halt the whole loop on one failure.

---

## 8. Stop criteria (when to stop iterating and write the paper)

Stop when **any** of the following is true:

1. You have run ≥ 15 experiments and the top-3 configurations are within 0.005 R² of each other on the primary outcome (diminishing returns).
2. The primary comparison passes for **both co-primary outcomes**: ΔR² (gait bout vs 8-ft) on `cogn_global` *and* ΔAP (gait bout vs 8-ft) on `mobility_disability_binary`, both with Nadeau–Bengio corrected p < 0.05 after Holm–Bonferroni correction across the two, **and** robust under at least two sensitivity analyses from section 6f. (If only one of the two co-primaries passes, this is the weaker version of the claim — keep iterating up to the experiment ceiling rather than stopping.)
3. You have exhausted the improvement menu and further ideas would require new data or violate a rigor rule.
4. 50 experiments total, regardless of outcome — hard ceiling, because more iterations mean higher multiplicity risk.

On stop: freeze the current best configuration. Run `final_evaluation.py` once to evaluate it on the lockbox. Report lockbox numbers alongside dev-set numbers in the paper with explicit labels. If lockbox and dev numbers disagree substantially, report both honestly; do not re-iterate.

---

## 9. Figures (produce all four, publication quality)

All figures: vector (PDF + SVG), 300 DPI PNG backup, colorblind-safe palette, single-column (89 mm) or double-column (183 mm) widths, sans-serif, no chart junk.

- **Figure 1 — Study & methods schematic.** Cohort flow (n enrolled → n with valid wrist data → n with each outcome → dev / lockbox split), feature pipeline diagram (raw accel → bouts → features → 5 feature sets), nested CV diagram.
- **Figure 2 — Incremental value ladder (the headline figure).** A two-panel figure, one panel per co-primary outcome (`cogn_global` left, `mobility_disability_binary` right). Each panel is a stepped bar / dot plot showing performance for each of the seven feature sets in section 3, ordered as the ladder (Demographics → +8-ft → +PA → +Sleep → +Gait Bout → Full sensor → +Gait Bout +8-ft). Each bar shows mean ± 95% CI across outer folds. Overlay corrected p-values for the key adjacent contrasts: Demographics vs +8-ft, +8-ft vs +Gait Bout, +Gait Bout vs +Gait Bout +8-ft. Use the same y-axis units within each panel (R² left, AP right). This single figure tells the whole story: how much each layer of information adds, and that the result holds across the cognitive and the functional axis.
- **Figure 3 — Model diagnostics.** ROC + PR curves for binary outcomes, predicted vs observed scatter for `cogn_global`, all from out-of-fold predictions of the best configuration per feature set. Lockbox curves shown as a separate panel.
- **Figure 4 — What is the gait signal?** Top-20 permutation + SHAP importance for the best `cogn_global` model, grouped by biomechanical domain (pace / regularity / spectral / quantity / day-to-day / temporal). Optionally a PDP for the top 3 features. This is the "mechanistic" figure that makes the result credible rather than magical.

All figure-generation code lives in `figures/` and is driven from a single `make_figures.py` so the paper can be regenerated from frozen results in one command.

---

## 10. Nature-style paper draft

Write `paper/manuscript.md`. Constraints:

- **Format:** Nature Letter / Brief Communication style. Results-first, methods in a dedicated section at the end.
- **Main text length:** ~1500 words (hard cap 2000), excluding Methods, references, and captions.
- **Structure:**
  1. *Opening paragraph* — clinical problem (mobility, falls, cognition in older adults), limitation of the 8-ft walk test, promise of continuous wrist monitoring, one-sentence statement of what this paper shows.
  2. *Results* — lead with the headline comparison (Figure 2), then the diagnostic figure, then the mechanistic figure. State every effect size with its CI and corrected p-value. Do **not** hide negative findings.
  3. *Discussion* — one paragraph on what the gait signal probably reflects physiologically, one paragraph on limitations (circularity of parksc/motor10, single-cohort, cross-sectional baseline, wrist vs lower-back trade-off), one paragraph on clinical implications and next steps.
  4. *Methods* — full nested CV description, preprocessing, feature set definitions, statistical comparisons, lockbox protocol, software versions, data/code availability statement.
- **Tone:** calibrated. No "groundbreaking", no "unprecedented". Claims match evidence.
- **Tables:** one main-text table (Table 1: cohort characteristics by outcome status), rest in supplement.
- **Supplementary:** all seven outcomes × all feature sets × all models, sensitivity analyses, subgroup results, experiment log summary, permutation-null results, full feature importance tables.
- **Data and code availability:** point to a `code/` folder containing the frozen final pipeline plus a `REPRODUCE.md` that runs baseline + final evaluation end-to-end from the CSV.

Write the paper only after stop criteria are hit and the lockbox evaluation is complete. The paper reports lockbox numbers as the primary headline if they are available and pre-registered as such.

---

## 11. Repository layout you should create

```
project/
├── preregistration.md           # written before experiment 1, never edited
├── exclusions.md
├── lockbox_ids.csv
├── dev_ids.csv
├── prediction_pipeline.py       # refactored baseline
├── final_evaluation.py          # lockbox eval, run exactly once
├── experiments.jsonl            # one line per experiment
├── research_log.md              # narrative
├── dashboard.md                 # ranked experiment table
├── runs/
│   ├── baseline/
│   ├── exp_0001/
│   ├── ...
│   └── current_best.json
├── figures/
│   ├── make_figures.py
│   ├── fig1_schematic.pdf
│   ├── fig2_headline.pdf
│   ├── fig3_diagnostics.pdf
│   └── fig4_mechanism.pdf
├── paper/
│   ├── manuscript.md
│   ├── supplement.md
│   └── tables/
└── code/
    └── REPRODUCE.md
```

---

## 12. Starting instructions for the agent

When you begin, do these in order and report back after each:

1. Read `MODELING_OVERVIEW.md`, `PIPELINE_OVERVIEW.md`, and `prediction_pipeline.py` in full. Summarize in 10 bullets your understanding of the current pipeline and the top 5 things you suspect are leaving performance on the table. Do not run anything yet.
2. Write `preregistration.md` per section 3 of this brief. Commit it.
3. Create the lockbox split and write `lockbox_ids.csv`, `dev_ids.csv`.
4. Run the baseline on the dev set. Report baseline numbers.
5. Propose your first three experiments as hypotheses (one sentence each, with expected effect direction and why). Wait for nothing — execute them in order, logging each to `experiments.jsonl`.
6. Continue the autoresearch loop until a stop criterion in section 8 is met.
7. Run `final_evaluation.py` on the lockbox. Once.
8. Generate the four figures.
9. Draft the Nature-style manuscript.
10. Run yourself through `REVIEWER_CHECKLIST.md` (50 items). Every item must be ✅ or ⚠️-with-remediation before you surface the paper. Save the completed checklist to `paper/reviewer_checklist.md` with the sign-off paragraph.
11. Produce a one-page summary of what worked, what didn't, and the headline numbers, and surface it at the top of `research_log.md`.

Good luck. Optimize for a result that survives peer review, not for the biggest number.
