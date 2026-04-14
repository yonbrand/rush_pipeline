# Pre-Submission Reviewer Checklist

**Purpose:** The agent runs through this checklist itself *before* declaring the manuscript done. Every item must be answered with one of: ✅ pass, ⚠️ partial (with remediation note), ❌ fail (blocks submission). The completed checklist is saved to `paper/reviewer_checklist.md` and referenced in the cover letter.

This is adapted from TRIPOD+AI, CONSORT-AI, and the kinds of comments that actually kill ML-in-clinical-research papers in revision.

---

## A. Pre-registration & data integrity

1. Does `preregistration.md` exist, was it written before any modeling, and has it been left untouched since? (`git log preregistration.md` should show one commit.)
2. Are the co-primary outcomes stated in the paper (`cogn_global` and `mobility_disability_binary`) identical to the ones in the preregistration?
3. Is the primary comparison (Gait Bout + demographics vs 8-ft Gait Speed + demographics, run identically for both co-primaries with Holm–Bonferroni correction) stated in the paper identical to the one in the preregistration?
4. Were any deviations from the preregistration disclosed explicitly in the Methods section under a "Deviations from preregistration" subheading? (If none: state that explicitly.)
5. Does `exclusions.md` account for every subject dropped between the source CSV (790) and the final analytic sample, with a numerical flow that matches Figure 1?
6. Is the lockbox split reproducible from a fixed seed and saved to `lockbox_ids.csv`?
7. Confirm by inspection of the code that no function in the iteration pipeline reads `lockbox_ids.csv` except `final_evaluation.py`.
8. Was `final_evaluation.py` run exactly once? (Check `runs/lockbox/` exists and contains exactly one timestamped run.)

## B. Leakage audit (the section reviewers care about most)

9. For every preprocessing step (imputation, scaling, variance filter, correlation filter, feature selection, PCA, SHAP-based selection), confirm by code inspection that it is inside the sklearn `Pipeline` and therefore re-fit on each training fold. List each step and the line number where it is wrapped.
10. Confirm that hyperparameter tuning happens only in the inner CV loop, never on the outer test fold.
11. Confirm that no feature was engineered using information from the outcome variable outside a CV fold (target encoding, target-aware binning, etc.). If any such feature exists, it must be re-implemented as a leak-free transformer.
12. Confirm that the `random_state` for the outer CV split has not changed between the baseline run and the final run. (The Nadeau–Bengio test is only valid if folds are paired.)
13. Confirm that subjects in the lockbox do not also appear in any dev fold. (Run an explicit set-intersection check.)
14. Confirm that no figure, table, or sentence in the manuscript was generated from data that included the lockbox during model fitting.
15. If wear-time / bout-count quality filters were applied, confirm the threshold was specified before seeing modeling results, with timestamp evidence.

## C. Statistical reporting

16. Every effect size in the manuscript is reported with an uncertainty interval (SD across folds, bootstrap CI, or analytic CI).
17. Every p-value in the manuscript is from a test that respects the dependency structure of the data (Nadeau–Bengio for paired CV scores, DeLong for AUCs, bootstrap for ΔAP). No naive paired t-tests on CV folds.
18. Multiple-testing correction (Holm–Bonferroni) is applied across pairwise feature-set comparisons within each outcome, and across the secondary outcomes. The correction is described in Methods.
19. Negative results from `experiments.jsonl` that bear on the primary claim are mentioned in the manuscript or supplement, not silently dropped.
20. The number of experiments run is disclosed in the supplement, with the experiment log either included or made available.
21. A permutation-null check has been run (shuffle outcome, re-fit pipeline) and the resulting null R² is approximately zero. The result is reported in the supplement.
22. If any subgroup result is highlighted in the main text, it was pre-specified, not data-mined.

## D. The circularity issue

23. The manuscript explicitly states that `parksc`, `motor10`, and `parkinsonism_yn` are all derived from the modified UPDRS motor exam, which includes the 8-ft walk test as a scoring component.
24. No comparison of "gait bout vs 8-ft speed" on `parksc`, `motor10`, or `parkinsonism_yn` is presented as a primary or supporting result. Such comparisons appear only in a clearly labeled exploratory / sensitivity section.
25. The headline numbers in the abstract are from `cogn_global` and `mobility_disability_binary` (the non-circular co-primaries), not from any mUPDRS-derived outcome.

## E. The incremental value ladder (specific to this paper)

26. All seven feature sets in the ladder share the same demographic baseline (`age_at_visit`, `msex`, `educ`). Confirm by inspecting the feature-list dump for each set.
27. Each step in the ladder is reported with an absolute performance value AND a delta vs the previous step, with CI and corrected p-value.
28. The "Demographics only" floor is reported and not omitted, even if it is non-trivial — reviewers will assume the worst if it's missing.
29. The "+Gait Bout +8-ft speed" set is reported, and the result of "does adding 8-ft speed *to* daily-living gait help?" is stated explicitly. If the answer is no, that strengthens the paper rather than weakening it; do not bury it.
30. The full incremental-ladder analysis is run for **both co-primary outcomes** (`cogn_global` regression and `mobility_disability_binary` classification), shown as the two panels of Figure 2, demonstrating that the result holds across the cognitive and the functional axis. Additionally, the ladder is run for at least one secondary outcome (e.g., `falls_binary`) in the supplement.

## F. Generalization & robustness

31. Lockbox performance is reported alongside dev-set performance for every headline number. If they differ by more than ~20% relative, the discrepancy is discussed honestly.
32. At least two sensitivity analyses from section 6f of the brief have been run, and their results are reported in the supplement.
33. Subgroup results (age tertile, sex) are in the supplement and are consistent with the main result, OR inconsistencies are discussed.
34. If a calibration assessment is appropriate (binary outcomes), a reliability diagram or Brier score is reported.

## G. Reproducibility

35. `REPRODUCE.md` exists and has been tested: a fresh checkout + the listed commands reproduce baseline numbers and final-evaluation numbers within Monte-Carlo noise of what is reported in the paper.
36. Software versions (`python`, `numpy`, `scikit-learn`, `xgboost`, `pandas`, OS) are pinned in `environment.yml` or `requirements.txt`.
37. Random seeds are set globally and per-component (numpy, sklearn, xgboost) and listed in Methods.
38. The frozen final pipeline is in `code/` and is the same code path as what produced the lockbox numbers. Not a cleaned-up rewrite.
39. Data availability statement is present and accurate (whether the CSV can be shared, under what conditions, and where).
40. Code availability statement is present and points to the actual repo / archive.

## H. Manuscript craftsmanship

41. The abstract states the claim, the magnitude with CI, and the comparison group, in that order. Not "we developed a model that…".
42. The first paragraph of Results contains the headline ΔR² with its CI and p-value. Reviewers should not have to scroll to find the main result.
43. Every figure has a self-contained caption: a reader can understand the figure without the main text.
44. Every figure number, table number, and reference cited in the text actually exists.
45. The Limitations paragraph addresses: (a) circularity of `parksc` / `motor10`, (b) single-cohort / external validity, (c) wrist (vs lower-back) sensor placement, (d) cross-sectional baseline (no longitudinal validation in this paper), (e) any wear-time or quality exclusions.
46. No "groundbreaking", "unprecedented", "revolutionary", "novel" (use sparingly), or "first to show" (only if literally true and verified by a literature check).
47. Word count is within 2000 for the main text excluding Methods. Verify with `wc -w`.
48. Author contributions, conflict of interest, funding, and ethics statements are present (placeholder text is fine; flag them as TODO for the human owner).

## I. The "would I send this back?" gut check

49. Read the abstract and Figure 2 in isolation. Does a skeptical reviewer who has never met you walk away thinking "the daily-living gait signal is real and adds value over what we already have", or thinking "they tortured the data until something popped"? If the latter, identify what is missing and fix it before submitting.
50. Search the manuscript for any sentence that says something stronger than the data supports. Soften it.

---

## Sign-off

When all 50 items are ✅ or ⚠️ with an acceptable remediation, write a one-paragraph sign-off at the bottom of `paper/reviewer_checklist.md` that says: "All 50 pre-submission checks passed or were resolved as documented. The frozen final configuration is `runs/exp_XXXX/`. Lockbox evaluation timestamp: `YYYY-MM-DD HH:MM:SS`. Headline results: (1) ΔR² = X.XXX [95% CI X.XXX, X.XXX], Holm-corrected p = X.XXX, on `cogn_global`, gait bout (+ demographics) vs 8-ft gait speed (+ demographics); (2) ΔAP = X.XXX [95% CI X.XXX, X.XXX], Holm-corrected p = X.XXX, on `mobility_disability_binary`, same comparison. State explicitly whether both, one, or neither of the co-primary tests passed at family-wise α = 0.05."

Then, and only then, surface the manuscript for human review.
