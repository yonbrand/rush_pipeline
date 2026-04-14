# Subject Exclusions Log

Source: `outputs/tables/merged_gait_clinical_abl.csv`, n = 790.

## Pre-modeling split (2026-04-14)

| Step | n in | n out | Reason |
|---|---|---|---|
| Source CSV | 790 | 790 | Loaded |
| Stratified 15% lockbox | 790 | 119 lockbox + 671 dev | Held out per preregistration §8 (seed 20260414, stratified on `mobility_disability_binary`) |

5 subjects with missing `mobility_disability_binary` were placed into a separate `"missing"` stratum so they were not silently lost; the split-balance check confirms they are distributed roughly proportionally.

## Per-outcome modeling exclusions

For every (outcome × feature set) the dev set is further restricted by `dropna(subset=[outcome])`. Counts per outcome on the dev set:

| Outcome | n in dev | n analyzed | dropped (missing outcome) |
|---|---|---|---|
| `cogn_global` | 671 | (computed at runtime) | (computed at runtime) |
| `mobility_disability_binary` | 671 | (computed at runtime) | (computed at runtime) |
| `falls_binary` | 671 | (computed at runtime) | (computed at runtime) |
| `cognitive_impairment` | 671 | (computed at runtime) | (computed at runtime) |
| `parkinsonism_yn` | 671 | (computed at runtime) | (computed at runtime) |
| `parksc` | 671 | (computed at runtime) | (computed at runtime) |
| `motor10` | 671 | (computed at runtime) | (computed at runtime) |

Final per-outcome counts are written to `runs/baseline/n_per_outcome.json` after the baseline run.

## Wear-time / quality exclusions

**None pre-specified before the baseline run.** If a wear-time or bout-count quality threshold is added later, it must be specified before its CV results are inspected, and the threshold + timestamp logged here. (Brief §6a, hard rule §4.7.)
