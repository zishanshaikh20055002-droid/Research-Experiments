# Offline Post-hoc Analysis Report

Date: 2026-10-15

## Scope

No model was trained and no raw run record was modified. The analysis used the saved 3,164-row test prediction files. Non-increasing isotonic regression was fitted separately within each complete bearing using only chronological endpoint positions and model predictions. True RUL labels were excluded from fitting and were used only for evaluation.

## Validation check

The previously analysed M1 prediction file reproduced the expected pooled-MAE change exactly enough for the preregistered gate:

- Raw M1 pooled MAE: `0.306988442979`
- Isotonic M1 pooled MAE: `0.298858289425`

## Three-seed saved-prediction results

Values below are arithmetic means across the three saved seeds. Standard deviations in the CSV are sample standard deviations.

| Method | Raw pooled MAE | Isotonic pooled MAE | Raw pooled RMSE | Isotonic pooled RMSE |
|---|---:|---:|---:|---:|
| M1 | 0.310181 | 0.303975 | 0.400495 | 0.394772 |
| M2 | 0.297749 | 0.291225 | 0.381213 | 0.376065 |
| M2-U | 0.241468 | 0.237826 | 0.298998 | 0.293373 |
| M3-U | 0.251374 | 0.246078 | 0.312603 | 0.304038 |
| M4 | 0.304551 | 0.294708 | 0.385299 | 0.376373 |
| M5 | 0.289684 | 0.281512 | 0.360542 | 0.353278 |
| Unconstrained age reference | 0.217944 | 0.196840 | 0.266408 | 0.232074 |

The projection enforces trajectory shape by construction. These lower saved-data errors are not evidence of a physical degradation law, a causal model improvement, or independent test confirmation.

## Whole-bearing bootstrap

- Resampling unit: one complete test bearing.
- Test bearings: nine.
- Replicates: 1,000.
- Random seed: 20261015.
- Pairing: identical sampled bearing IDs were used for every method and variant in each replicate.
- Neural aggregation: each metric was calculated per seed and then averaged across the three seeds.
- Primary statistics: equal-bearing macro MAE/RMSE.
- Secondary statistics: pooled MAE/RMSE over all windows contributed by each sampled bearing multiset.

Selected macro-MAE percentile intervals:

| Method | Variant | Point estimate | 95% percentile CI |
|---|---|---:|---:|
| M2-U | Raw | 0.224 | [0.176, 0.279] |
| M2-U | Isotonic | 0.220 | [0.173, 0.274] |
| M3-U | Raw | 0.241 | [0.195, 0.290] |
| M3-U | Isotonic | 0.235 | [0.190, 0.284] |
| Age reference | Raw | 0.238 | [0.166, 0.312] |
| Age reference | Isotonic | 0.222 | [0.152, 0.302] |

Intervals are broad because only nine independent bearing trajectories are available. The test set had already been inspected, so this is exploratory sampling sensitivity, not confirmatory uncertainty.

## Lambda sweep status

The six requested lambda=1/10 runs were not started. The machine has no usable local PyTorch interpreter: the prior environment is blocked by Windows Application Control while loading `torch/lib/c10.dll`, and the isolated baseline environment has no PyTorch. No package installation, policy bypass, or incompatible approximation was attempted.

Historical lambda=0 and lambda=0.1 artifacts do not contain saved validation predictions. They therefore cannot supply the preregistered validation RUL MAE and validation violation rate without checkpoint inference. No lambda was selected and no lambda result was added to the manuscript.

