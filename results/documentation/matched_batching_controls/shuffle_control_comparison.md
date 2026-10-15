# Matched-batching sensitivity comparison

Exploratory sensitivity analysis on the previously inspected test set; not an untouched evaluation. No significance testing or causal claim is made.

Pooled metrics weight sequence rows equally; macro metrics average per-bearing scores equally across nine bearings. Three-seed entries are mean ± sample SD. Neural entries summarize three distinct seeded runs; they are not treated as statistically independent bearing samples.

## Aggregate results

| model         |   n_seeds |   mae_mean |   mae_sample_sd |   rmse_mean |   rmse_sample_sd |   r2_mean |   r2_sample_sd |   macro_mae_mean |   macro_mae_sample_sd |   macro_rmse_mean |   macro_rmse_sample_sd |   strict_violation_rate_mean |   strict_violation_rate_sample_sd |   mean_positive_violation_mean |   mean_positive_violation_sample_sd |   mean_bearing_spearman_mean |   mean_bearing_spearman_sample_sd |   mean_abs_adjacent_change_mean |   mean_abs_adjacent_change_sample_sd |
|:--------------|----------:|-----------:|----------------:|------------:|-----------------:|----------:|---------------:|-----------------:|----------------------:|------------------:|-----------------------:|-----------------------------:|----------------------------------:|-------------------------------:|------------------------------------:|-----------------------------:|----------------------------------:|--------------------------------:|-------------------------------------:|
| M2-shuffled   |         3 |   0.297749 |       0.0138937 |    0.381213 |        0.0114925 | -0.813406 |       0.109549 |         0.229556 |            0.00669816 |          0.278336 |             0.0104752  |                     0.485261 |                        0.00682011 |                      0.0134101 |                          0.00110481 |                    -0.64708  |                         0.0519205 |                       0.0147688 |                           0.001044   |
| M3-shuffled   |         3 |   0.251374 |       0.0228724 |    0.312603 |        0.0346008 | -0.22862  |       0.265932 |         0.240879 |            0.0228791  |          0.294074 |             0.0270962  |                     0.490016 |                        0.00852255 |                      0.0157913 |                          0.00289353 |                    -0.54728  |                         0.150756  |                       0.0165723 |                           0.00297658 |
| M4-shuffled   |         3 |   0.304551 |       0.0133814 |    0.385299 |        0.0141331 | -0.853026 |       0.136978 |         0.236656 |            0.00940183 |          0.288148 |             0.00993472 |                     0.473217 |                        0.00952455 |                      0.0119428 |                          0.0023415  |                    -0.66767  |                         0.0258711 |                       0.0133168 |                           0.00203947 |
| M5-shuffled   |         3 |   0.289684 |       0.0140927 |    0.360542 |        0.0141393 | -0.622756 |       0.127043 |         0.242052 |            0.0193444  |          0.291685 |             0.0221899  |                     0.493608 |                        0.00461859 |                      0.0189169 |                          0.00273497 |                    -0.679674 |                         0.0117749 |                       0.0203581 |                           0.00263233 |
| M2-unshuffled |         3 |   0.241468 |       0.0315976 |    0.298998 |        0.0468671 | -0.133154 |       0.364549 |         0.223977 |            0.00570624 |          0.273588 |             0.00589974 |                     0.490016 |                        0.00664099 |                      0.0140365 |                          0.00116662 |                    -0.60218  |                         0.103743  |                       0.0149673 |                           0.00145904 |
| M4-unshuffled |         3 |   0.301129 |       0.0208155 |    0.379818 |        0.0271966 | -0.805222 |       0.255396 |         0.238536 |            0.0145333  |          0.289571 |             0.0141693  |                     0.49065  |                        0.0072624  |                      0.0194077 |                          0.0018748  |                    -0.647338 |                         0.0665426 |                       0.020895  |                           0.00185521 |

## Per-condition MAE

| model         | condition   |   mae_mean |   mae_sample_sd |
|:--------------|:------------|-----------:|----------------:|
| M2-shuffled   | 35Hz12kN    |   0.144562 |      0.0174771  |
| M2-shuffled   | 37.5Hz11kN  |   0.142892 |      0.0368756  |
| M2-shuffled   | 40Hz10kN    |   0.391095 |      0.0104777  |
| M3-shuffled   | 35Hz12kN    |   0.244962 |      0.0641334  |
| M3-shuffled   | 37.5Hz11kN  |   0.220828 |      0.0602861  |
| M3-shuffled   | 40Hz10kN    |   0.266105 |      0.0562464  |
| M4-shuffled   | 35Hz12kN    |   0.128025 |      0.0112425  |
| M4-shuffled   | 37.5Hz11kN  |   0.177104 |      0.027254   |
| M4-shuffled   | 40Hz10kN    |   0.389178 |      0.01077    |
| M5-shuffled   | 35Hz12kN    |   0.20125  |      0.0438838  |
| M5-shuffled   | 37.5Hz11kN  |   0.188273 |      0.0224286  |
| M5-shuffled   | 40Hz10kN    |   0.348976 |      0.012447   |
| M2-unshuffled | 35Hz12kN    |   0.214663 |      0.0577835  |
| M2-unshuffled | 37.5Hz11kN  |   0.191222 |      0.0217197  |
| M2-unshuffled | 40Hz10kN    |   0.268214 |      0.06629    |
| M4-unshuffled | 35Hz12kN    |   0.166796 |      0.00700092 |
| M4-unshuffled | 37.5Hz11kN  |   0.179821 |      0.0186366  |
| M4-unshuffled | 40Hz10kN    |   0.37647  |      0.0388227  |

## Matched-seed differences

Differences are control minus comparator for each seed; negative error deltas favor the control. These descriptive differences do not establish causality.

### M2-unshuffled vs M2-shuffled

| Seed | Δ MAE | Δ RMSE | Δ R² | Δ macro MAE | Δ macro RMSE | Δ violation rate | Δ violation magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.035872 | -0.040490 | 0.376742 | -0.012428 | -0.017393 | 0.009826 | 0.002286 |
| 123 | -0.065371 | -0.102810 | 0.842420 | 0.000965 | 0.005923 | 0.012361 | 0.000756 |
| 2024 | -0.067599 | -0.103345 | 0.821595 | -0.005275 | -0.002773 | -0.007924 | -0.001162 |

### M2-unshuffled vs M3-shuffled

| Seed | Δ MAE | Δ RMSE | Δ R² | Δ macro MAE | Δ macro RMSE | Δ violation rate | Δ violation magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.006236 | 0.009197 | -0.079877 | 0.001149 | -0.000131 | 0.003803 | -0.001257 |
| 123 | -0.030176 | -0.041810 | 0.310782 | -0.037163 | -0.044853 | -0.002536 | -0.003781 |
| 2024 | -0.005779 | -0.008205 | 0.055493 | -0.014693 | -0.016475 | -0.001268 | -0.000227 |

### M4-unshuffled vs M4-shuffled

| Seed | Δ MAE | Δ RMSE | Δ R² | Δ macro MAE | Δ macro RMSE | Δ violation rate | Δ violation magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.001653 | 0.003145 | -0.031598 | 0.008693 | 0.006754 | 0.019334 | 0.007968 |
| 123 | -0.017912 | -0.029699 | 0.270701 | 0.003539 | 0.002064 | 0.014580 | 0.007413 |
| 2024 | 0.005994 | 0.010112 | -0.095691 | -0.006594 | -0.004551 | 0.018384 | 0.007013 |

### M4-unshuffled vs M5-shuffled

| Seed | Δ MAE | Δ RMSE | Δ R² | Δ macro MAE | Δ macro RMSE | Δ violation rate | Δ violation magnitude |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.033458 | 0.043429 | -0.414519 | 0.023118 | 0.026038 | -0.011410 | 0.001499 |
| 123 | 0.003571 | 0.004406 | -0.038290 | 0.001726 | 0.004046 | 0.008241 | -0.000046 |
| 2024 | -0.002693 | 0.009994 | -0.094589 | -0.035393 | -0.036427 | -0.005705 | 0.000020 |
