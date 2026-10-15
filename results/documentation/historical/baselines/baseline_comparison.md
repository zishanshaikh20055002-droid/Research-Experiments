# Offline RUL Baseline Comparison

Baselines are evaluated on the same 3,164 fixed test sequence rows as the saved M1-M5 predictions. Neural values are three-seed mean +/- sample SD; each baseline is one deterministic fit.

## Pooled Window-Weighted Metrics

Every test sequence receives equal weight; overlapping windows are not independent bearing samples.

| Method | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| training_mean | 0.245 | 0.283 | -0.001 |
| piecewise_linear_age | 0.218 | 0.266 | 0.115 |
| M1 (3-seed mean +/- sample SD) | 0.310 +/- 0.005 | 0.400 +/- 0.004 | -1.000 +/- 0.042 |
| M2 (3-seed mean +/- sample SD) | 0.298 +/- 0.014 | 0.381 +/- 0.011 | -0.813 +/- 0.110 |
| M3 (3-seed mean +/- sample SD) | 0.251 +/- 0.023 | 0.313 +/- 0.035 | -0.229 +/- 0.266 |
| M4 (3-seed mean +/- sample SD) | 0.305 +/- 0.013 | 0.385 +/- 0.014 | -0.853 +/- 0.137 |
| M5 (3-seed mean +/- sample SD) | 0.290 +/- 0.014 | 0.361 +/- 0.014 | -0.623 +/- 0.127 |

## Equal-Bearing Macro Errors

Each test bearing receives equal weight after calculating its metrics separately. The neural standard deviation is across seeds of the nine-bearing macro metric.

| Method | Macro MAE | Macro RMSE |
| --- | ---: | ---: |
| training_mean | 0.236 | 0.274 |
| piecewise_linear_age | 0.238 | 0.275 |
| M1 (3-seed mean +/- sample SD) | 0.239 +/- 0.007 | 0.290 +/- 0.007 |
| M2 (3-seed mean +/- sample SD) | 0.230 +/- 0.007 | 0.278 +/- 0.010 |
| M3 (3-seed mean +/- sample SD) | 0.241 +/- 0.023 | 0.294 +/- 0.027 |
| M4 (3-seed mean +/- sample SD) | 0.237 +/- 0.009 | 0.288 +/- 0.010 |
| M5 (3-seed mean +/- sample SD) | 0.242 +/- 0.019 | 0.292 +/- 0.022 |

## Per-Condition MAE

| Condition | Method | MAE | RMSE |
| --- | --- | ---: | ---: |
| 35Hz12kN | piecewise_linear_age | 0.141 | 0.209 |
| 35Hz12kN | training_mean | 0.234 | 0.271 |
| 37.5Hz11kN | piecewise_linear_age | 0.265 | 0.310 |
| 37.5Hz11kN | training_mean | 0.244 | 0.282 |
| 40Hz10kN | piecewise_linear_age | 0.209 | 0.253 |
| 40Hz10kN | training_mean | 0.247 | 0.285 |
| 35Hz12kN | M1 (3-seed mean +/- sample SD) | 0.147 +/- 0.015 | 0.199 +/- 0.017 |
| 35Hz12kN | M2 (3-seed mean +/- sample SD) | 0.145 +/- 0.017 | 0.191 +/- 0.014 |
| 35Hz12kN | M3 (3-seed mean +/- sample SD) | 0.245 +/- 0.064 | 0.320 +/- 0.094 |
| 35Hz12kN | M4 (3-seed mean +/- sample SD) | 0.128 +/- 0.011 | 0.173 +/- 0.015 |
| 35Hz12kN | M5 (3-seed mean +/- sample SD) | 0.201 +/- 0.044 | 0.238 +/- 0.049 |
| 37.5Hz11kN | M1 (3-seed mean +/- sample SD) | 0.143 +/- 0.003 | 0.190 +/- 0.004 |
| 37.5Hz11kN | M2 (3-seed mean +/- sample SD) | 0.143 +/- 0.037 | 0.179 +/- 0.051 |
| 37.5Hz11kN | M3 (3-seed mean +/- sample SD) | 0.221 +/- 0.060 | 0.264 +/- 0.063 |
| 37.5Hz11kN | M4 (3-seed mean +/- sample SD) | 0.177 +/- 0.027 | 0.228 +/- 0.030 |
| 37.5Hz11kN | M5 (3-seed mean +/- sample SD) | 0.188 +/- 0.022 | 0.229 +/- 0.028 |
| 40Hz10kN | M1 (3-seed mean +/- sample SD) | 0.411 +/- 0.007 | 0.485 +/- 0.005 |
| 40Hz10kN | M2 (3-seed mean +/- sample SD) | 0.391 +/- 0.010 | 0.461 +/- 0.009 |
| 40Hz10kN | M3 (3-seed mean +/- sample SD) | 0.266 +/- 0.056 | 0.325 +/- 0.067 |
| 40Hz10kN | M4 (3-seed mean +/- sample SD) | 0.389 +/- 0.011 | 0.458 +/- 0.013 |
| 40Hz10kN | M5 (3-seed mean +/- sample SD) | 0.349 +/- 0.012 | 0.419 +/- 0.013 |

## Per-Bearing Metrics

See `per_bearing_metrics.csv` for baseline MAE, RMSE, R-squared, and the corresponding sequence counts for every bearing.

## Interpretation

Both simple baselines have lower pooled window MAE/RMSE than the best neural mean (M3: 0.251/0.313). The training-mean baseline is 0.245/0.283; the piecewise age baseline is 0.218/0.266. Under equal-bearing macro MAE, the lowest neural mean is M2 (0.230), compared with 0.236 for the training-mean baseline and 0.238 for the piecewise age baseline. The ranking therefore depends on whether windows or bearings receive equal weight.

The comparison is descriptive. The baseline fits use training labels only, and no baseline result was used to choose a model, knot count, or validation/test protocol. Differences among overlapping test windows do not represent independent bearing-level replication.

Training-mean estimate (train sequence targets only): `0.496204401`.
Piecewise-linear age knots (training positions only): `9.0, 100.0, 473.0, 2537.0`.
Fitted knot values before output clipping: `0.946780, 0.344974, 0.624107, 0.088323`.
