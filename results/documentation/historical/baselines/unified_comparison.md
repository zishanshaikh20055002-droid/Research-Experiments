# Unified Baseline and Neural Comparison

All values use the fixed 3,164 test sequence rows. Baselines are one deterministic fit; neural values are mean +/- sample standard deviation across the three saved seeds. Pooled window metrics weight each overlapping sequence equally. Equal-bearing macro metrics calculate a metric per test bearing and then average the nine values equally; they are not independent-sample inference.

## Pooled Window-Weighted Metrics

| Method | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| Training mean | 0.245 | 0.283 | -0.001 |
| Piecewise-linear age | 0.218 | 0.266 | 0.115 |
| M1 | 0.310 +/- 0.005 | 0.400 +/- 0.004 | -1.000 +/- 0.042 |
| M2 | 0.298 +/- 0.014 | 0.381 +/- 0.011 | -0.813 +/- 0.110 |
| M3 | 0.251 +/- 0.023 | 0.313 +/- 0.035 | -0.229 +/- 0.266 |
| M4 | 0.305 +/- 0.013 | 0.385 +/- 0.014 | -0.853 +/- 0.137 |
| M5 | 0.290 +/- 0.014 | 0.361 +/- 0.014 | -0.623 +/- 0.127 |

## Equal-Bearing Macro Errors

| Method | Macro MAE | Macro RMSE |
| --- | ---: | ---: |
| Training mean | 0.236 | 0.274 |
| Piecewise-linear age | 0.238 | 0.275 |
| M1 | 0.239 +/- 0.007 | 0.290 +/- 0.007 |
| M2 | 0.230 +/- 0.007 | 0.278 +/- 0.010 |
| M3 | 0.241 +/- 0.023 | 0.294 +/- 0.027 |
| M4 | 0.237 +/- 0.009 | 0.288 +/- 0.010 |
| M5 | 0.242 +/- 0.019 | 0.292 +/- 0.022 |

## Per-Bearing Metrics

| Bearing | Method | MAE | RMSE |
| --- | --- | ---: | ---: |
| Bearing1_3 | Training mean | 0.238 | 0.275 |
| Bearing1_3 | Piecewise-linear age | 0.077 | 0.137 |
| Bearing1_3 | M1 | 0.170 +/- 0.029 | 0.213 +/- 0.026 |
| Bearing1_3 | M2 | 0.134 +/- 0.020 | 0.171 +/- 0.035 |
| Bearing1_3 | M3 | 0.292 +/- 0.115 | 0.376 +/- 0.158 |
| Bearing1_3 | M4 | 0.138 +/- 0.002 | 0.182 +/- 0.006 |
| Bearing1_3 | M5 | 0.203 +/- 0.022 | 0.229 +/- 0.021 |
| Bearing1_4 | Training mean | 0.235 | 0.272 |
| Bearing1_4 | Piecewise-linear age | 0.129 | 0.153 |
| Bearing1_4 | M1 | 0.074 +/- 0.004 | 0.094 +/- 0.007 |
| Bearing1_4 | M2 | 0.135 +/- 0.045 | 0.170 +/- 0.061 |
| Bearing1_4 | M3 | 0.195 +/- 0.020 | 0.237 +/- 0.022 |
| Bearing1_4 | M4 | 0.085 +/- 0.014 | 0.107 +/- 0.019 |
| Bearing1_4 | M5 | 0.197 +/- 0.079 | 0.238 +/- 0.088 |
| Bearing1_5 | Training mean | 0.219 | 0.258 |
| Bearing1_5 | Piecewise-linear age | 0.396 | 0.428 |
| Bearing1_5 | M1 | 0.263 +/- 0.028 | 0.316 +/- 0.031 |
| Bearing1_5 | M2 | 0.207 +/- 0.047 | 0.267 +/- 0.049 |
| Bearing1_5 | M3 | 0.212 +/- 0.005 | 0.272 +/- 0.007 |
| Bearing1_5 | M4 | 0.208 +/- 0.041 | 0.260 +/- 0.041 |
| Bearing1_5 | M5 | 0.208 +/- 0.030 | 0.260 +/- 0.036 |
| Bearing2_3 | Training mean | 0.246 | 0.284 |
| Bearing2_3 | Piecewise-linear age | 0.282 | 0.327 |
| Bearing2_3 | M1 | 0.149 +/- 0.011 | 0.202 +/- 0.014 |
| Bearing2_3 | M2 | 0.149 +/- 0.040 | 0.186 +/- 0.064 |
| Bearing2_3 | M3 | 0.246 +/- 0.047 | 0.285 +/- 0.046 |
| Bearing2_3 | M4 | 0.186 +/- 0.026 | 0.247 +/- 0.031 |
| Bearing2_3 | M5 | 0.215 +/- 0.030 | 0.256 +/- 0.038 |
| Bearing2_4 | Training mean | 0.215 | 0.255 |
| Bearing2_4 | Piecewise-linear age | 0.451 | 0.481 |
| Bearing2_4 | M1 | 0.178 +/- 0.013 | 0.223 +/- 0.021 |
| Bearing2_4 | M2 | 0.208 +/- 0.028 | 0.252 +/- 0.040 |
| Bearing2_4 | M3 | 0.112 +/- 0.003 | 0.142 +/- 0.009 |
| Bearing2_4 | M4 | 0.205 +/- 0.024 | 0.256 +/- 0.031 |
| Bearing2_4 | M5 | 0.158 +/- 0.008 | 0.204 +/- 0.009 |
| Bearing2_5 | Training mean | 0.244 | 0.282 |
| Bearing2_5 | Piecewise-linear age | 0.219 | 0.255 |
| Bearing2_5 | M1 | 0.130 +/- 0.013 | 0.164 +/- 0.014 |
| Bearing2_5 | M2 | 0.126 +/- 0.036 | 0.154 +/- 0.036 |
| Bearing2_5 | M3 | 0.191 +/- 0.087 | 0.233 +/- 0.099 |
| Bearing2_5 | M4 | 0.161 +/- 0.030 | 0.191 +/- 0.027 |
| Bearing2_5 | M5 | 0.149 +/- 0.016 | 0.180 +/- 0.014 |
| Bearing3_3 | Training mean | 0.245 | 0.283 |
| Bearing3_3 | Piecewise-linear age | 0.232 | 0.270 |
| Bearing3_3 | M1 | 0.403 +/- 0.006 | 0.483 +/- 0.005 |
| Bearing3_3 | M2 | 0.404 +/- 0.024 | 0.480 +/- 0.023 |
| Bearing3_3 | M3 | 0.287 +/- 0.046 | 0.349 +/- 0.061 |
| Bearing3_3 | M4 | 0.421 +/- 0.014 | 0.495 +/- 0.019 |
| Bearing3_3 | M5 | 0.334 +/- 0.012 | 0.411 +/- 0.013 |
| Bearing3_4 | Training mean | 0.249 | 0.287 |
| Bearing3_4 | Piecewise-linear age | 0.207 | 0.254 |
| Bearing3_4 | M1 | 0.416 +/- 0.007 | 0.489 +/- 0.005 |
| Bearing3_4 | M2 | 0.394 +/- 0.009 | 0.462 +/- 0.009 |
| Bearing3_4 | M3 | 0.253 +/- 0.066 | 0.308 +/- 0.079 |
| Bearing3_4 | M4 | 0.385 +/- 0.011 | 0.452 +/- 0.013 |
| Bearing3_4 | M5 | 0.352 +/- 0.013 | 0.421 +/- 0.013 |
| Bearing3_5 | Training mean | 0.234 | 0.271 |
| Bearing3_5 | Piecewise-linear age | 0.149 | 0.169 |
| Bearing3_5 | M1 | 0.366 +/- 0.007 | 0.431 +/- 0.006 |
| Bearing3_5 | M2 | 0.308 +/- 0.034 | 0.363 +/- 0.045 |
| Bearing3_5 | M3 | 0.379 +/- 0.021 | 0.446 +/- 0.026 |
| Bearing3_5 | M4 | 0.343 +/- 0.061 | 0.403 +/- 0.061 |
| Bearing3_5 | M5 | 0.364 +/- 0.012 | 0.425 +/- 0.011 |

## Per-Condition MAE

| Condition | Method | MAE |
| --- | --- | ---: |
| 35Hz12kN | Training mean | 0.234 |
| 35Hz12kN | Piecewise-linear age | 0.141 |
| 35Hz12kN | M1 | 0.147 +/- 0.015 |
| 35Hz12kN | M2 | 0.145 +/- 0.017 |
| 35Hz12kN | M3 | 0.245 +/- 0.064 |
| 35Hz12kN | M4 | 0.128 +/- 0.011 |
| 35Hz12kN | M5 | 0.201 +/- 0.044 |
| 37.5Hz11kN | Training mean | 0.244 |
| 37.5Hz11kN | Piecewise-linear age | 0.265 |
| 37.5Hz11kN | M1 | 0.143 +/- 0.003 |
| 37.5Hz11kN | M2 | 0.143 +/- 0.037 |
| 37.5Hz11kN | M3 | 0.221 +/- 0.060 |
| 37.5Hz11kN | M4 | 0.177 +/- 0.027 |
| 37.5Hz11kN | M5 | 0.188 +/- 0.022 |
| 40Hz10kN | Training mean | 0.247 |
| 40Hz10kN | Piecewise-linear age | 0.209 |
| 40Hz10kN | M1 | 0.411 +/- 0.007 |
| 40Hz10kN | M2 | 0.391 +/- 0.010 |
| 40Hz10kN | M3 | 0.266 +/- 0.056 |
| 40Hz10kN | M4 | 0.389 +/- 0.011 |
| 40Hz10kN | M5 | 0.349 +/- 0.012 |

The piecewise-linear age baseline is intentionally unconstrained. Its fitted knot values include an increasing interval, so it is not a monotonic baseline.
