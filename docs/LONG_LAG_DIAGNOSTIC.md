# Long-Lag Degradation-Consistency Diagnostic

## Scope and rules

This offline analysis used only saved raw test predictions for M1-M5, historical M3-U, and M2-U, across seeds 42, 123, and 2024. Every prediction file had 3,164 unique rows over the same nine fixed test bearings. Predictions were sorted by `end_position` within each bearing; no cross-bearing pair was evaluated.

For lag `k`, a strict upward violation is `prediction[t + k] > prediction[t]`. The mean positive increase is the arithmetic mean of positive differences only; it is zero when an eligible bearing has no positive differences. Results use no numerical tolerance for raw predictions.

## Aggregation

For each seed and lag:

- Macro metrics first compute a rate or mean-positive-increase for each eligible bearing and then average those bearing values equally.
- Pooled metrics combine every valid transition from eligible bearings before calculating the rate or mean-positive-increase.
- Across neural seeds, reported CSV aggregate means and SDs are arithmetic means and sample SDs of the per-seed values.

The diagnostic includes lag 1 to retain comparability with the existing adjacent-step metric, plus lags 10 and 20. All nine test bearings are eligible for every reported lag. There are 3,155, 3,074, and 2,984 valid pooled transitions at lags 1, 10, and 20 respectively. Bearing2_4 has 33 windows and therefore contributes 32, 23, and 13 transitions at those lags; it is never treated as an ineligible zero-valued bearing.

## Selected raw results

| Method | Lag | Macro violation rate, mean ± SD | Pooled violation rate, mean ± SD | Pooled positive increase, mean ± SD |
|---|---:|---:|---:|---:|
| M2-U | 10 | 0.371505 ± 0.028204 | 0.434396 ± 0.009525 | 0.021150 ± 0.002615 |
| M2-U | 20 | 0.320773 ± 0.040856 | 0.405608 ± 0.023296 | 0.024901 ± 0.004400 |
| M3-U | 10 | 0.391790 ± 0.044763 | 0.447083 ± 0.021156 | 0.024788 ± 0.004171 |
| M3-U | 20 | 0.354928 ± 0.061644 | 0.431189 ± 0.033027 | 0.028998 ± 0.005310 |

These are exploratory diagnostics on a previously inspected fixed test set. They do not isolate a causal monotonicity-loss effect because objective and batching history remain confounded in the historical comparison.

## Isotonic sanity check

The already saved non-increasing isotonic projections have zero strict upward violations for every model, seed, and lag. That is expected by construction and is reported only as a numerical sanity check, not as evidence that a model learned a physical law. Raw-versus-projected MAE/RMSE remains the relevant post-hoc comparison and retains its documented label-free fitting restrictions.

## Files

- `long_lag_per_bearing.csv`: one record per method, seed, variant, lag, and bearing; ineligible bearings would appear with `eligible=False` and `NaN` metrics.
- `long_lag_per_seed.csv`: macro and pooled raw metrics per saved seed.
- `long_lag_aggregate.csv`: mean and sample SD across seeds.
- `long_lag_isotonic_sanity.csv`: zero-violation check for projected outputs.
- `long_lag_sources.json`: exact raw prediction sources and lags.

