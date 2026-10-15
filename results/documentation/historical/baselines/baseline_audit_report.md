# Baseline Integration Audit

## Scope

This audit reviewed `evaluate_baselines.py`, `baseline_methodology.md`, and
the three generated metric CSV files. It also compared the reconstructed test
sequence keys and labels with all saved M1-M5 predictions (three seeds per
model). It is an offline audit; no model was retrained.

## Verified Findings

- **Fit data:** the training-mean reference is fitted from the 5,764 sequence
  endpoints from Bearing1_1, Bearing1_2, Bearing2_1, Bearing3_1, and
  Bearing3_2 only. Bearing2_2 is excluded, as are all nine test bearings.
- **Age baseline fit:** its knot positions use only the five training bearings'
  chronological endpoint positions, at fixed weighted empirical quantiles
  0, 1/3, 2/3, and 1. Equal total fitting weight is assigned to each training
  bearing. Weighted least squares uses only their RUL labels.
- **Prediction-time input:** the age predictor receives only the observed
  sequence end position. It never receives a test bearing's last position,
  total recording count, final lifetime, future observations, validation/test
  labels, vibration features, or fault labels. Values outside the training
  position support are endpoint-clamped, then predictions are clipped to
  [0, 1]. This is leakage-free under the stated online-position assumption.
- **Test alignment:** the reconstructed test set has 3,164 rows. For each of
  15 saved neural prediction files, the bearing ID, start position, end
  position, and target match exactly, allowing an absolute target tolerance of
  1e-7 for float32 serialization.
- **Metrics:** pooled MAE/RMSE/R-squared give each test sequence equal weight.
  Macro MAE/RMSE first calculate a metric within each of nine bearings and
  then take their unweighted average. The distinction is explicitly stated in
  the methodology and comparison tables.

## Piecewise-Linear Baseline Limitation

The implementation is mathematically consistent with the documented
piecewise-linear interpolation basis and weighted least-squares objective. Its
training-only knots are 9, 100, 473, and 2,537; the fitted knot values are
0.946780, 0.344974, 0.624107, and 0.088323. The increase from 0.344974 to
0.624107 is a genuine non-monotonic fitted segment. No monotonicity constraint
was specified or applied. It must therefore be described as an *unconstrained
piecewise-linear age baseline*, never as a monotonic reference.

The four-knot design is a fixed modeling choice rather than a complexity
chosen using validation or test performance. It is documented and does not
constitute test leakage, but it remains a methodological assumption that
should be reported.

## Audit Conclusion

No substantive implementation or leakage flaw was found that prevents the
baseline results from being integrated. The retrospective full-trajectory RUL
label is an evaluation characteristic shared by all compared methods; it is
not used as an input to either baseline. Rankings must remain qualified by the
aggregation protocol because pooled windows are overlapping and long bearings
receive more weight than short bearings.
