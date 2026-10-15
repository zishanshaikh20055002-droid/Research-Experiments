# Offline Baseline Methodology

## Scope and Fixed Data

This is an offline evaluation of two deterministic RUL references. It does not
train or alter M1-M5. The source is the processed metadata
`data/processed/xjtu_features_metadata.csv.gz`; the fixed neural comparison
summaries and per-seed prediction files are read-only inputs.

Training fit bearings are Bearing1_1, Bearing1_2, Bearing2_1, Bearing3_1, and
Bearing3_2. Bearing2_2 is validation and is not used to fit either baseline.
The fixed test bearings are Bearing1_3, Bearing1_4, Bearing1_5, Bearing2_3,
Bearing2_4, Bearing2_5, Bearing3_3, Bearing3_4, and Bearing3_5.

For each bearing, sequence targets are constructed from metadata exactly as the
neural pipeline does: sequence length 10, stride 1, end positions 9 through
`N-1`, and target equal to the metadata RUL at the sequence end. No window
crosses a bearing boundary. This gives 5,764 training sequence targets and
3,164 test sequence targets. Every test key
`(bearing_id, start_position, end_position)` and target was verified against
all 15 saved M1-M5 prediction files (five models, three seeds); row counts and
keys matched exactly, and labels matched within `1e-7` to account for float32
serialization in the neural outputs.

The processed RUL labels are the existing retrospective benchmark targets,
which were normalized using each bearing's observed full trajectory length.
That label construction is used only as the evaluation target; the age
predictor receives the observed end position and never receives the test
bearing's total length or endpoint.

## Baseline A: Training-Mean Predictor

The fitted scalar is the arithmetic mean of RUL values at the 5,764 training
sequence endpoints:

`mu_train = mean(y_train_sequence_targets)`.

Every test sequence receives `yhat = mu_train`. The validation bearing, all test
labels, test trajectory lengths, and test features do not enter the estimate.

## Baseline B: Piecewise-Linear Age Predictor

The predictor uses only the observable chronological sequence end position `x`
and maps it to normalized RUL. It does not use test trajectory length, final
recording index, future observations, fault label, vibration features, or
validation/test RUL labels.

The fitting procedure is fixed as follows:

1. Build the 5,764 training sequence endpoint pairs `(end_position, RUL)` from
   the five training fit bearings.
2. Assign each endpoint weight `1 / n_b`, where `n_b` is the number of sequence
   targets from its bearing. Thus, each training bearing has equal total weight.
3. Place four knots at the weighted empirical quantiles 0, 1/3, 2/3, and 1 of
   the training end-position distribution. Knot placement uses training
   positions only; there is no validation-based choice of knots, complexity, or
   hyperparameters.
4. Fit four knot values by weighted least squares using the piecewise-linear
   interpolation basis between adjacent knots. The objective is
   `sum_i w_i (f(x_i) - y_i)^2`. No monotonicity constraint or regularization is
   applied; the fitted relationship is determined solely by the fixed basis
   and weighted training targets.
5. At prediction time, clamp an end position below/above training support to
   the nearest endpoint knot, giving constant endpoint extrapolation. Clip the
   resulting prediction to `[0,1]`.

This is a time-index baseline, not a signal-based estimator. It tests whether
the training trajectories' age-to-label relationship transfers to held-out
bearings without using a held-out bearing's future duration.

## Metrics and Aggregation

For paired test labels `y` and predictions `yhat`:

- `MAE = mean(|yhat - y|)`.
- `RMSE = sqrt(mean((yhat - y)^2))`.
- `R-squared = 1 - sum((yhat-y)^2) / sum((y-mean(y))^2)`.

Pooled window-weighted metrics give every test sequence equal weight. They are
not bearing-level independent-sample estimates: adjacent windows overlap, and
the longer trajectories contribute more rows. Equal-bearing macro MAE and RMSE
are calculated by first calculating each metric separately for each of the nine
test bearings and then taking the unweighted mean of those nine values. The
same distinction is retained for the M1-M5 comparison: pooled neural metrics
are the existing three-seed means, while macro-bearing neural values are
calculated per seed and then summarized across seeds using the mean and sample
standard deviation.

Condition metrics are pooled over the test sequences belonging to each
operating condition. Their neural comparison is the three-seed mean and sample
standard deviation of the existing per-condition metrics.

## Reproduction

From the repository root, run:

`python research/xjtu_physics_mtl/results/baselines/evaluate_baselines.py`

The script asserts exact test-row alignment against all saved neural prediction
files before writing the baseline metric tables and comparison report in this
directory.
