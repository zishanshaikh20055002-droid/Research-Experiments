# M1 Diagnostic Report

This audit uses the existing M1 artifacts only. It does not modify M1, retrain M1, or implement M2.

## Sequence Alignment

- Target convention: `features[t:t+10] are paired with RUL at final timestep t+9`
- Chronological ordering OK: `True`
- Bearing boundary handling OK: `True`
- Sequence examples saved: `m1_sequence_alignment_examples.csv`

## Split Findings

- Selected fold: `val_Bearing2_2`
- Train bearings: `['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing3_1', 'Bearing3_2']`
- Validation bearing: `['Bearing2_2']`
- Test bearings: `['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']`
- Splits disjoint: `True`
- Test used for scaling/model selection/early stopping: `False`

## Feature Distribution Shift

Top train-test standardized mean shifts:

| feature | abs_standardized_train_test_mean_shift | train_mean | test_mean | train_std | test_std |
| --- | --- | --- | --- | --- | --- |
| h_band_energy_ratio_6000_12800hz | 2.00487 | 0.469533 | 0.244523 | 0.123784 | 0.0993449 |
| h_spectral_centroid_hz | 1.6268 | 5377.58 | 3531.09 | 1324.18 | 907.307 |
| v_band_energy_ratio_0_1000hz | 1.24742 | 0.350375 | 0.541092 | 0.153915 | 0.151857 |
| h_band_energy_ratio_1000_3000hz | 1.20315 | 0.250508 | 0.368127 | 0.0984582 | 0.0970561 |
| h_spectral_bandwidth_hz | 1.17253 | 3723.29 | 3050.38 | 640.898 | 497.962 |
| v_spectral_centroid_hz | 1.05172 | 3551.44 | 2559.22 | 1080.98 | 782.043 |
| v_skewness | 0.906396 | -0.088924 | 0.0104849 | 0.123884 | 0.0933265 |
| v_band_energy_ratio_6000_12800hz | 0.835811 | 0.207806 | 0.135178 | 0.0999829 | 0.0714489 |
| v_band_energy_ratio_3000_6000hz | 0.831425 | 0.189918 | 0.117958 | 0.095145 | 0.0770034 |
| v_spectral_bandwidth_hz | 0.646799 | 3537.93 | 3094.99 | 768.066 | 589.965 |

## RUL Distributions

| split | sequence_count | mean_rul | std_rul | min_rul | max_rul |
| --- | --- | --- | --- | --- | --- |
| train | 5764 | 0.496204 | 0.286847 | 0 | 0.996453 |
| validation | 152 | 0.471875 | 0.275142 | 0 | 0.94375 |
| test | 3164 | 0.488061 | 0.283217 | 0 | 0.994052 |

Per-bearing RUL distribution saved: `m1_rul_distribution_by_bearing.csv`

## Diagnostic Baselines

| baseline | mae | rmse | r2 | prediction_min | prediction_max | predictions_outside_0_1 |
| --- | --- | --- | --- | --- | --- | --- |
| train_mean_rul | 0.245094 | 0.28329 | -0.00082704 | 0.496204 | 0.496204 | 0 |
| sequence_start_rul_persistence_label_diagnostic | 0.0238783 | 0.0411646 | 0.978868 | 0.00594845 | 1 | 0 |

## Per-Bearing M1 Performance

| bearing_id | sequence_count | mae | rmse | r2 | prediction_min | prediction_max | predictions_outside_0_1 | condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bearing1_3 | 149 | 0.151242 | 0.190765 | 0.51513 | 0.0490225 | 0.819848 | 0 | 35Hz12kN |
| Bearing1_4 | 113 | 0.0563587 | 0.0761187 | 0.920272 | 0.171772 | 0.911553 | 0 | 35Hz12kN |
| Bearing1_5 | 43 | 0.0571615 | 0.0693541 | 0.918761 | 0.0275263 | 0.735345 | 0 | 35Hz12kN |
| Bearing2_3 | 524 | 0.171289 | 0.230729 | 0.341515 | 0.0178436 | 0.926265 | 0 | 37.5Hz11kN |
| Bearing2_4 | 33 | 0.206181 | 0.265981 | -0.311659 | 0.0342959 | 0.811869 | 0 | 37.5Hz11kN |
| Bearing2_5 | 330 | 0.127352 | 0.159522 | 0.679644 | 0.0179729 | 0.910858 | 0 | 37.5Hz11kN |
| Bearing3_3 | 362 | 0.397236 | 0.482329 | -1.91648 | 0.031284 | 0.969665 | 0 | 40Hz10kN |
| Bearing3_4 | 1505 | 0.4117 | 0.490647 | -1.91961 | 0.0578879 | 0.960861 | 0 | 40Hz10kN |
| Bearing3_5 | 105 | 0.361481 | 0.4146 | -1.38923 | 0.0199292 | 0.629306 | 0 | 40Hz10kN |

## Per-Condition M1 Performance

| condition | sequence_count | mae | rmse | r2 | prediction_min | prediction_max | predictions_outside_0_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 35Hz12kN | 305 | 0.102825 | 0.143537 | 0.71516 | 0.0275263 | 0.911553 | 0 |
| 37.5Hz11kN | 887 | 0.156241 | 0.208683 | 0.453448 | 0.0178436 | 0.926265 | 0 |
| 40Hz10kN | 1972 | 0.406371 | 0.48537 | -1.89144 | 0.0199292 | 0.969665 | 0 |

## Prediction Bias

- mean_true_rul: `0.4880608739733438`
- mean_predicted_rul: `0.6803827021273705`
- mean_prediction_error_pred_minus_true: `0.19232182815402657`
- prediction_std: `0.35423187363809944`
- error_std: `0.3522447938047852`

## Sigmoid/Saturation Analysis

- prediction_min: `0.01784359`
- prediction_max: `0.9696648`
- prediction_mean: `0.6803827021273705`
- prediction_std: `0.35423187363809944`
- fraction_below_0_05: `0.06479140328697851`
- fraction_above_0_95: `0.11946902654867257`
- fraction_below_0_01: `0.0`
- fraction_above_0_99: `0.0`
- interpretation: `No hard sigmoid saturation if only a small fraction is near 0 or 1.`
- quantiles: `{'0%': 0.01784359, '1%': 0.021863506629999998, '5%': 0.041877271, '25%': 0.32765557749999996, '50%': 0.903351135, '75%': 0.9436235475, '95%': 0.9552632005, '99%': 0.963072069, '100%': 0.9696648}`

## Technical Interpretation

The sequence-label alignment is consistent with the intended final-timestep target. The selected M1 fold is leakage-free and its scaler was fit on fold-training bearings only. The weak aggregate test score appears more consistent with cross-bearing/domain shift and bearing-level generalization difficulty than with a simple indexing bug. The largest feature shift is `h_band_energy_ratio_6000_12800hz` with standardized train-test mean shift 2.005. Performance is uneven by bearing; the highest RMSE is on `Bearing3_4`. The prediction range and near-boundary fractions do not indicate severe sigmoid saturation, although the sigmoid does constrain outputs inside [0,1]. The sequence-start RUL persistence diagnostic is label-derived and not a valid deployable model, but it confirms that much of the target is chronological and that M1 is not yet learning a robust cross-bearing degradation mapping.
