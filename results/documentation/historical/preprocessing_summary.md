# XJTU-SY Preprocessing Summary

- Generated UTC: `2026-10-08T12:11:12.817054+00:00`
- Processed samples: `9215`
- Feature matrix shape: `[9215, 32]`
- Feature count: `32`

## Feature List

- `h_mean`
- `h_std`
- `h_rms`
- `h_peak_abs`
- `h_peak_to_peak`
- `h_crest_factor`
- `h_kurtosis`
- `h_skewness`
- `h_dominant_freq_hz`
- `h_spectral_centroid_hz`
- `h_spectral_bandwidth_hz`
- `h_spectral_rms`
- `h_band_energy_ratio_0_1000hz`
- `h_band_energy_ratio_1000_3000hz`
- `h_band_energy_ratio_3000_6000hz`
- `h_band_energy_ratio_6000_12800hz`
- `v_mean`
- `v_std`
- `v_rms`
- `v_peak_abs`
- `v_peak_to_peak`
- `v_crest_factor`
- `v_kurtosis`
- `v_skewness`
- `v_dominant_freq_hz`
- `v_spectral_centroid_hz`
- `v_spectral_bandwidth_hz`
- `v_spectral_rms`
- `v_band_energy_ratio_0_1000hz`
- `v_band_energy_ratio_1000_3000hz`
- `v_band_energy_ratio_3000_6000hz`
- `v_band_energy_ratio_6000_12800hz`

## Samples Per Bearing

- Bearing1_1: `123`
- Bearing1_2: `161`
- Bearing1_3: `158`
- Bearing1_4: `122`
- Bearing1_5: `52`
- Bearing2_1: `491`
- Bearing2_2: `161`
- Bearing2_3: `533`
- Bearing2_4: `42`
- Bearing2_5: `339`
- Bearing3_1: `2538`
- Bearing3_2: `2496`
- Bearing3_3: `371`
- Bearing3_4: `1514`
- Bearing3_5: `114`

## Fault Class Distribution

- Cage: `655` samples from `2` bearings
- Inner: `2376` samples from `3` bearings
- Mixed: `2548` samples from `2` bearings
- Outer: `3636` samples from `8` bearings

## RUL Statistics

- min: `0.0`
- max: `1.0`
- mean: `0.5`
- std: `0.28916344589811815`
- median: `0.5`

## Missing Records

- Bearing3_4: `[1105]`

## Validation Checks

- rul_within_0_1: `True`
- fault_labels_match_mapping: `True`
- features_all_finite: `True`
- train_test_bearing_disjoint: `True`
- no_test_bearing_in_training_pool: `True`
- all_validation_folds_leakage_free: `True`
- chronological_ordering_ok: `True`
- sequence_boundary_check: `True`
- sequence_length: `10`
- sequence_stride: `1`
- scaler_fitted: `False`
- normalization_not_fitted_on_test: `True`
- passed: `True`

## Warnings

- Bearing3_4: missing numeric CSV indices [1105]; existing files are kept in numeric order.
