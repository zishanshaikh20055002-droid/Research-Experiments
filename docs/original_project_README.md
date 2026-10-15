# XJTU-SY Physics-Constrained MTL Research

Isolated research workspace for the paper experiment:

"Physics-Constrained Multi-Task Learning for Remaining Useful Life Prediction and Failure-Mode Diagnosis of Rolling Bearings"

This directory is intentionally separate from the PulseGuard production application. Raw XJTU-SY files remain under `data/` and must be treated as read-only.

## Current Stage

Dataset inspection and feature preprocessing are complete. No neural-network model
training has been run yet.

## Planned Leakage-Free Split

The experiment will preserve bearing-level separation using the XJTU-SY split convention:

| Condition | Operating condition | Train bearings | Test bearings |
| --- | --- | --- | --- |
| A | 35Hz12kN | Bearing1_1, Bearing1_2 | Bearing1_3, Bearing1_4, Bearing1_5 |
| B | 37.5Hz11kN | Bearing2_1, Bearing2_2 | Bearing2_3, Bearing2_4, Bearing2_5 |
| C | 40Hz10kN | Bearing3_1, Bearing3_2 | Bearing3_3, Bearing3_4, Bearing3_5 |

Validation should be bearing-level and drawn only from the training bearings. Because each condition has only two training bearings, the default proposed strategy is leave-one-training-bearing-out validation during model selection, without changing the fixed test bearings.

## Inspection

Run from the repository root:

```powershell
python research\xjtu_physics_mtl\scripts\inspect_dataset.py
```

Outputs are written to:

- `research/xjtu_physics_mtl/results/dataset_inspection_summary.md`
- `research/xjtu_physics_mtl/results/dataset_inspection_summary.json`

## Preprocessing

Run from the repository root:

```powershell
python research\xjtu_physics_mtl\scripts\extract_features.py --config research\xjtu_physics_mtl\configs\experiment.yaml
```

The preprocessing stage reads the raw XJTU-SY CSVs in place, validates the two
vibration channels, and writes only compact derived artifacts under
`research/xjtu_physics_mtl/data/processed/`. Raw CSV files are never copied,
renamed, repaired, or overwritten.

Recordings are sorted numerically by filename inside each bearing folder. This
is required because lexicographic ordering would place names such as `10.csv`
before `2.csv`. RUL is generated independently per bearing as
`(N - 1 - i) / (N - 1)`, where `i` is the zero-based chronological position
among valid existing recordings. `Bearing3_4/1105.csv` is missing in the raw
dataset; preprocessing records this in metadata, keeps existing files in numeric
order, and does not fabricate or renumber any recording.

For each horizontal and vertical channel, preprocessing extracts:

- mean
- standard deviation
- RMS
- peak absolute amplitude
- peak-to-peak amplitude
- crest factor
- kurtosis
- skewness
- dominant frequency
- spectral centroid
- spectral bandwidth
- spectral RMS
- band energy ratio, 0-1000 Hz
- band energy ratio, 1000-3000 Hz
- band energy ratio, 3000-6000 Hz
- band energy ratio, 6000-12800 Hz

This gives 32 total features per recording. The processed dataset is saved as:

- `data/processed/xjtu_features.npz`: compact feature matrix, RUL targets, fault
  labels, feature names, bearing IDs, chronological positions, and recording
  indices.
- `data/processed/xjtu_features_metadata.csv.gz`: tabular metadata plus feature
  values for auditability.
- `data/processed/preprocessing_metadata.json`: split, label, feature, missing
  record, and normalization policy metadata.
- `data/processed/preprocessing_checks.json`: validation checks.

No normalization scaler is fitted during preprocessing. Scaling must be fitted
inside each validation fold using training bearings only, then applied to that
fold's validation bearing and the fixed test bearings. The nine test bearings
must not be used for feature selection, normalization statistics, early stopping,
threshold selection, hyperparameter selection, or model selection.
