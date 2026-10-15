# XJTU-SY Dataset Inspection Summary

- Generated UTC: `2026-10-08T11:56:32.921326+00:00`
- Repository root: `<REPOSITORY_ROOT>`
- Dataset path: `<REPOSITORY_ROOT>\data\XJTU-SY_Bearing_Datasets`
- Total recordings: `9215`
- Approximate raw CSV size: `11.379 GB`

## Bearings

| Bearing | Condition | Recordings | Row counts | Column signatures | Missing files |
| --- | --- | ---: | --- | --- | --- |
| Bearing1_1 | 35Hz12kN | 123 | 32768: 123 | Horizontal_vibration_signals \| Vertical_vibration_signals: 123 | None |
| Bearing1_2 | 35Hz12kN | 161 | 32768: 161 | Horizontal_vibration_signals \| Vertical_vibration_signals: 161 | None |
| Bearing1_3 | 35Hz12kN | 158 | 32768: 158 | Horizontal_vibration_signals \| Vertical_vibration_signals: 158 | None |
| Bearing1_4 | 35Hz12kN | 122 | 32768: 122 | Horizontal_vibration_signals \| Vertical_vibration_signals: 122 | None |
| Bearing1_5 | 35Hz12kN | 52 | 32768: 52 | Horizontal_vibration_signals \| Vertical_vibration_signals: 52 | None |
| Bearing2_1 | 37.5Hz11kN | 491 | 32768: 491 | Horizontal_vibration_signals \| Vertical_vibration_signals: 491 | None |
| Bearing2_2 | 37.5Hz11kN | 161 | 32768: 161 | Horizontal_vibration_signals \| Vertical_vibration_signals: 161 | None |
| Bearing2_3 | 37.5Hz11kN | 533 | 32768: 533 | Horizontal_vibration_signals \| Vertical_vibration_signals: 533 | None |
| Bearing2_4 | 37.5Hz11kN | 42 | 32768: 42 | Horizontal_vibration_signals \| Vertical_vibration_signals: 42 | None |
| Bearing2_5 | 37.5Hz11kN | 339 | 32768: 339 | Horizontal_vibration_signals \| Vertical_vibration_signals: 339 | None |
| Bearing3_1 | 40Hz10kN | 2538 | 32768: 2538 | Horizontal_vibration_signals \| Vertical_vibration_signals: 2538 | None |
| Bearing3_2 | 40Hz10kN | 2496 | 32768: 2496 | Horizontal_vibration_signals \| Vertical_vibration_signals: 2496 | None |
| Bearing3_3 | 40Hz10kN | 371 | 32768: 371 | Horizontal_vibration_signals \| Vertical_vibration_signals: 371 | None |
| Bearing3_4 | 40Hz10kN | 1514 | 32768: 1514 | Horizontal_vibration_signals \| Vertical_vibration_signals: 1514 | 1105 |
| Bearing3_5 | 40Hz10kN | 114 | 32768: 114 | Horizontal_vibration_signals \| Vertical_vibration_signals: 114 | None |

## Data Quality

- missing_bearing_folders: `0`
- unexpected_bearing_folders: `0`
- missing_files: `1`
- malformed_csvs: `0`
- shape_problems: `0`
- column_problems: `0`
- value_problems: `0`
- duplicate_or_suspicious_files: `0`
- filename_ordering_notes: `15`

## Proposed Split

- Train: Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1, Bearing3_2
- Test: Bearing1_3, Bearing1_4, Bearing1_5, Bearing2_3, Bearing2_4, Bearing2_5, Bearing3_3, Bearing3_4, Bearing3_5
- Validation: leave-one-training-bearing-out over the six training bearings; never sample-split within a bearing.

## GPU

- PyTorch available: `False`
- CUDA available via PyTorch: `False`
- CUDA device count: `0`
- nvidia-smi: `NVIDIA GeForce RTX 3050 A Laptop GPU, 4094, 610.62`
- PyTorch detection error: `ModuleNotFoundError("No module named 'torch'")`
