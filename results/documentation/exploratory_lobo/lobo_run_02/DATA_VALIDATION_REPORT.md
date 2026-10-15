# Data Validation Report

- Processed rows agree: X=(9215, 32), metadata=9215, rul=9215.
- Feature values and RUL values are finite; NPZ and metadata RUL values agree within float32 representation tolerance (1e-7).
- Built 9080 windows using the NPZ feature matrix and target vector: x[t:t+10] -> RUL[t+9] across 15 bearings.
- All sequence targets are normalized RUL values in [0, 1]; windows never span bearing identities.
- Bearing3_4 retains its original numeric-index gap at 1105; no file is fabricated or renumbered.
- Each of 15 folds holds out exactly one bearing; every held-out bearing appears once and never occurs in its fold's fit set.
- For each model, prediction and target rows use the same held-out sequence row and final-step target.
- Ridge scaling is implemented as a Pipeline, fitted only when estimator.fit receives fold training data. No scaler is fit for trees or constants.
- No validation or historical fixed-test split is used in fitting, feature selection, hyperparameter selection, or model selection.

## Frozen inputs

- Features: `<PROJECT_ROOT>\data\processed\xjtu_features.npz`
- Metadata: `<PROJECT_ROOT>\data\processed\xjtu_features_metadata.csv.gz`
- The benchmark is a new exploratory LOBO protocol and is not directly comparable with the historical fixed nine-bearing test split.
