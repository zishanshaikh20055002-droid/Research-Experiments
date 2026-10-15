# M1 Baseline Experiment Report

M1 is a single-task BiLSTM RUL regressor. It does not include fault classification, monotonicity loss, or adaptive task weighting.

## Architecture

- Input tensor: `(batch, 10, 32)`
- Encoder: 1-layer bidirectional LSTM, hidden size `64`
- Output head: linear, ReLU, linear, sigmoid
- Output: one normalized RUL scalar in `[0, 1]`
- Loss: `mse`

## Split

- Training pool: `['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']`
- Test bearings: `['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']`
- Validation: leave-one-training-bearing-out over training pool

## Results

- Selected fold: `val_Bearing2_2`
- Best validation loss: `0.009780887783946176`
- Best epoch: `7`
- Epochs completed: `22`
- Training time, selected fold seconds: `55.797156800000266`
- Test MAE: `0.306988443284878`
- Test RMSE: `0.4012790368197732`
- Test R2: `-1.0081235508286865`
- Prediction range: `0.017843589186668396` to `0.9696648120880127`
- Predictions outside [0,1]: `0`

## Warnings

- Bearing3_4: missing numeric CSV indices [1105]; existing files are kept in numeric order.
- CUDA is not available to PyTorch; M1-standardized trained on CPU.
- Existing M1 trainer already used fold-training-only StandardScaler; this run is an explicitly named standardized rerun.
