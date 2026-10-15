# M2 Experiment Report

M2 is a clean multi-task baseline with shared BiLSTM encoder, one RUL regression head, and one fault classification head. It does not use monotonicity loss, uncertainty weighting, physics loss, PCGrad, GradNorm, or domain adaptation.

## Results

- Train/validation/test sequences: `5764` / `152` / `3164`
- Training time seconds: `132.16958779999914`
- Test MAE: `0.3137107426063325`
- Test RMSE: `0.3932937344852861`
- Test R2: `-0.9289969468171038`
- Fault accuracy: `0.16182048040455121`
- Fault macro-F1: `0.11245001671283134`
- M2 improves RUL vs M1: `False`

## Comparison With M1

- M1: `{'mae': 0.306988, 'rmse': 0.401279, 'r2': -1.008124}`
- M2: `{'mae': 0.3137107426063325, 'rmse': 0.3932937344852861, 'r2': -0.9289969468171038}`
- Delta M2 - M1: `{'mae': 0.0067227426063325435, 'rmse': -0.007985265514713902, 'r2': 0.07912705318289626}`

## Warnings

- Bearing3_4: missing numeric CSV indices [1105]; existing files are kept in numeric order.
- CUDA is not available to PyTorch; M2 trained on CPU.
