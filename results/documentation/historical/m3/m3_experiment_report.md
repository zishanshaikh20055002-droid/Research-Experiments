# M3 Experiment Report

- Loss: `MSE + CE + lambda_mono*L_mono`
- Lambda mono: `0.1`
- Best epoch: `7`
- Training time seconds: `95.33819289999974`
- RUL MAE/RMSE/R2: `0.27160226624775013` / `0.3436060817018794` / `-0.47237748872014396`
- Fault macro-F1: `0.06890648796832165`
- Mono violation rate: `0.4846275752773376`
- Mean violation magnitude: `0.015689365509739102`

Known limitation: the adopted split has no Cage-class bearing in the training subset, so Cage classification is not learnable in the conventional supervised sense.
