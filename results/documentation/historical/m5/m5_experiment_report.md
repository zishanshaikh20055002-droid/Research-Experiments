# M5 Experiment Report

- Loss: `exp(-s_R)*MSE+s_R + exp(-s_F)*CE+s_F + lambda_mono*L_mono`
- Lambda mono: `0.1`
- Best epoch: `11`
- Training time seconds: `139.1258858000001`
- RUL MAE/RMSE/R2: `0.2881490593696559` / `0.3609679467267377` / `-0.624930396742098`
- Fault macro-F1: `0.07105328934213158`
- Mono violation rate: `0.49413629160063394`
- Mean violation magnitude: `0.01577640248067089`

Known limitation: the adopted split has no Cage-class bearing in the training subset, so Cage classification is not learnable in the conventional supervised sense.
