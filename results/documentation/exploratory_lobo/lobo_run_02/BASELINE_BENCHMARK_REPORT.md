# Baseline Benchmark v1: Exploratory LOBO Results

## Scope

This is a leakage-free leave-one-bearing-out (LOBO) exploratory baseline benchmark on the frozen 32-feature processed dataset. It does not train a neural or physics-guided model. It must not be numerically compared as if it shared the historical fixed nine-bearing test split.

## Protocol

For each of 15 folds, all windows of one bearing are held out and all windows from the other 14 bearings are used to fit the baseline. Windows are ten chronological recordings, stride one, with the final recording RUL as the target. Ridge uses a fold-local StandardScaler. The recording-index diagnostic receives only the observable original recording index at the window end; it receives neither total bearing length nor a normalized age.

## Pooled and Equal-Bearing Metrics

Pooled metrics give each window equal weight. Macro metrics average the 15 per-bearing metrics and therefore give each bearing equal weight. Window overlap means these are not 9,080 independent bearing trajectories.

| model                             | n_predictions | mae      | rmse     | r2        | macro_bearing_mae | macro_bearing_rmse |
| --------------------------------- | ------------- | -------- | -------- | --------- | ----------------- | ------------------ |
| recording_index_ridge_diagnostic  | 9080          | 0.182820 | 0.230670 | 0.346704  | 0.256866          | 0.312130           |
| extra_trees_flattened_window      | 9080          | 0.204192 | 0.252092 | 0.219724  | 0.194085          | 0.230569           |
| hist_gradient_boosting_final_step | 9080          | 0.205926 | 0.253315 | 0.212140  | 0.193348          | 0.231259           |
| extra_trees_final_step            | 9080          | 0.211993 | 0.254333 | 0.205792  | 0.189987          | 0.226247           |
| random_forest_final_step          | 9080          | 0.216595 | 0.267268 | 0.122953  | 0.227848          | 0.270433           |
| training_mean                     | 9080          | 0.247055 | 0.285419 | -0.000215 | 0.238492          | 0.276287           |
| ridge_final_step                  | 9080          | 0.317620 | 0.392672 | -0.893167 | 0.294722          | 0.339530           |

## Interpretation Rules

- Vibration-feature models are considered descriptively useful only if they improve both pooled and macro MAE over the training-mean and recording-index-only references; this is not a significance test.
- A competitive recording-index-only diagnostic would indicate that sampling-progress information can be predictive under this protocol; it would not demonstrate deployable RUL performance because real deployment may not expose comparable observation histories.
- Physics-guided neural work is not justified or rejected by this one baseline screen alone. Any next experiment should be preregistered against an untouched bearing-level protocol.

## Historical-Result Boundary

Historical fixed-split baseline results, if present, are retained separately at `<PROJECT_ROOT>\results\baselines\baseline_metrics.csv`. They are not pooled with this LOBO output and were not used to choose models here.

## Observed Benchmark Decision

The strongest vibration-feature result on pooled MAE is the flattened-window ExtraTrees model (MAE 0.204192), which improves on the training-mean reference (0.247055) but not on the recording-index-only diagnostic (0.182820). The lowest equal-bearing macro MAE among the vibration-feature models is from final-step ExtraTrees (0.189987), which improves on both the training-mean (0.238492) and recording-index-only (0.256866) references.

Thus, vibration features contain bearing-balanced predictive signal relative to both simple references, but no vibration baseline improves both pooled and macro MAE over the index-only diagnostic. The index-only diagnostic is highly competitive under pooled window weighting, indicating that observed recording progress is an important confound under this protocol. This result does **not** by itself support proceeding to physics-guided neural experiments. The next scientifically useful step is to resolve and predefine how sampling-progress information should be handled, then evaluate any additional model family under a protocol that keeps that comparison explicit.
