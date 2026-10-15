# Index, Vibration, and Condition Ablation

## Status

**POST-HOC AND EXPLORATORY.** These follow-up comparisons use observed prior results and overlapping LOBO training sets. The uncertainty summaries resample 15 bearings, not windows or seeds; they are descriptive rather than confirmatory.

## Input Audit

All 32 horizontal/vertical engineered features are Class A recording-local, prospectively observable features. Each is calculated only from the current 32,768-sample raw recording using time statistics or its mean-centered rFFT. The extraction code contains no future-record, trajectory-length, terminal-value, global-run, or elapsed-time feature transformation. Full feature-level evidence is in `feature_time_leakage_audit.csv`.

## Pooled and Equal-Bearing Results

| model                    | seed | mae      | rmse     | r2        | macro_bearing_mae | macro_bearing_rmse |
| ------------------------ | ---- | -------- | -------- | --------- | ----------------- | ------------------ |
| A5_index_condition_hgb   | 42   | 0.140256 | 0.19356  | 0.539997  | 0.215565          | 0.245129           |
| A1_index_hgb             | 42   | 0.187412 | 0.241874 | 0.281696  | 0.261848          | 0.312986           |
| B1_index_extra           | 123  | 0.187552 | 0.242195 | 0.279792  | 0.262128          | 0.313123           |
| B1_index_extra           | 2024 | 0.187564 | 0.242202 | 0.27975   | 0.262128          | 0.313123           |
| B1_index_extra           | 42   | 0.187571 | 0.242197 | 0.27978   | 0.262128          | 0.313123           |
| B5_index_window_extra    | 42   | 0.190079 | 0.22455  | 0.38091   | 0.199929          | 0.231791           |
| B5_index_window_extra    | 123  | 0.19043  | 0.226391 | 0.370717  | 0.199929          | 0.231791           |
| B5_index_window_extra    | 2024 | 0.192286 | 0.22793  | 0.362133  | 0.199929          | 0.231791           |
| B4_window_extra          | 2024 | 0.202432 | 0.248332 | 0.242827  | 0.194233          | 0.230042           |
| B4_window_extra          | 42   | 0.204192 | 0.252092 | 0.219724  | 0.194233          | 0.230042           |
| A2_vibration_hgb         | 42   | 0.205926 | 0.253315 | 0.21214   | 0.193348          | 0.231259           |
| B2_vibration_extra       | 123  | 0.206734 | 0.250369 | 0.230355  | 0.189998          | 0.227075           |
| B4_window_extra          | 123  | 0.208659 | 0.256414 | 0.19274   | 0.194233          | 0.230042           |
| B2_vibration_extra       | 2024 | 0.210911 | 0.253468 | 0.211186  | 0.189998          | 0.227075           |
| B2_vibration_extra       | 42   | 0.211993 | 0.254333 | 0.205792  | 0.189998          | 0.227075           |
| B3_index_vibration_extra | 2024 | 0.221743 | 0.271823 | 0.092802  | 0.208293          | 0.244756           |
| B3_index_vibration_extra | 123  | 0.225963 | 0.277214 | 0.056466  | 0.208293          | 0.244756           |
| B3_index_vibration_extra | 42   | 0.228458 | 0.281259 | 0.028723  | 0.208293          | 0.244756           |
| A3_index_vibration_hgb   | 42   | 0.239718 | 0.298547 | -0.094342 | 0.213986          | 0.24908            |
| A4_condition_hgb         | 42   | 0.246969 | 0.285281 | 0.000753  | 0.237946          | 0.275388           |
| training_mean            | 0    | 0.247055 | 0.285419 | -0.000215 | 0.238492          | 0.276287           |

## Historical Reference Boundary

The training-mean row above was recomputed in this run. The historical `lobo_run_02` recording-index Ridge diagnostic used the same frozen target, sequence construction, LOBO folds, and observable final recording-index input: pooled MAE 0.182820, RMSE 0.230670, R2 0.346704; equal-bearing MAE 0.256866 and RMSE 0.312130. It is retained as a read-only reference at `../lobo_run_02/baseline_metrics_by_model.csv`; it was not merged into the seeded tree averages or used to tune any configuration here.

## Paired Bearing Contrasts

A negative difference means the candidate has lower error. Tree seed metrics are averaged within each bearing before forming contrasts. Bootstrap intervals use 10,000 fixed-seed resamples of the 15 bearings. Sign and Wilcoxon tests are exploratory two-sided supplements; zero differences are omitted from both tests.

### A3_minus_A1

- mae: `{"mean_difference": -0.047862306353728824, "median_difference": -0.08885268397219728, "improved": 10, "worsened": 5, "tied": 0, "bootstrap_percentile_95_ci": [-0.10353511217710598, 0.01388099986477091], "sign_test_two_sided_p": 0.30175781249999994, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.207763671875, "wilcoxon_statistic": 37.0, "wilcoxon_nonzero_n": 15}`
- rmse: `{"mean_difference": -0.06390578950530737, "median_difference": -0.11756377878900584, "improved": 10, "worsened": 5, "tied": 0, "bootstrap_percentile_95_ci": [-0.12462930896381307, 0.004640657662916524], "sign_test_two_sided_p": 0.30175781249999994, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.10699462890625, "wilcoxon_statistic": 31.0, "wilcoxon_nonzero_n": 15}`
- Length/extrapolation rank associations: `{"spearman_difference_vs_length": 0.4843612300500202, "spearman_difference_vs_extrapolation_proportion": 0.37115374447904503}`

### B3_minus_B1

- mae: `{"mean_difference": -0.05383499200614425, "median_difference": -0.07400273470218857, "improved": 10, "worsened": 5, "tied": 0, "bootstrap_percentile_95_ci": [-0.10866511426291522, 0.0030114316212406737], "sign_test_two_sided_p": 0.30175781249999994, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.12054443359375, "wilcoxon_statistic": 32.0, "wilcoxon_nonzero_n": 15}`
- rmse: `{"mean_difference": -0.06836737034086286, "median_difference": -0.11236091030308273, "improved": 10, "worsened": 5, "tied": 0, "bootstrap_percentile_95_ci": [-0.12935851035718815, -0.003224589133898891], "sign_test_two_sided_p": 0.30175781249999994, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.05535888671875, "wilcoxon_statistic": 26.0, "wilcoxon_nonzero_n": 15}`
- Length/extrapolation rank associations: `{"spearman_difference_vs_length": 0.5898125679575891, "spearman_difference_vs_extrapolation_proportion": 0.43301270189221924}`

### B5_minus_B3

- mae: `{"mean_difference": -0.008363646777657757, "median_difference": -0.005031736639621531, "improved": 8, "worsened": 7, "tied": 0, "bootstrap_percentile_95_ci": [-0.021392228099639348, 0.0037866569377226346], "sign_test_two_sided_p": 1.0, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.33026123046875, "wilcoxon_statistic": 42.0, "wilcoxon_nonzero_n": 15}`
- rmse: `{"mean_difference": -0.012965174185135821, "median_difference": -0.00738933107822079, "improved": 8, "worsened": 7, "tied": 0, "bootstrap_percentile_95_ci": [-0.02927397302468866, 0.001575096546623669], "sign_test_two_sided_p": 1.0, "sign_test_nonzero_n": 15, "wilcoxon_two_sided_p": 0.27685546875, "wilcoxon_statistic": 40.0, "wilcoxon_nonzero_n": 15}`
- Length/extrapolation rank associations: `{"spearman_difference_vs_length": -0.47363736517806404, "spearman_difference_vs_extrapolation_proportion": -0.37115374447904503}`

## Interpretation

The primary effect is A3 minus A1 equal-bearing MAE; secondary and temporal effects are B3 minus B1 and B5 minus B3. See `paired_bearing_differences.csv` for all 15 differences, ordered by trajectory length. Condition controls are descriptive only and exclude bearing identity. Tree behavior outside the observed training-index range is a plausible diagnostic explanation, not evidence of a causal mechanism.

The normalized-relative target represents remaining fraction of each trajectory, not an absolute deployable lifetime. Because strong prediction from raw recording index may depend on a test setting with comparable observation progress, a separate target/deployment study should precede physics-guided learning. Pooled and equal-bearing summaries must be interpreted separately because longer trajectories contribute many more overlapping windows.

## Direct Answers to the Prespecified Questions

1. **HGB, A3 versus A1:** Adding vibration to index improved equal-bearing MAE from 0.261848 to 0.213986 (mean paired difference -0.047862; 10/15 bearings lower; bootstrap percentile interval [-0.103535, 0.013881]). It worsened pooled MAE from 0.187412 to 0.239718. Thus the answer is yes for this equal-bearing summary, but not for pooled windows; this post-hoc exploratory interval includes zero.
2. **ExtraTrees, B3 versus B1:** Adding vibration improved equal-bearing MAE from 0.262128 to 0.208293 (mean paired difference -0.053835; 10/15 lower; interval [-0.108665, 0.003011]) but worsened pooled MAE from 0.187562 to 0.225388. The same aggregation-dependent pattern appears.
3. **Does index improve vibration-only models?** No for the tested final-step designs. HGB vibration-only had pooled/macro MAE 0.205926/0.193348 versus 0.239718/0.213986 for index plus vibration. ExtraTrees vibration-only averaged 0.209879/0.189998 versus 0.225388/0.208293 after adding index. This is descriptive, not an optimized-model claim.
4. **Does flattened context add value beyond B3?** B5 improved pooled MAE (0.190931 versus 0.225388) and macro MAE (0.199929 versus 0.208293). Its bearing-level MAE difference was -0.008364, with 8/15 lower and interval [-0.021392, 0.003787], so the direction is modest and uncertain under this post-hoc bearing bootstrap.
5. **Length and extrapolation:** Only held-out Bearing3_1 has indices beyond the maximum index available in its training fold (42/2,529 windows; 1.66%). The paired MAE difference versus length has Spearman rho 0.484 for A3-A1 and 0.590 for B3-B1; their positive signs mean the index-plus-vibration advantage tends to diminish as held-out trajectory length rises. With only one nonzero extrapolation proportion, no broad extrapolation association can be inferred.
6. **Condition/progress signal:** Index plus condition (A5) is the best pooled design here (MAE 0.140256, RMSE 0.193560, R2 0.539997), substantially ahead of index-only HGB (0.187412 MAE). This indicates that operating condition and observed recording progress explain a considerable part of the apparent predictive signal under this target/protocol. It does not establish a deployable condition-aware prognostic method.
7. **Pooled versus equal-bearing interpretation:** Yes. Pooled metrics favor index-only trees because long trajectories contribute many overlapping windows. Equal-bearing summaries favor vibration-only and window-vibration designs over index-only designs. Neither aggregation treats overlapping windows as independent bearing trajectories.
8. **Target suitability:** The normalized-relative target is valid for the recorded experimental protocol, but it is not an absolute remaining-life target and its relationship to prospective deployment is unresolved. A separate target and deployment-observability study should precede physics-guided learning.

## Caveats

- The condition inputs are encoded from training-fold categories only. They are controls, not bearing identifiers.
- The exact sign and Wilcoxon calculations are supplements only: ties/zeros are excluded, all 15 comparisons use overlapping LOBO training sets, and no formal significance claim is warranted.
- Per-bearing tables, source index sequences, missing-index information, and each prediction are retained in the accompanying CSV files.
