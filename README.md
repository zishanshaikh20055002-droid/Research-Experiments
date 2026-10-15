# Research-Experiments

## Evaluating PINN-Inspired Monotonicity Constraints for Multi-Task Bearing RUL Prediction on XJTU-SY

**Authors:** Zishan Shaikh and Minal Jungare. [Read the approved seven-page paper](manuscript/IEEE_XJTU_RUL_publication_polish_2026-10-15.pdf).

This repository is a reproducibility record for an exploratory empirical comparison of five BiLSTM bearing remaining-useful-life (RUL) configurations, two simple training-only references, six matched-batching controls, and offline temporal analyses. The XJTU-SY fixed test set had already been inspected in earlier project work, so these results are not an untouched confirmation. The five original neural configurations had negative mean pooled test R-squared. The unconstrained piecewise-linear age reference had the lowest pooled MAE/RMSE, while the historical neural ranking changed under matched batch ordering. The local monotonicity penalty did not establish a causal improvement or a governing physical law.

### Repository contents

| Path | Contents |
|---|---|
| `src/`, `scripts/`, `analysis/` | Feature extraction, model definitions, training, evaluation, and offline analysis code |
| `configs/`, `environment/` | Saved settings, seeds, and historical environment records |
| `results/metrics/`, `results/predictions/` | Aggregate/per-seed/per-bearing metrics and saved raw/projected predictions |
| `results/checkpoints/`, `results/logs/`, `results/plots/` | Completed-run checkpoints, histories/logs, and figures |
| `results/long_lag/`, `results/bootstrap/`, `results/isotonic/` | Offline temporal and paired whole-bearing bootstrap analyses |
| `manuscript/` | Final author-approved PDF; the editable DOCX was not in the verified release archive |
| `MANIFEST.csv` | SHA-256 and size for the published files, excluding the manifest itself |
| `docs/ARCHIVE_MANIFEST.csv` | Untouched checksum manifest from the approved v3 ZIP |

`results/metrics/exploratory_lobo/` and corresponding prediction files are a separate post-hoc classical baseline study, **not** the paper's fixed-split confirmatory test. Historical original M1-M5 outputs, matched-batching controls, and post-hoc projection outputs are kept separate. No raw XJTU-SY CSVs or complete derived feature matrix are redistributed here.

### Data and citation

Download XJTU-SY from the [dataset creator's official page](https://biaowang.tech/xjtu-sy-bearing-datasets/) and follow its access conditions. The creator requests citation of B. Wang, Y. Lei, N. Li, and N. Li, “A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings,” *IEEE Transactions on Reliability*, 69(1), 401–412, 2020, [doi:10.1109/TR.2018.2882682](https://doi.org/10.1109/TR.2018.2882682). Public availability and permission to use the dataset do not by themselves establish permission to redistribute raw CSVs; none are included.

The experiment uses 15 bearing trajectories, 9,215 valid recordings, 32 vibration features, numeric file ordering, and 10-step within-bearing windows. Bearing3_4/1105.csv is absent in the source data and was not fabricated. The six candidate training-pool bearings were Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1, and Bearing3_2. The selected fit used all except Bearing2_2, which served as validation; the nine remaining bearings were test. Standardization was fitted on the five training-fit bearings only. Historical and control seeds were 42, 123, and 2024.

### Environment and reproduction

Use a **new scratch copy**, not this archived-results checkout: historical training scripts write into `results/` and must not overwrite the released artifacts. Place that copy under a `research/xjtu_physics_mtl/` path as in the original project, obtain the authorized raw dataset, and set `dataset.root` in `configs/experiment.yaml` to its location. The package records historical Python dependency versions in `environment/`. PyTorch must be installed separately using a compatible build from its official instructions; no package installation occurs automatically.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r environment/requirements_baseline.txt
# Install a compatible PyTorch build separately, then set dataset.root.
.venv\Scripts\python.exe scripts/extract_features.py --config configs/experiment.yaml
.venv\Scripts\python.exe scripts/run_multiseed.py
.venv\Scripts\python.exe scripts/run_m4_multiseed.py
.venv\Scripts\python.exe scripts/aggregate_multiseed.py
```

Those are the original entry points, not a claim that one environment reproduces every historical host exactly. Analysis scripts may retain original directory-layout assumptions; inspect their saved configurations and run manifests before rerunning. The published CSVs and checkpoints are the record of work actually completed. Do not use test labels for hyperparameter selection.

### Metric definitions and limitations

Pooled MAE/RMSE/R-squared give each test window equal weight. Equal-bearing macro MAE/RMSE calculate a score per test bearing and average the nine bearings equally. Overlapping windows are not independent bearing trajectories. Strict temporal violation rate is the fraction of within-bearing chronologically adjacent saved predictions that rise; mean positive violation magnitude averages only those rises. The constant training-mean predictor's zero violations are degenerate. The fitted piecewise-linear age reference is **unconstrained** and has a non-monotonic segment.

Historical M1/M2/M4 training loaders shuffled, while M3/M5 did not. The six unshuffled matched controls were exploratory because the test set was already inspected. Only penalty weight 0.1 was completed; the later six-run lambda sweep completed **0/6** runs. No governing-equation residual was implemented. There were nine independent test bearings, one fixed validation bearing, no Cage-class bearing in the selected training fit, and no statistical-significance claim. Isotonic projection is post-processing, not a trained physical law.

No repository-wide license has been granted. The authors must choose terms for original code and derived results; third-party dataset rights remain with their owners.
