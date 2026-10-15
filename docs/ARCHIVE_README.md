# XJTU-SY Bearing RUL: exploratory reproducibility release

This package accompanies **Evaluating PINN-Inspired Monotonicity Constraints for Multi-Task Bearing RUL Prediction on XJTU-SY**. It records historical M1-M5 results, matched-batching controls, deterministic references, and offline temporal/post-hoc analyses. The fixed test set was examined in earlier project iterations. These findings are exploratory, not a fresh confirmatory evaluation.

## What is included

- `src/`, `scripts/`, `analysis/`, `configs/`: original project-specific code and saved configuration. Some scripts were built for the original research-directory layout and may need path adaptation in a new checkout; no rerun was performed for this release.
- `results/metrics/`, `results/predictions/`, `results/checkpoints/`, `results/plots/`, `results/logs/`: completed historical and matched-control artifacts, grouped by experiment. The six matched controls are under `matched_batching_controls`; historical M1-M5 runs remain distinct.
- `results/long_lag/`, `results/bootstrap/`, `results/isotonic/`: completed offline analyses. The bootstrap intervals are exploratory and resample whole test bearings.
- `results/metrics/exploratory_lobo/` and related folders: a separate post-hoc leave-one-bearing-out classical baseline analysis, explicitly not a fresh confirmatory test of the paper's fixed split.
- `manuscript/`: the authors' approved final PDF.
- `MANIFEST.csv`: relative path, byte count, and SHA-256 digest of each released file other than the manifest itself.

Raw XJTU-SY CSVs and the derived full processed feature matrix are **not redistributed**. Obtain the dataset from its authorized source and cite the XJTU-SY dataset paper identified in the manuscript. The fixed experiment used 15 bearing trajectories, 9,215 valid recordings, 32 features, 10-step windows, and numeric file ordering. Bearing3_4/1105.csv is missing in the source data; no file was fabricated. Do not treat overlapping windows as independent bearings.

## Protocol

The six candidate training-pool bearings were Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1, Bearing3_2. The final fit used Bearing1_1, Bearing1_2, Bearing2_1, Bearing3_1, Bearing3_2; Bearing2_2 was validation. Test bearings were Bearing1_3, Bearing1_4, Bearing1_5, Bearing2_3, Bearing2_4, Bearing2_5, Bearing3_3, Bearing3_4, Bearing3_5. Standardization used the five training-fit bearings only. Three historical seeds were 42, 123, and 2024. Matched controls used the same seeds and unshuffled training batches.

The historical M1/M2/M4 loaders shuffled training samples, while M3/M5 loaders did not. The comparison between historical arms is therefore not a causal isolation of the penalty. M2-control and M4-control are the completed unshuffled sensitivity runs. The only completed nonzero monotonicity weight was 0.1; the later lambda sweep completed 0 of 6 intended runs. There is no governing-equation residual or established PINN constraint here.

## Environment and commands

Use an isolated Python environment with compatible `numpy`, `pandas`, `scipy`, `scikit-learn`, `PyTorch`, `matplotlib`, and `PyYAML`. Historical resolved versions are in `environment/`; they are records, not a guarantee that one environment reproduces every historical host. The commands below are **not executed by this release**. Run them only in a new scratch checkout with authorized raw data and sufficient compute; historical scripts write into `results/`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r environment/requirements_baseline.txt
# Install a compatible PyTorch build separately from the official PyTorch instructions.
# Edit configs/experiment.yaml: dataset.root must point to your authorized XJTU-SY CSV directory.
.venv\Scripts\python.exe scripts/extract_features.py --config configs/experiment.yaml
.venv\Scripts\python.exe scripts/run_multiseed.py
.venv\Scripts\python.exe scripts/run_m4_multiseed.py
.venv\Scripts\python.exe scripts/aggregate_multiseed.py
.venv\Scripts\python.exe analysis/evaluate_baselines.py
```

These commands describe the original script entry points, but the analysis scripts may expect the original `research/xjtu_physics_mtl` folder nesting. For exact artifact-level comparisons, use the saved predictions and metrics in this package rather than rerunning a new experiment. Do not use test labels to tune or select a model.

## Metric definitions

Pooled MAE/RMSE/R-squared weight every test window equally. Equal-bearing macro MAE/RMSE compute each metric for each of the nine test bearings, then average bearings equally. Strict violation rate counts within-bearing chronological adjacent pairs where predicted RUL rises. Mean positive violation magnitude averages only those positive rises. Spearman correlation is undefined for a constant predictor. The training-mean reference is constant and its zero violations are degenerate. The age reference is unconstrained and has a non-monotonic fitted segment.

## Availability and limitations

This local package is staged for public release, but a public GitHub URL is **not** asserted here. No repository license is included because the authors have not selected one. The raw data's redistribution status was not verified. See `docs/PUBLIC_RELEASE_NOTES.md` and the manuscript for additional scientific limitations and provenance.
