# Deadline Update Reproducibility Guide

## Important status

The offline isotonic and bootstrap analyses completed. The lambda sweep did not run because no local interpreter can import PyTorch under the current Windows Application Control policy. Do not treat the presence of `run_lambda_sweep.py` as evidence that those experiments completed.

## Inputs

Experiment root:

`<PROJECT_ROOT>`

New output root:

`<PROJECT_ROOT>\deadline_paper_update_2026-10-15`

The exact source locations and compatibility audit are recorded in `PREFLIGHT_REPORT.md`.

## Reproduce the offline analyses

This command uses the existing isolated scientific-Python environment and does not require PyTorch:

```powershell
& 'D:\venvs\xjtu-baselines\Scripts\python.exe' `
  '<PROJECT_ROOT>\deadline_paper_update_2026-10-15\run_offline_posthoc.py'
```

The script refuses to overwrite existing outputs. Use a new copied output directory or archive the current one before a deliberate reproduction.

## Run the blocked lambda sweep later

Use a compatible environment that already provides PyTorch, NumPy, pandas, and scikit-learn. The script installs nothing. It runs in the required paired order: seed 42 lambda 1/10, seed 123 lambda 1/10, then seed 2024 lambda 1/10.

```powershell
<compatible-python> `
  '<PROJECT_ROOT>\deadline_paper_update_2026-10-15\run_lambda_sweep.py' `
  --project-root '<PROJECT_ROOT>' `
  --output '<PROJECT_ROOT>\deadline_paper_update_2026-10-15\lambda_sweep_runs'
```

The output directory must be new or empty. Every run writes its config, training history, checkpoint, test predictions, validation predictions, metrics, and run log. Selection is by lowest mean validation RUL MAE across completed seeds. Early stopping remains based on validation total loss to match historical M3-U.

## Rebuild the manuscript

```powershell
& '<USER_HOME>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  '<PROJECT_ROOT>\deadline_paper_update_2026-10-15\build_deadline_manuscript.py'
```

The generated DOCX is exported with Microsoft Word 2024. LibreOffice was unavailable. The final PDF was rendered to PNG with bundled Poppler at 150 dpi and every page was inspected.

## Core outputs

- `isotonic_metrics_by_seed.csv`: raw and projected metrics per seed.
- `isotonic_metrics_aggregate.csv`: across-seed means and sample SDs.
- `bearing_bootstrap_ci.csv`: point estimates and 95% percentile intervals.
- `bearing_bootstrap_replicates.csv`: all paired bootstrap replicate metrics.
- `projected_predictions/`: separate label-free projected prediction CSVs.
- `analysis_config.json`: paths, seed, and analysis settings.
- `PHM2027_XJTU_RUL_deadline_update_2026-10-15_v5.docx` and matching PDF: final visually checked manuscript.

