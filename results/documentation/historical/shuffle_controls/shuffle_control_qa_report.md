# Matched-Batching Control QA Report

## Phase 1: implementation audit

The final three-seed runs are implemented by `scripts/run_multiseed.py`; `scripts/run_m4_multiseed.py` imports and calls the same `run_multi` function. Its training loader sets `shuffle=True` for M2/M4 and `False` for M3/M5. Validation and test loaders use `shuffle=False`.

| Setting | M2 / M2-unshuffled | M3 | M4 / M4-unshuffled | M5 |
|---|---|---|---|---|
| Architecture | Shared one-layer BiLSTM, hidden size 64 per direction; 128-D output; 64-unit ReLU RUL and four-class fault heads | Same | Same | Same |
| Objective | MSE + cross-entropy | MSE + cross-entropy + 0.1 monotonicity penalty | Adaptive uncertainty-style RUL/fault sum | Adaptive sum + 0.1 monotonicity penalty |
| Optimizer | Adam, lr 0.001 | Same | Same | Same |
| Batch size | 32 | 32 | 32 | 32 |
| Stopping | Max 40 epochs, patience 8, validation total loss, gradient clip 1.0 | Same | Same | Same |
| Training loader | Shuffled originally; control is unshuffled | Unshuffled | Shuffled originally; control is unshuffled | Unshuffled |
| Validation/test loaders | Unshuffled | Unshuffled | Unshuffled | Unshuffled |

All four use sequence length 10, stride 1, final-step RUL targets, seed values 42/123/2024, the selected fold (train: Bearing1_1, Bearing1_2, Bearing2_1, Bearing3_1, Bearing3_2; validation: Bearing2_2; test: the same nine fixed test bearings), and a StandardScaler fit on training bearings only. The split, model, optimizer, loss formulas, stopping logic, and evaluation implementation are shared in the final runner. The only added control change is forcing the training loader shuffle request to false.

The intended objective differences in the matched comparisons remain: M3 adds its minibatch-local monotonicity term to the M2 summed objective; M5 adds that term to M4's adaptive objective. No other unexplained configuration difference was found in these entry points.

## Phase 2: execution status

**No training run started.** The system Python failed to import PyTorch with Windows Application Control `WinError 4551` while loading `torch\\lib\\c10.dll`. The bundled Codex Python runtime does not contain PyTorch. Installing another build or attempting to bypass host application control would exceed this task's safe execution boundary, so the six requested runs were not approximated or silently substituted.

The prepared `run_shuffle_controls.py` reuses the audited final training function and overrides only a requested training `shuffle=True` to `False`; it writes each run to a separate output directory. It was not successfully executed. No checkpoints, predictions, histories, or new metrics were produced.

## Phase 3: comparison status

Not performed because the six control runs did not execute. Consequently, `shuffle_control_aggregate.csv`, `shuffle_control_per_condition.csv`, and `shuffle_control_comparison.md` are intentionally absent; no results are fabricated. The existing M1–M5 artifacts remain unchanged. This comparison, if later run, must be described as exploratory matched-batching sensitivity analysis because the test set has already been inspected.

## Files created

- `results/shuffle_controls/run_shuffle_controls.py`
- `results/shuffle_controls/shuffle_control_run_manifest.json`
- `results/shuffle_controls/shuffle_control_qa_report.md`

Only new files under the requested output directory were created. No original result, checkpoint, prediction, training script, configuration, manuscript, or production file was modified.
