# Final Offline Diagnostics Audit

Offline analysis only. No training, model changes, prediction changes, manuscript edits, or existing-result overwrites were performed. Baselines were reconstructed from training labels and checked against their saved pooled metrics.

## A. Temporal reference diagnostics

Predictions are sorted by `end_position` within each bearing. A transition is the difference between consecutive saved predictions from that same bearing only. Violation rate is total positive adjacent changes divided by all valid adjacent transitions; mean violation magnitude averages positive changes only. Overall aggregation matches the neural temporal diagnostic: pooled transition weighting for violation rate and violation magnitude, arithmetic mean of per-bearing Spearman values (excluding undefined values), and arithmetic mean of per-bearing mean absolute changes. Condition rows apply the same rules within each condition.

| Reference | Scope | Violation rate | Mean positive violation | Spearman rho | Mean abs. adjacent change | Transitions (violating/valid) |
|---|---|---:|---:|---:|---:|---:|
| training_mean | ALL | 0.000000 | 0.00000000 | N/A | 0.00000000 | 0/3155 |
| training_mean | 35Hz12kN | 0.000000 | 0.00000000 | N/A | 0.00000000 | 0/302 |
| training_mean | 37.5Hz11kN | 0.000000 | 0.00000000 | N/A | 0.00000000 | 0/884 |
| training_mean | 40Hz10kN | 0.000000 | 0.00000000 | N/A | 0.00000000 | 0/1969 |
| piecewise_linear_age | ALL | 0.426307 | 0.00074835 | -0.551447 | 0.00400566 | 1345/3155 |
| piecewise_linear_age | 35Hz12kN | 0.258278 | 0.00074835 | -0.946830 | 0.00549377 | 78/302 |
| piecewise_linear_age | 37.5Hz11kN | 0.691176 | 0.00074835 | -0.228851 | 0.00356583 | 611/884 |
| piecewise_linear_age | 40Hz10kN | 0.333164 | 0.00074835 | -0.478661 | 0.00295738 | 656/1969 |

The training-mean reference is constant: its zero violation rate, zero violation magnitude, and zero adjacent change are degenerate consequences of a flat prediction, not evidence of degradation tracking. Spearman is undefined for a constant series and is reported as N/A. The piecewise-linear age reference is unconstrained: its training-fit knot values are `0.946780, 0.344974, 0.624107, 0.088323` at training-derived end-position knots `9.0, 100.0, 473.0, 2537.0`; the second-to-third knot segment increases with age. Lower violation/smoothness metrics alone indicate neither better RUL accuracy nor physical validity.

Per-bearing and per-condition rows for both references are in `baseline_temporal_diagnostics.csv`. The two deterministic predictions use only training sequence targets for the training mean and training sequence end positions/RUL for the piecewise fit. Test labels are only used to verify alignment and existing pooled metrics, never to fit either reference.

## B. Majority-class fault reference

The training-only bearing counts are `{"Outer": 3, "Inner": 1, "Cage": 0, "Mixed": 1}`; the majority class is **Outer**. Fault labels are constant bearing-level annotations, not timestamp-varying onset labels.

Primary evaluation is at the bearing level (nine bearing-level test units): accuracy 0.444444, macro precision 0.111111, macro recall 0.250000, macro-F1 0.153846. Confusion matrix (rows=true, columns=predicted; order Outer, Inner, Cage, Mixed): `[[4, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]]`.

For comparability with the paper's sequence-row classification calculations, repeated-label window-weighted metrics are also recorded: accuracy 0.195006, macro precision 0.048752, macro recall 0.250000, macro-F1 0.081592. This 3,164-row view repeats each bearing label across overlapping windows and must not be interpreted as 3,164 independent bearing examples. Full per-class values and both confusion matrices are in `majority_class_reference.csv`.

Because Cage has no training bearing in the selected fit subset, this majority reference does not resolve absent Cage supervision and cannot validate four-class diagnostic capability.

## C. Final training-loader audit

Final three-seed entry points inspected: `scripts/run_multiseed.py` (M1, M2, M3, M5) and `scripts/run_m4_multiseed.py` (which invokes `run_multi("m4", ...)`).

| Model | Training batches shuffled? | Validation shuffled? | Test shuffled? |
|---|---|---|---|
| M1 | Yes | No | No |
| M2 | Yes | No | No |
| M3 | No | No | No |
| M4 | Yes | No | No |
| M5 | No | No | No |

The final M3/M5 monotonicity loss is computed independently per minibatch. It groups predictions by bearing identity, sorts each group by sequence end position, and penalizes positive increases between neighboring available predictions. No pair crosses bearing IDs. The unshuffled training sequence is in bearing/chronological order; a pair split at a minibatch cut is absent from that minibatch's loss. With batch size 32, there are **180 batch cuts, of which 180 cut a within-bearing chronological adjacent pair per epoch** for the fixed training sequence order. The omitted transitions recur at those same cuts each epoch. This count is the maximum direct adjacent pairs omitted at the boundaries (and no transition is instead compared across a boundary).

The corrected manuscript's extracted text contains the sentence `Final loaders are unshuffled`, immediately in the M3/M5 loss discussion. It is correct for the M3/M5 training loaders and all validation/test loaders, but is overbroad if read as describing all final training loaders: M1, M2, and M4 training loaders are shuffled. Manuscript was not modified in this task.

## Checks performed

- Reconstructed test set: 3164 rows across 9 test bearings.
- Exact keys and true RUL alignment cross-checked against all 15 saved model/seed prediction CSVs (47,460 prediction rows inspected); tolerance for saved label comparison: 1e-7.
- Reconstructed baseline pooled MAE, RMSE, and R-squared agree with `results/baselines/baseline_metrics.csv` within 2e-9.
- Fault labels agree with the authoritative `src/labels.py` mapping and remain constant within each bearing.
- Training sequence rows used for batch-cut audit: 5764; batch size: 32.
- Existing outputs and manuscript were preserved; the only generated outputs are this audit, its two CSVs, and the reproducibility script in this new directory.
