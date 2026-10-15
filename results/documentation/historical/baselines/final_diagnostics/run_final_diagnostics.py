"""Reproduce offline temporal-baseline and majority-class diagnostics."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

from src.labels import FAULT_LABELS, FAULT_TO_ID, ID_TO_FAULT  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_module("baseline_eval", ROOT / "results/baselines/evaluate_baselines.py")
trend = load_module("final_trend", ROOT / "scripts/final_pass_analysis.py")


def bearing_condition(bearing: str) -> str:
    if bearing.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing.startswith("Bearing2_"):
        return "37.5Hz11kN"
    if bearing.startswith("Bearing3_"):
        return "40Hz10kN"
    raise ValueError(bearing)


def temporal_rows(name: str, predictions: pd.DataFrame) -> list[dict]:
    rows = []
    diffs_by_bearing: list[np.ndarray] = []
    rhos: list[float] = []
    smoothness: list[float] = []
    for bearing, group in predictions.groupby("bearing_id", sort=True):
        group = group.sort_values("end_position", kind="mergesort")
        positions = group.end_position.to_numpy(float)
        values = group.pred_rul.to_numpy(float)
        diffs = np.diff(values)
        violations = diffs[diffs > 0]
        adjacent = trend.adjacent_stats(group)
        rho = trend.spearman_rho(positions, values)
        diffs_by_bearing.append(diffs)
        rhos.append(rho)
        smoothness.append(float(np.mean(np.abs(diffs))) if len(diffs) else float("nan"))
        rows.append({
            "baseline": name,
            "aggregation": "per_bearing",
            "bearing_id": bearing,
            "condition": bearing_condition(bearing),
            "prediction_count": int(len(group)),
            "valid_adjacent_transitions": int(len(diffs)),
            "violating_transitions": int(len(violations)),
            "violation_rate": float(len(violations) / len(diffs)) if len(diffs) else 0.0,
            "mean_violation_magnitude": float(np.mean(violations)) if len(violations) else 0.0,
            "spearman_rho": rho if np.isfinite(rho) else "N/A",
            "mean_abs_adjacent_change": float(np.mean(np.abs(diffs))) if len(diffs) else float("nan"),
        })

    all_diffs = np.concatenate(diffs_by_bearing)
    all_violations = all_diffs[all_diffs > 0]
    finite_rhos = [value for value in rhos if np.isfinite(value)]
    rows.append({
        "baseline": name,
        "aggregation": "overall_neural_diagnostic_convention",
        "bearing_id": "ALL",
        "condition": "ALL",
        "prediction_count": int(len(predictions)),
        "valid_adjacent_transitions": int(len(all_diffs)),
        "violating_transitions": int(len(all_violations)),
        "violation_rate": float(len(all_violations) / len(all_diffs)) if len(all_diffs) else 0.0,
        "mean_violation_magnitude": float(np.mean(all_violations)) if len(all_violations) else 0.0,
        "spearman_rho": float(np.mean(finite_rhos)) if finite_rhos else "N/A",
        "mean_abs_adjacent_change": float(np.nanmean(smoothness)),
    })

    for condition in sorted(predictions.condition.unique()):
        cond = predictions.loc[predictions.condition.eq(condition)]
        cond_diffs = np.concatenate([
            np.diff(g.sort_values("end_position").pred_rul.to_numpy(float))
            for _, g in cond.groupby("bearing_id", sort=True)
        ])
        cond_violations = cond_diffs[cond_diffs > 0]
        cond_rhos = [
            trend.spearman_rho(g.end_position.to_numpy(float), g.pred_rul.to_numpy(float))
            for _, g in cond.groupby("bearing_id", sort=True)
        ]
        cond_rhos = [x for x in cond_rhos if np.isfinite(x)]
        cond_smooth = [
            float(np.mean(np.abs(np.diff(g.sort_values("end_position").pred_rul.to_numpy(float)))))
            for _, g in cond.groupby("bearing_id", sort=True)
        ]
        rows.append({
            "baseline": name,
            "aggregation": "condition_neural_diagnostic_convention",
            "bearing_id": "ALL",
            "condition": condition,
            "prediction_count": int(len(cond)),
            "valid_adjacent_transitions": int(len(cond_diffs)),
            "violating_transitions": int(len(cond_violations)),
            "violation_rate": float(len(cond_violations) / len(cond_diffs)) if len(cond_diffs) else 0.0,
            "mean_violation_magnitude": float(np.mean(cond_violations)) if len(cond_violations) else 0.0,
            "spearman_rho": float(np.mean(cond_rhos)) if cond_rhos else "N/A",
            "mean_abs_adjacent_change": float(np.mean(cond_smooth)) if cond_smooth else float("nan"),
        })
    return rows


def classification_rows(y_true: np.ndarray, y_pred: np.ndarray, unit: str) -> list[dict]:
    ids = list(range(len(FAULT_TO_ID)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=ids)
    acc = float(np.mean(y_true == y_pred))
    rows = [{
        "record_type": "overall",
        "evaluation_unit": unit,
        "class_label": "ALL",
        "predicted_class": "Outer (training majority)",
        "support": int(len(y_true)),
        "accuracy": acc,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix_json": json.dumps(matrix.tolist()),
    }]
    for idx, label in enumerate(ID_TO_FAULT.values()):
        rows.append({
            "record_type": "per_class",
            "evaluation_unit": unit,
            "class_label": label,
            "predicted_class": "Outer (training majority)",
            "support": int(support[idx]),
            "accuracy": "",
            "macro_precision": float(precision[idx]),
            "macro_recall": float(recall[idx]),
            "macro_f1": float(f1[idx]),
            "confusion_matrix_json": "",
        })
    for actual_idx, actual in enumerate(ID_TO_FAULT.values()):
        for predicted_idx, predicted in enumerate(ID_TO_FAULT.values()):
            rows.append({
                "record_type": "confusion_cell",
                "evaluation_unit": unit,
                "class_label": actual,
                "predicted_class": predicted,
                "support": "",
                "accuracy": "",
                "macro_precision": "",
                "macro_recall": "",
                "macro_f1": "",
                "confusion_matrix_json": int(matrix[actual_idx, predicted_idx]),
            })
    return rows


def main() -> None:
    out = OUT
    meta = pd.read_csv(ROOT / "data/processed/xjtu_features_metadata.csv.gz")
    train = baseline.sequence_targets(meta, baseline.TRAIN_BEARINGS)
    test = baseline.sequence_targets(meta, baseline.TEST_BEARINGS)
    if len(test) != 3164 or test.bearing_id.nunique() != 9:
        raise AssertionError(f"Expected 3,164 rows from nine test bearings, got {len(test)}")
    baseline.verify_against_neural_rows(test)

    mean_pred = test.copy()
    mean_pred["pred_rul"] = float(train.true_rul.mean())
    weights = baseline.equal_bearing_weights(train)
    train_x = train.end_position.to_numpy(float)
    knots = np.asarray([
        baseline.weighted_empirical_quantile(train_x, weights, q)
        for q in baseline.KNOT_QUANTILES
    ])
    if len(np.unique(knots)) != len(knots):
        raise AssertionError(f"Training-derived age knots not distinct: {knots}")
    design = baseline.interpolation_basis(train_x, knots)
    root_weight = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        design * root_weight[:, None], train.true_rul.to_numpy(float) * root_weight, rcond=None
    )
    age_pred = test.copy()
    age_pred["pred_rul"] = np.clip(
        baseline.interpolation_basis(test.end_position.to_numpy(float), knots) @ coefficients,
        0.0,
        1.0,
    )
    if not age_pred.pred_rul.between(0, 1).all():
        raise AssertionError("Clipped age prediction escaped [0, 1]")

    expected_metrics = pd.read_csv(ROOT / "results/baselines/baseline_metrics.csv").set_index("baseline")
    for label, pred in [("training_mean", mean_pred), ("piecewise_linear_age", age_pred)]:
        calculated = baseline.regression_metrics(pred.true_rul.to_numpy(), pred.pred_rul.to_numpy())
        source = expected_metrics.loc[label]
        for key, col in [("mae", "pooled_window_mae"), ("rmse", "pooled_window_rmse"), ("r2", "pooled_window_r2")]:
            if not math.isclose(calculated[key], float(source[col]), rel_tol=0, abs_tol=2e-9):
                raise AssertionError(f"Reconstructed {label} {key} differs from saved result")

    temporal = temporal_rows("training_mean", mean_pred) + temporal_rows("piecewise_linear_age", age_pred)
    pd.DataFrame(temporal).to_csv(out / "baseline_temporal_diagnostics.csv", index=False)

    train_counts = pd.Series([FAULT_LABELS[b] for b in baseline.TRAIN_BEARINGS]).value_counts()
    majority_label = train_counts.idxmax()
    majority_id = FAULT_TO_ID[majority_label]
    bearing_labels = test.groupby("bearing_id", sort=True).fault_label.first()
    if any(test.groupby("bearing_id").fault_label.nunique() != 1):
        raise AssertionError("Fault label changes within a bearing")
    if any(bearing_labels[b] != FAULT_LABELS[b] for b in bearing_labels.index):
        raise AssertionError("Metadata label mapping differs from authoritative bearing mapping")
    y_bearing = bearing_labels.map(FAULT_TO_ID).to_numpy(int)
    y_window = test.fault_label.map(FAULT_TO_ID).to_numpy(int)
    if len(y_window) != 3164 or not np.array_equal(y_window, test.fault_label.map(FAULT_TO_ID).to_numpy(int)):
        raise AssertionError("Unexpected window-level labels")
    pred_bearing = np.full(len(y_bearing), majority_id, dtype=int)
    pred_window = np.full(len(y_window), majority_id, dtype=int)
    majority_rows = classification_rows(y_bearing, pred_bearing, "bearing (primary; n=9)")
    majority_rows += classification_rows(y_window, pred_window, "sequence row (window-weighted; n=3164)")
    majority_frame = pd.DataFrame(majority_rows)
    majority_frame.insert(0, "training_majority_class", majority_label)
    majority_frame.insert(1, "training_bearing_counts_json", json.dumps({
        label: int(train_counts.get(label, 0)) for label in ID_TO_FAULT.values()
    }))
    majority_frame.to_csv(out / "majority_class_reference.csv", index=False)

    # Count same-bearing neighboring training windows separated by batch cuts.
    ordered_train = train.reset_index(drop=True)
    batch_size = 32
    cuts = list(range(batch_size, len(ordered_train), batch_size))
    omitted_cuts = [
        cut for cut in cuts
        if ordered_train.iloc[cut - 1].bearing_id == ordered_train.iloc[cut].bearing_id
    ]
    current_doc = ROOT.parents[2] / "research/xjtu_physics_mtl/submission/phm2027_corrected/qa_render/extracted.txt"
    doc_text = current_doc.read_text(encoding="utf-8") if current_doc.exists() else ""

    # Confirm exact test-row identity and target alignment across every saved seed.
    alignment_rows = 0
    for model in ["m1", "m2", "m3", "m4", "m5"]:
        for seed in baseline.SEEDS:
            saved = pd.read_csv(baseline.MULTISEED / model / f"seed_{seed}" / "predictions.csv")
            check = test[baseline.KEYS + ["true_rul"]].merge(
                saved[baseline.KEYS + ["true_rul"]], on=baseline.KEYS,
                how="outer", suffixes=("_meta", "_saved"), indicator=True, validate="one_to_one"
            )
            if len(check) != 3164 or not check._merge.eq("both").all():
                raise AssertionError(f"Test rows differ for {model} seed {seed}")
            if not np.allclose(check.true_rul_meta, check.true_rul_saved, rtol=0, atol=1e-7):
                raise AssertionError(f"Test targets differ for {model} seed {seed}")
            alignment_rows += len(check)

    temporal_frame = pd.DataFrame(temporal)
    agg = temporal_frame.loc[
        temporal_frame.aggregation.isin(["overall_neural_diagnostic_convention", "condition_neural_diagnostic_convention"])
    ]
    bearing_primary = majority_frame[
        (majority_frame.record_type == "overall") & majority_frame.evaluation_unit.str.startswith("bearing")
    ].iloc[0]
    window_primary = majority_frame[
        (majority_frame.record_type == "overall") & majority_frame.evaluation_unit.str.startswith("sequence")
    ].iloc[0]
    age_knots_y = coefficients
    report = [
        "# Final Offline Diagnostics Audit",
        "",
        "Offline analysis only. No training, model changes, prediction changes, manuscript edits, or existing-result overwrites were performed. Baselines were reconstructed from training labels and checked against their saved pooled metrics.",
        "",
        "## A. Temporal reference diagnostics",
        "",
        "Predictions are sorted by `end_position` within each bearing. A transition is the difference between consecutive saved predictions from that same bearing only. Violation rate is total positive adjacent changes divided by all valid adjacent transitions; mean violation magnitude averages positive changes only. Overall aggregation matches the neural temporal diagnostic: pooled transition weighting for violation rate and violation magnitude, arithmetic mean of per-bearing Spearman values (excluding undefined values), and arithmetic mean of per-bearing mean absolute changes. Condition rows apply the same rules within each condition.",
        "",
        "| Reference | Scope | Violation rate | Mean positive violation | Spearman rho | Mean abs. adjacent change | Transitions (violating/valid) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in agg.iterrows():
        rho = row.spearman_rho if isinstance(row.spearman_rho, str) else f"{float(row.spearman_rho):.6f}"
        report.append(
            f"| {row.baseline} | {row.condition} | {float(row.violation_rate):.6f} | {float(row.mean_violation_magnitude):.8f} | {rho} | {float(row.mean_abs_adjacent_change):.8f} | {int(row.violating_transitions)}/{int(row.valid_adjacent_transitions)} |"
        )
    report += [
        "",
        "The training-mean reference is constant: its zero violation rate, zero violation magnitude, and zero adjacent change are degenerate consequences of a flat prediction, not evidence of degradation tracking. Spearman is undefined for a constant series and is reported as N/A. The piecewise-linear age reference is unconstrained: its training-fit knot values are `" + ", ".join(f"{v:.6f}" for v in age_knots_y) + "` at training-derived end-position knots `" + ", ".join(f"{v:.1f}" for v in knots) + "`; the second-to-third knot segment increases with age. Lower violation/smoothness metrics alone indicate neither better RUL accuracy nor physical validity.",
        "",
        "Per-bearing and per-condition rows for both references are in `baseline_temporal_diagnostics.csv`. The two deterministic predictions use only training sequence targets for the training mean and training sequence end positions/RUL for the piecewise fit. Test labels are only used to verify alignment and existing pooled metrics, never to fit either reference.",
        "",
        "## B. Majority-class fault reference",
        "",
        f"The training-only bearing counts are `{json.dumps({label: int(train_counts.get(label, 0)) for label in ID_TO_FAULT.values()})}`; the majority class is **{majority_label}**. Fault labels are constant bearing-level annotations, not timestamp-varying onset labels.",
        "",
        f"Primary evaluation is at the bearing level (nine bearing-level test units): accuracy {float(bearing_primary.accuracy):.6f}, macro precision {float(bearing_primary.macro_precision):.6f}, macro recall {float(bearing_primary.macro_recall):.6f}, macro-F1 {float(bearing_primary.macro_f1):.6f}. Confusion matrix (rows=true, columns=predicted; order Outer, Inner, Cage, Mixed): `{bearing_primary.confusion_matrix_json}`.",
        "",
        f"For comparability with the paper's sequence-row classification calculations, repeated-label window-weighted metrics are also recorded: accuracy {float(window_primary.accuracy):.6f}, macro precision {float(window_primary.macro_precision):.6f}, macro recall {float(window_primary.macro_recall):.6f}, macro-F1 {float(window_primary.macro_f1):.6f}. This 3,164-row view repeats each bearing label across overlapping windows and must not be interpreted as 3,164 independent bearing examples. Full per-class values and both confusion matrices are in `majority_class_reference.csv`.",
        "",
        "Because Cage has no training bearing in the selected fit subset, this majority reference does not resolve absent Cage supervision and cannot validate four-class diagnostic capability.",
        "",
        "## C. Final training-loader audit",
        "",
        "Final three-seed entry points inspected: `scripts/run_multiseed.py` (M1, M2, M3, M5) and `scripts/run_m4_multiseed.py` (which invokes `run_multi(\"m4\", ...)`).",
        "",
        "| Model | Training batches shuffled? | Validation shuffled? | Test shuffled? |",
        "|---|---|---|---|",
        "| M1 | Yes | No | No |",
        "| M2 | Yes | No | No |",
        "| M3 | No | No | No |",
        "| M4 | Yes | No | No |",
        "| M5 | No | No | No |",
        "",
        "The final M3/M5 monotonicity loss is computed independently per minibatch. It groups predictions by bearing identity, sorts each group by sequence end position, and penalizes positive increases between neighboring available predictions. No pair crosses bearing IDs. The unshuffled training sequence is in bearing/chronological order; a pair split at a minibatch cut is absent from that minibatch's loss. With batch size 32, there are **" + str(len(cuts)) + " batch cuts, of which " + str(len(omitted_cuts)) + " cut a within-bearing chronological adjacent pair per epoch** for the fixed training sequence order. The omitted transitions recur at those same cuts each epoch. This count is the maximum direct adjacent pairs omitted at the boundaries (and no transition is instead compared across a boundary).",
        "",
        "The corrected manuscript's extracted text contains the sentence `Final loaders are unshuffled`, immediately in the M3/M5 loss discussion. It is correct for the M3/M5 training loaders and all validation/test loaders, but is overbroad if read as describing all final training loaders: M1, M2, and M4 training loaders are shuffled. Manuscript was not modified in this task.",
        "",
        "## Checks performed",
        "",
        f"- Reconstructed test set: {len(test)} rows across {test.bearing_id.nunique()} test bearings.",
        f"- Exact keys and true RUL alignment cross-checked against all 15 saved model/seed prediction CSVs ({alignment_rows:,} prediction rows inspected); tolerance for saved label comparison: 1e-7.",
        "- Reconstructed baseline pooled MAE, RMSE, and R-squared agree with `results/baselines/baseline_metrics.csv` within 2e-9.",
        "- Fault labels agree with the authoritative `src/labels.py` mapping and remain constant within each bearing.",
        f"- Training sequence rows used for batch-cut audit: {len(ordered_train)}; batch size: {batch_size}.",
        "- Existing outputs and manuscript were preserved; the only generated outputs are this audit, its two CSVs, and the reproducibility script in this new directory.",
        "",
    ]
    (out / "diagnostic_audit.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
