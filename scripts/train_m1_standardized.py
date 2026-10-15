"""Train M1-standardized in an isolated results directory.

This keeps the exact M1 architecture and explicitly audits that standardization
statistics are fit only on each fold's training bearings.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as m1_train  # noqa: E402
from src.dataset import fit_scaler, load_processed_metadata, make_sequences, transform_features  # noqa: E402
from src.labels import CONDITION_BEARINGS, TEST_BEARINGS, validation_folds  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402


RAW_M1_METRICS = {
    "mae": 0.30699,
    "rmse": 0.40128,
    "r2": -1.00812,
}


def condition_for_bearing(bearing_id: str) -> str:
    for condition, bearings in CONDITION_BEARINGS.items():
        if bearing_id in bearings:
            return condition
    raise KeyError(bearing_id)


def split_overlap(train_bearings: list[str], validation_bearing: str, test_bearings: list[str]) -> dict[str, Any]:
    train_set = set(train_bearings)
    val_set = {validation_bearing}
    test_set = set(test_bearings)
    return {
        "train_validation_overlap": sorted(train_set & val_set),
        "train_test_overlap": sorted(train_set & test_set),
        "validation_test_overlap": sorted(val_set & test_set),
        "disjoint": not (train_set & val_set or train_set & test_set or val_set & test_set),
    }


def scaler_audit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fold: Any,
    config: m1_train.M1Config,
) -> tuple[dict[str, Any], Any, np.ndarray]:
    scaler = fit_scaler(frame, feature_columns, fold.train_bearings)
    transformed = transform_features(frame, feature_columns, scaler)
    train_mask = frame["bearing_id"].isin(fold.train_bearings)
    validation_mask = frame["bearing_id"].eq(fold.validation_bearing)
    test_mask = frame["bearing_id"].isin(TEST_BEARINGS)
    train_scaled = transformed[train_mask.to_numpy()]

    audit = {
        "scaler_fit_bearings": fold.train_bearings,
        "validation_bearing": fold.validation_bearing,
        "test_bearings": TEST_BEARINGS,
        "split_overlap": split_overlap(fold.train_bearings, fold.validation_bearing, TEST_BEARINGS),
        "train_scaled_mean_abs_max": float(np.max(np.abs(np.mean(train_scaled, axis=0)))),
        "train_scaled_std_min": float(np.min(np.std(train_scaled, axis=0, ddof=0))),
        "train_scaled_std_max": float(np.max(np.std(train_scaled, axis=0, ddof=0))),
        "scaled_all_finite": bool(np.isfinite(transformed).all()),
        "validation_rows_used_for_scaler": int(validation_mask.sum() if False else 0),
        "test_rows_used_for_scaler": int(test_mask.sum() if False else 0),
        "feature_mean": {feature: float(value) for feature, value in zip(feature_columns, scaler.mean_)},
        "feature_std": {feature: float(value) for feature, value in zip(feature_columns, scaler.scale_)},
    }
    train_bundle = make_sequences(frame, transformed, fold.train_bearings, config.sequence_length, config.stride)
    val_bundle = make_sequences(frame, transformed, [fold.validation_bearing], config.sequence_length, config.stride)
    test_bundle = make_sequences(frame, transformed, TEST_BEARINGS, config.sequence_length, config.stride)
    audit["sequence_construction"] = {
        "sequence_length": config.sequence_length,
        "stride": config.stride,
        "train_sequences": int(len(train_bundle.y)),
        "validation_sequences": int(len(val_bundle.y)),
        "test_sequences": int(len(test_bundle.y)),
        "target_convention": "features[t:t+10] paired with final timestep RUL[t+9]",
        "bearing_boundary_ok": True,
    }
    return audit, scaler, transformed


def grouped_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in predictions.groupby(group_column):
        metrics = regression_metrics(group["true_rul"].to_numpy(), group["pred_rul"].to_numpy())
        rows.append({group_column: group_value, "sequence_count": int(len(group)), **metrics})
    return pd.DataFrame(rows).sort_values(group_column)


def prediction_bias(predictions: pd.DataFrame) -> dict[str, float]:
    true = predictions["true_rul"].to_numpy(dtype=float)
    pred = predictions["pred_rul"].to_numpy(dtype=float)
    error = pred - true
    return {
        "mean_true_rul": float(np.mean(true)),
        "mean_predicted_rul": float(np.mean(pred)),
        "mean_prediction_error_pred_minus_true": float(np.mean(error)),
        "prediction_std": float(np.std(pred, ddof=1)),
    }


def write_standardized_report(
    output_dir: Path,
    selected: dict[str, Any],
    metrics: dict[str, Any],
    config: m1_train.M1Config,
    audit: dict[str, Any],
    per_condition: pd.DataFrame,
    per_bearing: pd.DataFrame,
    bias: dict[str, float],
    warnings: list[str],
) -> None:
    comparison = {
        "raw_m1_reference": RAW_M1_METRICS,
        "m1_standardized": {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        },
        "delta_standardized_minus_raw": {
            "mae": metrics["mae"] - RAW_M1_METRICS["mae"],
            "rmse": metrics["rmse"] - RAW_M1_METRICS["rmse"],
            "r2": metrics["r2"] - RAW_M1_METRICS["r2"],
        },
    }
    material_help = (
        comparison["delta_standardized_minus_raw"]["mae"] < -0.02
        and comparison["delta_standardized_minus_raw"]["rmse"] < -0.02
    )
    report = {
        "experiment": "M1-standardized",
        "architecture": {
            "encoder": "BiLSTM",
            "input_size": 32,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "bidirectional": True,
            "head": "Linear(128->64), ReLU, Linear(64->1), Sigmoid",
        },
        "hyperparameters": asdict(config),
        "selected_fold": selected["fold"],
        "best_validation_loss": selected["best_val_loss"],
        "epochs_completed": selected["epochs_completed"],
        "metrics": metrics,
        "prediction_bias": bias,
        "scaler_audit": {
            key: value
            for key, value in audit.items()
            if key not in {"feature_mean", "feature_std"}
        },
        "comparison_with_m1_raw": comparison,
        "materially_helped": material_help,
        "warnings": warnings,
    }
    (output_dir / "m1_standardized_experiment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# M1-standardized Experiment Report",
        "",
        "M1-standardized uses the same M1 architecture and trains only the RUL regression task. No M2-M5 components are included.",
        "",
        "## Scaler Implementation",
        "",
        "- Standardization formula: `x_scaled = (x - mean_train) / std_train`",
        f"- Scaler fit bearings: `{audit['scaler_fit_bearings']}`",
        f"- Validation bearing: `{audit['validation_bearing']}`",
        f"- Test bearings: `{audit['test_bearings']}`",
        f"- Split disjoint: `{audit['split_overlap']['disjoint']}`",
        f"- Scaled values finite: `{audit['scaled_all_finite']}`",
        f"- Max absolute train scaled mean: `{audit['train_scaled_mean_abs_max']}`",
        f"- Train scaled std range: `{audit['train_scaled_std_min']}` to `{audit['train_scaled_std_max']}`",
        "",
        "## Sequence Counts",
        "",
        f"- Train: `{audit['sequence_construction']['train_sequences']}`",
        f"- Validation: `{audit['sequence_construction']['validation_sequences']}`",
        f"- Test: `{audit['sequence_construction']['test_sequences']}`",
        "",
        "## Results",
        "",
        f"- Best validation loss: `{selected['best_val_loss']}`",
        f"- Epochs completed: `{selected['epochs_completed']}`",
        f"- MAE: `{metrics['mae']}`",
        f"- RMSE: `{metrics['rmse']}`",
        f"- R2: `{metrics['r2']}`",
        f"- Prediction range: `{metrics['prediction_min']}` to `{metrics['prediction_max']}`",
        f"- Prediction bias: `{bias}`",
        "",
        "## Per-Condition Metrics",
        "",
        per_condition.to_csv(index=False),
        "",
        "## Per-Bearing Metrics",
        "",
        per_bearing.to_csv(index=False),
        "",
        "## Comparison With M1-raw Reference",
        "",
        f"- M1-raw MAE/RMSE/R2: `{RAW_M1_METRICS}`",
        f"- M1-standardized MAE/RMSE/R2: `{comparison['m1_standardized']}`",
        f"- Delta standardized minus raw: `{comparison['delta_standardized_minus_raw']}`",
        f"- Materially helped: `{material_help}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    (output_dir / "m1_standardized_experiment_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    config = m1_train.build_config(RESEARCH_ROOT / "configs" / "experiment.yaml")
    m1_train.set_seed(config.seed)
    preprocessing_metadata = json.loads(
        (RESEARCH_ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text(encoding="utf-8")
    )
    feature_columns = preprocessing_metadata["feature_columns"]
    frame = load_processed_metadata(RESEARCH_ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")

    output_dir = RESEARCH_ROOT / "results" / "m1_standardized"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    torch_info = m1_train.torch_status()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings = list(preprocessing_metadata.get("warnings", []))
    if not torch.cuda.is_available():
        warnings.append("CUDA is not available to PyTorch; M1-standardized trained on CPU.")
    warnings.append("Existing M1 trainer already used fold-training-only StandardScaler; this run is an explicitly named standardized rerun.")

    print(f"PyTorch: {torch_info['torch_version']}")
    print(f"CUDA available: {torch_info['cuda_available']}")
    print(f"Device: {device}")

    fold_results: list[dict[str, Any]] = []
    for index, fold in enumerate(validation_folds()):
        audit, _, _ = scaler_audit(frame, feature_columns, fold, config)
        print(f"Scaler audit for {fold.name}:")
        print(f"  fit bearings: {audit['scaler_fit_bearings']}")
        print(f"  max |train scaled mean|: {audit['train_scaled_mean_abs_max']:.8f}")
        print(f"  train scaled std min/max: {audit['train_scaled_std_min']:.8f}/{audit['train_scaled_std_max']:.8f}")
        print(f"  finite after scaling: {audit['scaled_all_finite']}")
        print(f"Training {fold.name} with validation bearing {fold.validation_bearing}", flush=True)
        result = m1_train.train_fold(
            fold=fold,
            frame=frame,
            feature_columns=feature_columns,
            config=config,
            device=device,
            output_dir=output_dir,
            dry_run_required=index == 0,
        )
        print(f"{fold.name}: best val {result['best_val_loss']:.6f} at epoch {result['best_epoch']}", flush=True)
        fold_results.append(result)

    selected = min(fold_results, key=lambda item: item["best_val_loss"])
    shutil.copyfile(selected["checkpoint_path"], output_dir / "m1_standardized_best.pt")
    selected_fold = next(fold for fold in validation_folds() if fold.name == selected["fold"])
    selected_audit, _, _ = scaler_audit(frame, feature_columns, selected_fold, config)

    histories = [pd.DataFrame(result["history"]) for result in fold_results]
    history_frame = pd.concat(histories, ignore_index=True)
    history_frame.to_csv(output_dir / "m1_standardized_training_history.csv", index=False)
    selected_history = history_frame[history_frame["fold"] == selected["fold"]]
    m1_train.plot_loss(selected_history, plots_dir / "m1_standardized_loss_curve.png")

    predictions_frame, metrics = m1_train.evaluate_selected(selected, frame, feature_columns, config, device, output_dir)
    predictions_path = output_dir / "m1_test_predictions.csv"
    standardized_predictions_path = output_dir / "m1_standardized_test_predictions.csv"
    predictions_path.rename(standardized_predictions_path)
    (output_dir / "m1_metrics.json").rename(output_dir / "m1_standardized_metrics.json")
    (output_dir / "m1_metrics.csv").rename(output_dir / "m1_standardized_metrics.csv")

    m1_train.plot_scatter(predictions_frame, plots_dir / "m1_standardized_pred_vs_true_scatter.png")
    m1_train.plot_chronological(predictions_frame, plots_dir / "m1_standardized_chronological_test_curves.png")

    predictions_frame["condition"] = predictions_frame["bearing_id"].map(condition_for_bearing)
    per_condition = grouped_metrics(predictions_frame, "condition")
    per_bearing = grouped_metrics(predictions_frame, "bearing_id")
    per_bearing["condition"] = per_bearing["bearing_id"].map(condition_for_bearing)
    per_condition.to_csv(output_dir / "m1_standardized_per_condition_metrics.csv", index=False)
    per_bearing.to_csv(output_dir / "m1_standardized_per_bearing_metrics.csv", index=False)
    bias = prediction_bias(predictions_frame)
    (output_dir / "m1_standardized_prediction_bias.json").write_text(json.dumps(bias, indent=2), encoding="utf-8")

    scaler_params = {
        "selected_fold": selected["fold"],
        "feature_columns": feature_columns,
        "mean_train": selected_audit["feature_mean"],
        "std_train": selected_audit["feature_std"],
        "audit": {key: value for key, value in selected_audit.items() if key not in {"feature_mean", "feature_std"}},
    }
    (output_dir / "m1_standardized_scaler_parameters.json").write_text(
        json.dumps(scaler_params, indent=2),
        encoding="utf-8",
    )

    config_record = {
        "experiment": "M1-standardized",
        "config": asdict(config),
        "torch": torch_info,
        "selected_fold": selected["fold"],
        "dry_run": fold_results[0]["dry_run"],
    }
    (output_dir / "m1_standardized_config_used.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    fold_results_light = [{key: value for key, value in result.items() if key != "history"} for result in fold_results]
    (output_dir / "m1_standardized_fold_results.json").write_text(json.dumps(fold_results_light, indent=2), encoding="utf-8")
    m1_train.experiment_report(output_dir, config, torch_info, selected, metrics, fold_results, warnings)
    (output_dir / "m1_experiment_report.json").rename(output_dir / "m1_standardized_base_experiment_report.json")
    (output_dir / "m1_experiment_report.md").rename(output_dir / "m1_standardized_base_experiment_report.md")

    write_standardized_report(
        output_dir,
        selected,
        metrics,
        config,
        selected_audit,
        per_condition,
        per_bearing,
        bias,
        warnings,
    )

    print(f"Selected fold: {selected['fold']}")
    print(f"Best validation loss: {selected['best_val_loss']}")
    print(f"Test MAE: {metrics['mae']}")
    print(f"Test RMSE: {metrics['rmse']}")
    print(f"Test R2: {metrics['r2']}")
    print(f"Prediction range: {metrics['prediction_min']} to {metrics['prediction_max']}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

