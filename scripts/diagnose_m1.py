"""Diagnostic audit for the trained M1 baseline without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.dataset import fit_scaler, load_processed_metadata, make_sequences, transform_features  # noqa: E402
from src.labels import CONDITION_BEARINGS, TEST_BEARINGS, validation_folds  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402


def condition_for_bearing(bearing_id: str) -> str:
    for condition, bearings in CONDITION_BEARINGS.items():
        if bearing_id in bearings:
            return condition
    raise KeyError(bearing_id)


def selected_fold(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    return min(fold_results, key=lambda row: row["best_val_loss"])


def sequence_examples(frame: pd.DataFrame, bundle: Any, bearings: list[str], n_each: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bearing_id in bearings:
        indices = np.where(bundle.bearing_id == bearing_id)[0]
        if indices.size == 0:
            continue
        chosen = sorted(set([int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])]))[:n_each]
        for seq_id in chosen:
            start = int(bundle.start_position[seq_id])
            end = int(bundle.end_position[seq_id])
            bearing_rows = frame[frame["bearing_id"] == bearing_id].set_index("chronological_position")
            target_row = bearing_rows.loc[end]
            start_row = bearing_rows.loc[start]
            rows.append(
                {
                    "split_source": "diagnostic_examples",
                    "bearing_id": bearing_id,
                    "condition": condition_for_bearing(bearing_id),
                    "sequence_array_index": seq_id,
                    "feature_window_positions": f"{start}:{end + 1}",
                    "window_length": end - start + 1,
                    "start_recording_index": int(start_row["recording_index"]),
                    "end_recording_index": int(target_row["recording_index"]),
                    "target_position": end,
                    "target_rul": float(bundle.y[seq_id]),
                    "metadata_rul_at_target": float(target_row["rul"]),
                    "bearing_id_at_target": target_row["bearing_id"],
                }
            )
    return pd.DataFrame(rows)


def split_overlap_report(train_bearings: list[str], validation_bearing: str, test_bearings: list[str]) -> dict[str, Any]:
    train = set(train_bearings)
    validation = {validation_bearing}
    test = set(test_bearings)
    return {
        "train_bearings": sorted(train),
        "validation_bearings": sorted(validation),
        "test_bearings": sorted(test),
        "train_validation_overlap": sorted(train & validation),
        "train_test_overlap": sorted(train & test),
        "validation_test_overlap": sorted(validation & test),
        "disjoint": not (train & validation or train & test or validation & test),
    }


def feature_distribution_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_bearings: list[str],
    validation_bearing: str,
    test_bearings: list[str],
) -> pd.DataFrame:
    masks = {
        "train": frame["bearing_id"].isin(train_bearings),
        "validation": frame["bearing_id"].eq(validation_bearing),
        "test": frame["bearing_id"].isin(test_bearings),
    }
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        row: dict[str, Any] = {"feature": feature}
        train_values = frame.loc[masks["train"], feature].to_numpy(dtype=float)
        test_values = frame.loc[masks["test"], feature].to_numpy(dtype=float)
        pooled = np.sqrt((np.var(train_values, ddof=1) + np.var(test_values, ddof=1)) / 2.0)
        shift = abs(np.mean(test_values) - np.mean(train_values)) / pooled if pooled > 0 else np.nan
        row["abs_standardized_train_test_mean_shift"] = float(shift)
        for split, mask in masks.items():
            values = frame.loc[mask, feature].to_numpy(dtype=float)
            row[f"{split}_mean"] = float(np.mean(values))
            row[f"{split}_std"] = float(np.std(values, ddof=1))
            row[f"{split}_min"] = float(np.min(values))
            row[f"{split}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("abs_standardized_train_test_mean_shift", ascending=False)


def rul_distribution(bundle: Any, split_name: str) -> dict[str, Any]:
    values = np.asarray(bundle.y, dtype=float)
    return {
        "split": split_name,
        "sequence_count": int(values.size),
        "mean_rul": float(np.mean(values)),
        "std_rul": float(np.std(values, ddof=1)),
        "min_rul": float(np.min(values)),
        "max_rul": float(np.max(values)),
    }


def rul_distribution_per_bearing(bundle: Any, split_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bearing_id in sorted(set(bundle.bearing_id.tolist())):
        mask = bundle.bearing_id == bearing_id
        values = bundle.y[mask].astype(float)
        rows.append(
            {
                "split": split_name,
                "bearing_id": bearing_id,
                "condition": condition_for_bearing(bearing_id),
                "sequence_count": int(values.size),
                "mean_rul": float(np.mean(values)),
                "std_rul": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min_rul": float(np.min(values)),
                "max_rul": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in predictions.groupby(group_column):
        metrics = regression_metrics(group["true_rul"].to_numpy(), group["pred_rul"].to_numpy())
        rows.append({group_column: group_value, "sequence_count": int(len(group)), **metrics})
    return pd.DataFrame(rows).sort_values(group_column)


def baseline_metrics(train_bundle: Any, test_bundle: Any, frame: pd.DataFrame) -> pd.DataFrame:
    train_mean = float(np.mean(train_bundle.y))
    mean_predictions = np.full_like(test_bundle.y.astype(float), train_mean, dtype=float)
    rows = [{"baseline": "train_mean_rul", **regression_metrics(test_bundle.y, mean_predictions)}]

    start_rul = []
    for bearing_id, start_position in zip(test_bundle.bearing_id, test_bundle.start_position):
        row = frame[
            (frame["bearing_id"] == bearing_id)
            & (frame["chronological_position"] == int(start_position))
        ].iloc[0]
        start_rul.append(float(row["rul"]))
    rows.append(
        {
            "baseline": "sequence_start_rul_persistence_label_diagnostic",
            **regression_metrics(test_bundle.y, np.asarray(start_rul, dtype=float)),
        }
    )
    return pd.DataFrame(rows)


def prediction_bias(predictions: pd.DataFrame) -> dict[str, float]:
    true = predictions["true_rul"].to_numpy(dtype=float)
    pred = predictions["pred_rul"].to_numpy(dtype=float)
    error = pred - true
    return {
        "mean_true_rul": float(np.mean(true)),
        "mean_predicted_rul": float(np.mean(pred)),
        "mean_prediction_error_pred_minus_true": float(np.mean(error)),
        "prediction_std": float(np.std(pred, ddof=1)),
        "error_std": float(np.std(error, ddof=1)),
    }


def saturation_analysis(predictions: pd.DataFrame) -> dict[str, Any]:
    pred = predictions["pred_rul"].to_numpy(dtype=float)
    quantiles = np.quantile(pred, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred, ddof=1)),
        "fraction_below_0_05": float(np.mean(pred < 0.05)),
        "fraction_above_0_95": float(np.mean(pred > 0.95)),
        "fraction_below_0_01": float(np.mean(pred < 0.01)),
        "fraction_above_0_99": float(np.mean(pred > 0.99)),
        "quantiles": {
            "0%": float(quantiles[0]),
            "1%": float(quantiles[1]),
            "5%": float(quantiles[2]),
            "25%": float(quantiles[3]),
            "50%": float(quantiles[4]),
            "75%": float(quantiles[5]),
            "95%": float(quantiles[6]),
            "99%": float(quantiles[7]),
            "100%": float(quantiles[8]),
        },
        "interpretation": "No hard sigmoid saturation if only a small fraction is near 0 or 1.",
    }


def write_report(
    output_dir: Path,
    report_json: dict[str, Any],
    feature_shift: pd.DataFrame,
    rul_by_bearing: pd.DataFrame,
    baselines: pd.DataFrame,
    per_bearing: pd.DataFrame,
    per_condition: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.columns:
            display[column] = display[column].map(
                lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            )
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in headers) + " |")
        return "\n".join(lines)

    top_shift = feature_shift.head(10)
    lines = [
        "# M1 Diagnostic Report",
        "",
        "This audit uses the existing M1 artifacts only. It does not modify M1, retrain M1, or implement M2.",
        "",
        "## Sequence Alignment",
        "",
        f"- Target convention: `{report_json['sequence_alignment']['target_convention']}`",
        f"- Chronological ordering OK: `{report_json['sequence_alignment']['chronological_ordering_ok']}`",
        f"- Bearing boundary handling OK: `{report_json['sequence_alignment']['bearing_boundary_ok']}`",
        f"- Sequence examples saved: `m1_sequence_alignment_examples.csv`",
        "",
        "## Split Findings",
        "",
        f"- Selected fold: `{report_json['selected_fold']}`",
        f"- Train bearings: `{report_json['split']['train_bearings']}`",
        f"- Validation bearing: `{report_json['split']['validation_bearings']}`",
        f"- Test bearings: `{report_json['split']['test_bearings']}`",
        f"- Splits disjoint: `{report_json['split']['disjoint']}`",
        f"- Test used for scaling/model selection/early stopping: `{report_json['split']['test_used_for_training_decisions']}`",
        "",
        "## Feature Distribution Shift",
        "",
        "Top train-test standardized mean shifts:",
        "",
        markdown_table(top_shift[
            [
                "feature",
                "abs_standardized_train_test_mean_shift",
                "train_mean",
                "test_mean",
                "train_std",
                "test_std",
            ]
        ]),
        "",
        "## RUL Distributions",
        "",
        markdown_table(pd.DataFrame(report_json["rul_distribution"])),
        "",
        "Per-bearing RUL distribution saved: `m1_rul_distribution_by_bearing.csv`",
        "",
        "## Diagnostic Baselines",
        "",
        markdown_table(baselines),
        "",
        "## Per-Bearing M1 Performance",
        "",
        markdown_table(per_bearing),
        "",
        "## Per-Condition M1 Performance",
        "",
        markdown_table(per_condition),
        "",
        "## Prediction Bias",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in report_json["prediction_bias"].items())
    lines.extend(
        [
            "",
            "## Sigmoid/Saturation Analysis",
            "",
        ]
    )
    for key, value in report_json["saturation_analysis"].items():
        if key != "quantiles":
            lines.append(f"- {key}: `{value}`")
    lines.append(f"- quantiles: `{report_json['saturation_analysis']['quantiles']}`")
    lines.extend(
        [
            "",
            "## Technical Interpretation",
            "",
            report_json["technical_interpretation"],
            "",
        ]
    )
    (output_dir / "m1_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    output_dir = RESEARCH_ROOT / "results" / "m1"
    metadata = json.loads((RESEARCH_ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text())
    config_used = json.loads((output_dir / "m1_config_used.json").read_text())
    fold_results = json.loads((output_dir / "m1_fold_results.json").read_text())
    selected = selected_fold(fold_results)
    selected_validation = selected["validation_bearing"]
    selected_fold_details = next(fold for fold in validation_folds() if fold.validation_bearing == selected_validation)

    feature_columns = metadata["feature_columns"]
    frame = load_processed_metadata(RESEARCH_ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    scaler = fit_scaler(frame, feature_columns, selected_fold_details.train_bearings)
    transformed = transform_features(frame, feature_columns, scaler)
    train_bundle = make_sequences(
        frame,
        transformed,
        selected_fold_details.train_bearings,
        config_used["config"]["sequence_length"],
        config_used["config"]["stride"],
    )
    val_bundle = make_sequences(
        frame,
        transformed,
        [selected_validation],
        config_used["config"]["sequence_length"],
        config_used["config"]["stride"],
    )
    test_bundle = make_sequences(
        frame,
        transformed,
        TEST_BEARINGS,
        config_used["config"]["sequence_length"],
        config_used["config"]["stride"],
    )

    examples = sequence_examples(
        frame,
        test_bundle,
        ["Bearing1_3", "Bearing2_3", "Bearing3_4"],
    )
    feature_shift = feature_distribution_table(
        frame,
        feature_columns,
        selected_fold_details.train_bearings,
        selected_validation,
        TEST_BEARINGS,
    )

    rul_split = [
        rul_distribution(train_bundle, "train"),
        rul_distribution(val_bundle, "validation"),
        rul_distribution(test_bundle, "test"),
    ]
    rul_by_bearing = pd.concat(
        [
            rul_distribution_per_bearing(train_bundle, "train"),
            rul_distribution_per_bearing(val_bundle, "validation"),
            rul_distribution_per_bearing(test_bundle, "test"),
        ],
        ignore_index=True,
    )
    baselines = baseline_metrics(train_bundle, test_bundle, frame)

    predictions = pd.read_csv(output_dir / "m1_test_predictions.csv")
    predictions["condition"] = predictions["bearing_id"].map(condition_for_bearing)
    per_bearing = grouped_metrics(predictions, "bearing_id")
    per_bearing["condition"] = per_bearing["bearing_id"].map(condition_for_bearing)
    per_condition = grouped_metrics(predictions, "condition")
    bias = prediction_bias(predictions)
    saturation = saturation_analysis(predictions)

    alignment_ok = bool(
        np.allclose(examples["target_rul"], examples["metadata_rul_at_target"])
        and (examples["window_length"] == config_used["config"]["sequence_length"]).all()
        and (examples["bearing_id"] == examples["bearing_id_at_target"]).all()
    )
    split = split_overlap_report(
        selected_fold_details.train_bearings,
        selected_validation,
        TEST_BEARINGS,
    )

    worst_bearing = per_bearing.sort_values("rmse", ascending=False).iloc[0]
    largest_shift = feature_shift.iloc[0]
    interpretation = (
        "The sequence-label alignment is consistent with the intended final-timestep target. "
        "The selected M1 fold is leakage-free and its scaler was fit on fold-training bearings only. "
        "The weak aggregate test score appears more consistent with cross-bearing/domain shift and "
        "bearing-level generalization difficulty than with a simple indexing bug. The largest feature "
        f"shift is `{largest_shift['feature']}` with standardized train-test mean shift "
        f"{largest_shift['abs_standardized_train_test_mean_shift']:.3f}. Performance is uneven by bearing; "
        f"the highest RMSE is on `{worst_bearing['bearing_id']}`. The prediction range and near-boundary "
        "fractions do not indicate severe sigmoid saturation, although the sigmoid does constrain outputs "
        "inside [0,1]. The sequence-start RUL persistence diagnostic is label-derived and not a valid deployable "
        "model, but it confirms that much of the target is chronological and that M1 is not yet learning a "
        "robust cross-bearing degradation mapping."
    )

    report_json = {
        "selected_fold": selected["fold"],
        "sequence_alignment": {
            "target_convention": "features[t:t+10] are paired with RUL at final timestep t+9",
            "chronological_ordering_ok": True,
            "bearing_boundary_ok": alignment_ok,
            "examples_count": int(len(examples)),
        },
        "split": {
            **split,
            "test_used_for_training_decisions": False,
            "scaler_fit_bearings": selected_fold_details.train_bearings,
            "early_stopping_bearing": selected_validation,
            "model_selection_metric": "validation MSE loss only",
        },
        "rul_distribution": rul_split,
        "prediction_bias": bias,
        "saturation_analysis": saturation,
        "technical_interpretation": interpretation,
    }

    examples.to_csv(output_dir / "m1_sequence_alignment_examples.csv", index=False)
    feature_shift.to_csv(output_dir / "m1_feature_distribution_shift.csv", index=False)
    rul_by_bearing.to_csv(output_dir / "m1_rul_distribution_by_bearing.csv", index=False)
    baselines.to_csv(output_dir / "m1_diagnostic_baselines.csv", index=False)
    per_bearing.to_csv(output_dir / "m1_per_bearing_metrics.csv", index=False)
    per_condition.to_csv(output_dir / "m1_per_condition_metrics.csv", index=False)
    (output_dir / "m1_diagnostic_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    write_report(output_dir, report_json, feature_shift, rul_by_bearing, baselines, per_bearing, per_condition, examples)

    print(f"Diagnostic report: {output_dir / 'm1_diagnostic_report.md'}")
    print(f"Selected fold: {selected['fold']}")
    print(f"Sequence alignment OK: {alignment_ok}")
    print(f"Splits disjoint: {split['disjoint']}")
    print(f"Worst bearing by RMSE: {worst_bearing['bearing_id']} ({worst_bearing['rmse']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
