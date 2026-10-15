"""Evaluate training-only RUL reference baselines against the fixed test rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
META_PATH = ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz"
FINAL_SUMMARY = ROOT / "results" / "final_experiment_summary" / "final_model_comparison.csv"
MULTISEED = ROOT / "results" / "multiseed"
TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing3_1", "Bearing3_2"]
VALIDATION_BEARING = "Bearing2_2"
TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5",
    "Bearing2_3", "Bearing2_4", "Bearing2_5",
    "Bearing3_3", "Bearing3_4", "Bearing3_5",
]
SEEDS = [42, 123, 2024]
SEQUENCE_LENGTH = 10
KNOT_QUANTILES = [0.0, 1 / 3, 2 / 3, 1.0]
KEYS = ["bearing_id", "start_position", "end_position"]


def condition_for_bearing(bearing_id: str) -> str:
    if bearing_id.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing_id.startswith("Bearing2_"):
        return "37.5Hz11kN"
    if bearing_id.startswith("Bearing3_"):
        return "40Hz10kN"
    raise ValueError(f"Unknown bearing ID: {bearing_id}")


def sequence_targets(frame: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    records = []
    for bearing in bearings:
        group = frame.loc[frame["bearing_id"].eq(bearing)].sort_values("chronological_position")
        if group.empty:
            raise ValueError(f"No metadata rows for {bearing}")
        positions = group["chronological_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(group))):
            raise ValueError(f"Nonconsecutive chronological positions in {bearing}")
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            row = group.iloc[end]
            records.append({
                "bearing_id": bearing,
                "condition": condition_for_bearing(bearing),
                "start_position": end - SEQUENCE_LENGTH + 1,
                "end_position": end,
                "true_rul": float(row["rul"]),
                "fault_label": row["fault_label"],
                "recording_index_end": int(row["recording_index"]),
            })
    return pd.DataFrame.from_records(records)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def equal_bearing_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("bearing_id")["bearing_id"].transform("size").to_numpy(float)
    return 1.0 / counts


def weighted_empirical_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    index = int(np.searchsorted(np.cumsum(weights), quantile * weights.sum(), side="left"))
    return float(values[min(index, len(values) - 1)])


def interpolation_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), knots[0], knots[-1])
    basis = np.zeros((len(clipped), len(knots)), dtype=np.float64)
    interval = np.searchsorted(knots, clipped, side="right") - 1
    interval = np.clip(interval, 0, len(knots) - 2)
    left, right = knots[interval], knots[interval + 1]
    right_weight = (clipped - left) / (right - left)
    rows = np.arange(len(clipped))
    basis[rows, interval] = 1.0 - right_weight
    basis[rows, interval + 1] = right_weight
    return basis


def verify_against_neural_rows(expected: pd.DataFrame) -> None:
    expected_sorted = expected.sort_values(KEYS).reset_index(drop=True)
    if expected_sorted.duplicated(KEYS).any():
        raise AssertionError("Metadata sequence keys are not unique")
    for model in ["m1", "m2", "m3", "m4", "m5"]:
        for seed in SEEDS:
            path = MULTISEED / model / f"seed_{seed}" / "predictions.csv"
            neural = pd.read_csv(path)
            if neural.duplicated(KEYS).any():
                raise AssertionError(f"Duplicate sequence keys in {path}")
            joined = expected_sorted.merge(
                neural[KEYS + ["true_rul"]].sort_values(KEYS),
                on=KEYS,
                how="outer",
                suffixes=("_metadata", "_neural"),
                indicator=True,
                validate="one_to_one",
            )
            if len(joined) != len(expected_sorted) or not joined["_merge"].eq("both").all():
                raise AssertionError(f"Test sequence rows differ from neural predictions: {path}")
            if not np.allclose(joined.true_rul_metadata, joined.true_rul_neural, rtol=0, atol=1e-7):
                raise AssertionError(f"Test labels differ from metadata: {path}")
            if not neural["bearing_id"].isin(TEST_BEARINGS).all():
                raise AssertionError(f"Non-test bearing in prediction file: {path}")
            if not ((neural.end_position - neural.start_position) == SEQUENCE_LENGTH - 1).all():
                raise AssertionError(f"Unexpected sequence length in {path}")


def summarize_predictions(name: str, frame: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    pooled = regression_metrics(frame.true_rul.to_numpy(), frame.pred_rul.to_numpy())
    bearing_rows = []
    for bearing, group in frame.groupby("bearing_id", sort=True):
        bearing_rows.append({
            "baseline": name,
            "bearing_id": bearing,
            "condition": group.condition.iloc[0],
            "sequence_count": int(len(group)),
            **regression_metrics(group.true_rul.to_numpy(), group.pred_rul.to_numpy()),
            "mean_true_rul": float(group.true_rul.mean()),
            "mean_predicted_rul": float(group.pred_rul.mean()),
        })
    condition_rows = []
    for condition, group in frame.groupby("condition", sort=True):
        condition_rows.append({
            "baseline": name,
            "condition": condition,
            "sequence_count": int(len(group)),
            **regression_metrics(group.true_rul.to_numpy(), group.pred_rul.to_numpy()),
        })
    return {
        "baseline": name,
        "test_sequence_count": int(len(frame)),
        "test_bearing_count": int(frame.bearing_id.nunique()),
        "prediction_min": float(frame.pred_rul.min()),
        "prediction_max": float(frame.pred_rul.max()),
        "pooled_window_mae": pooled["mae"],
        "pooled_window_rmse": pooled["rmse"],
        "pooled_window_r2": pooled["r2"],
        "macro_bearing_mae": float(np.mean([row["mae"] for row in bearing_rows])),
        "macro_bearing_rmse": float(np.mean([row["rmse"] for row in bearing_rows])),
    }, bearing_rows, condition_rows


def seed_mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def load_neural_comparisons(expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(FINAL_SUMMARY)
    overall = []
    macro = []
    conditions = []
    for model in ["M1", "M2", "M3", "M4", "M5"]:
        per_seed_metrics = []
        per_seed_bearings = []
        for seed in SEEDS:
            seed_dir = MULTISEED / model.lower() / f"seed_{seed}"
            metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
            per_seed_metrics.append(metrics)
            pred = pd.read_csv(seed_dir / "predictions.csv")
            joined = expected[KEYS + ["true_rul", "condition"]].merge(
                pred[KEYS + ["pred_rul"]], on=KEYS, how="inner", validate="one_to_one"
            )
            bmae, brmse = [], []
            for _, group in joined.groupby("bearing_id", sort=True):
                vals = regression_metrics(group.true_rul.to_numpy(), group.pred_rul.to_numpy())
                bmae.append(vals["mae"]); brmse.append(vals["rmse"])
            per_seed_bearings.append((float(np.mean(bmae)), float(np.mean(brmse))))

        pooled = final.loc[final.Model.eq(model)].iloc[0]
        overall.append({
            "model": model,
            "mae_mean": float(str(pooled["MAE mean±std"]).split("+/-")[0].strip()),
            "mae_sd": float(str(pooled["MAE mean±std"]).split("+/-")[1].strip()),
            "rmse_mean": float(str(pooled["RMSE mean±std"]).split("+/-")[0].strip()),
            "rmse_sd": float(str(pooled["RMSE mean±std"]).split("+/-")[1].strip()),
            "r2_mean": float(str(pooled["R2 mean±std"]).split("+/-")[0].strip()),
            "r2_sd": float(str(pooled["R2 mean±std"]).split("+/-")[1].strip()),
        })
        # Cross-check aggregate rows against the underlying saved seed metrics.
        for key, col in [("MAE", "mae"), ("RMSE", "rmse"), ("R2", "r2")]:
            got_mean, got_sd = seed_mean_std([float(item[col]) for item in per_seed_metrics])
            saved_mean = float(str(pooled[f"{key} mean±std"]).split("+/-")[0].strip())
            saved_sd = float(str(pooled[f"{key} mean±std"]).split("+/-")[1].strip())
            if abs(got_mean - saved_mean) > 2e-6 or abs(got_sd - saved_sd) > 2e-6:
                raise AssertionError(f"Final summary does not match per-seed {model} {key}")

        macro_mae_mean, macro_mae_sd = seed_mean_std([item[0] for item in per_seed_bearings])
        macro_rmse_mean, macro_rmse_sd = seed_mean_std([item[1] for item in per_seed_bearings])
        macro.append({"model": model, "macro_bearing_mae_mean": macro_mae_mean, "macro_bearing_mae_sd": macro_mae_sd,
                      "macro_bearing_rmse_mean": macro_rmse_mean, "macro_bearing_rmse_sd": macro_rmse_sd})
        for condition in sorted(expected.condition.unique()):
            condition_values = []
            for seed in SEEDS:
                seed_dir = MULTISEED / model.lower() / f"seed_{seed}"
                metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
                entry = next(item for item in metrics["per_condition"] if item["condition"] == condition)
                condition_values.append(entry)
            mean_mae, sd_mae = seed_mean_std([float(item["mae"]) for item in condition_values])
            mean_rmse, sd_rmse = seed_mean_std([float(item["rmse"]) for item in condition_values])
            conditions.append({"model": model, "condition": condition, "mae_mean": mean_mae, "mae_sd": sd_mae,
                               "rmse_mean": mean_rmse, "rmse_sd": sd_rmse})
    return pd.DataFrame(overall), pd.DataFrame(macro), pd.DataFrame(conditions)


def fmt(mean: float, sd: float | None = None) -> str:
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {sd:.3f}"


def write_markdown(metrics: pd.DataFrame, per_bearing: pd.DataFrame, per_condition: pd.DataFrame,
                   neural_overall: pd.DataFrame, neural_macro: pd.DataFrame, neural_condition: pd.DataFrame,
                   train_mean: float, knots: np.ndarray, coefficients: np.ndarray) -> None:
    lines = [
        "# Offline RUL Baseline Comparison", "",
        "Baselines are evaluated on the same 3,164 fixed test sequence rows as the saved M1-M5 predictions. Neural values are three-seed mean +/- sample SD; each baseline is one deterministic fit.", "",
        "## Pooled Window-Weighted Metrics", "",
        "Every test sequence receives equal weight; overlapping windows are not independent bearing samples.", "",
        "| Method | MAE | RMSE | R-squared |", "| --- | ---: | ---: | ---: |",
    ]
    for _, row in metrics.iterrows():
        lines.append(f"| {row.baseline} | {fmt(row.pooled_window_mae)} | {fmt(row.pooled_window_rmse)} | {fmt(row.pooled_window_r2)} |")
    for _, row in neural_overall.iterrows():
        lines.append(f"| {row.model} (3-seed mean +/- sample SD) | {fmt(row.mae_mean, row.mae_sd)} | {fmt(row.rmse_mean, row.rmse_sd)} | {fmt(row.r2_mean, row.r2_sd)} |")
    lines += ["", "## Equal-Bearing Macro Errors", "", "Each test bearing receives equal weight after calculating its metrics separately. The neural standard deviation is across seeds of the nine-bearing macro metric.", "",
              "| Method | Macro MAE | Macro RMSE |", "| --- | ---: | ---: |"]
    for _, row in metrics.iterrows():
        lines.append(f"| {row.baseline} | {fmt(row.macro_bearing_mae)} | {fmt(row.macro_bearing_rmse)} |")
    for _, row in neural_macro.iterrows():
        lines.append(f"| {row.model} (3-seed mean +/- sample SD) | {fmt(row.macro_bearing_mae_mean, row.macro_bearing_mae_sd)} | {fmt(row.macro_bearing_rmse_mean, row.macro_bearing_rmse_sd)} |")
    lines += ["", "## Per-Condition MAE", "", "| Condition | Method | MAE | RMSE |", "| --- | --- | ---: | ---: |"]
    for _, row in per_condition.sort_values(["condition", "baseline"]).iterrows():
        lines.append(f"| {row.condition} | {row.baseline} | {fmt(row.mae)} | {fmt(row.rmse)} |")
    for _, row in neural_condition.sort_values(["condition", "model"]).iterrows():
        lines.append(f"| {row.condition} | {row.model} (3-seed mean +/- sample SD) | {fmt(row.mae_mean, row.mae_sd)} | {fmt(row.rmse_mean, row.rmse_sd)} |")
    best_neural_pooled = neural_overall.sort_values("mae_mean").iloc[0]
    best_neural_macro = neural_macro.sort_values("macro_bearing_mae_mean").iloc[0]
    lines += ["", "## Per-Bearing Metrics", "", "See `per_bearing_metrics.csv` for baseline MAE, RMSE, R-squared, and the corresponding sequence counts for every bearing.", "",
              "## Interpretation", "",
              f"Both simple baselines have lower pooled window MAE/RMSE than the best neural mean (M3: {best_neural_pooled.mae_mean:.3f}/{best_neural_pooled.rmse_mean:.3f}). The training-mean baseline is {metrics.iloc[0].pooled_window_mae:.3f}/{metrics.iloc[0].pooled_window_rmse:.3f}; the piecewise age baseline is {metrics.iloc[1].pooled_window_mae:.3f}/{metrics.iloc[1].pooled_window_rmse:.3f}. Under equal-bearing macro MAE, the lowest neural mean is {best_neural_macro.model} ({best_neural_macro.macro_bearing_mae_mean:.3f}), compared with {metrics.iloc[0].macro_bearing_mae:.3f} for the training-mean baseline and {metrics.iloc[1].macro_bearing_mae:.3f} for the piecewise age baseline. The ranking therefore depends on whether windows or bearings receive equal weight.",
              "", "The comparison is descriptive. The baseline fits use training labels only, and no baseline result was used to choose a model, knot count, or validation/test protocol. Differences among overlapping test windows do not represent independent bearing-level replication.", "",
              f"Training-mean estimate (train sequence targets only): `{train_mean:.9f}`.",
              f"Piecewise-linear age knots (training positions only): `{', '.join(f'{value:.1f}' for value in knots)}`.",
              f"Fitted knot values before output clipping: `{', '.join(f'{value:.6f}' for value in coefficients)}`.", ""]
    (OUT / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = pd.read_csv(META_PATH).sort_values(["bearing_id", "chronological_position"]).reset_index(drop=True)
    train = sequence_targets(frame, TRAIN_BEARINGS)
    test = sequence_targets(frame, TEST_BEARINGS)
    if set(frame.loc[frame.bearing_id.isin(TRAIN_BEARINGS), "bearing_id"].unique()) != set(TRAIN_BEARINGS):
        raise AssertionError("Training fit bearings differ from the fixed split")
    if VALIDATION_BEARING in TRAIN_BEARINGS or VALIDATION_BEARING in TEST_BEARINGS:
        raise AssertionError("Validation bearing overlaps another partition")
    if len(test) != 3164 or test.bearing_id.nunique() != 9:
        raise AssertionError(f"Unexpected test sequence rows: {len(test)} rows, {test.bearing_id.nunique()} bearings")
    if not test.true_rul.between(0, 1).all() or not train.true_rul.between(0, 1).all():
        raise AssertionError("RUL labels out of expected range")
    verify_against_neural_rows(test)

    train_mean = float(train.true_rul.mean())
    pred_mean = test.copy()
    pred_mean["pred_rul"] = train_mean

    age_weights = equal_bearing_weights(train)
    age_x = train.end_position.to_numpy(dtype=np.float64)
    age_y = train.true_rul.to_numpy(dtype=np.float64)
    knots = np.asarray([
        weighted_empirical_quantile(age_x, age_weights, quantile)
        for quantile in KNOT_QUANTILES
    ], dtype=np.float64)
    if len(np.unique(knots)) != len(knots):
        raise AssertionError(f"Training-only knot quantiles are not distinct: {knots}")
    design = interpolation_basis(age_x, knots)
    root_weight = np.sqrt(age_weights)
    coefficients, *_ = np.linalg.lstsq(design * root_weight[:, None], age_y * root_weight, rcond=None)
    pred_age = test.copy()
    raw_predictions = interpolation_basis(test.end_position.to_numpy(float), knots) @ coefficients
    pred_age["pred_rul"] = np.clip(raw_predictions, 0.0, 1.0)

    metric_rows, bearing_rows, condition_rows = [], [], []
    for name, predictions in [("training_mean", pred_mean), ("piecewise_linear_age", pred_age)]:
        pooled, per_bearing, per_condition = summarize_predictions(name, predictions)
        pooled["training_target_mean"] = train_mean
        pooled["piecewise_knots"] = ";".join(f"{value:.6f}" for value in knots)
        metric_rows.append(pooled)
        bearing_rows.extend(per_bearing)
        condition_rows.extend(per_condition)

    metrics_df = pd.DataFrame(metric_rows)
    bearings_df = pd.DataFrame(bearing_rows)
    conditions_df = pd.DataFrame(condition_rows)
    neural_overall, neural_macro, neural_condition = load_neural_comparisons(test)
    metrics_df.to_csv(OUT / "baseline_metrics.csv", index=False)
    bearings_df.to_csv(OUT / "per_bearing_metrics.csv", index=False)
    conditions_df.to_csv(OUT / "per_condition_metrics.csv", index=False)
    write_markdown(metrics_df, bearings_df, conditions_df, neural_overall, neural_macro, neural_condition,
                   train_mean, knots, coefficients)
    print(metrics_df[["baseline", "pooled_window_mae", "pooled_window_rmse", "pooled_window_r2",
                      "macro_bearing_mae", "macro_bearing_rmse"]].to_string(index=False))
    print(f"Rows verified against 15 neural prediction files: {len(test)}")


if __name__ == "__main__":
    main()
