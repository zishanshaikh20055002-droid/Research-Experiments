"""Build descriptive baseline and three-seed neural comparison tables.

This script reads existing outputs only; it never trains models or alters them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MULTISEED = ROOT / "results" / "multiseed"
FINAL = ROOT / "results" / "final_experiment_summary" / "final_model_comparison.csv"
SEEDS = (42, 123, 2024)
MODELS = ("M1", "M2", "M3", "M4", "M5")
KEYS = ["bearing_id", "start_position", "end_position"]


def metrics(y: pd.Series, prediction: pd.Series) -> dict[str, float]:
    error = prediction.to_numpy(float) - y.to_numpy(float)
    target = y.to_numpy(float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2)),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def show(mean: float, standard_deviation: float | None) -> str:
    return f"{mean:.3f}" if standard_deviation is None else f"{mean:.3f} +/- {standard_deviation:.3f}"


def main() -> None:
    baseline_overall = pd.read_csv(OUT / "baseline_metrics.csv")
    baseline_bearing = pd.read_csv(OUT / "per_bearing_metrics.csv")
    baseline_condition = pd.read_csv(OUT / "per_condition_metrics.csv")
    summary = pd.read_csv(FINAL)
    rows: list[dict] = []

    # Deterministic references have no seed dispersion.
    for _, item in baseline_overall.iterrows():
        method = "Training mean" if item.baseline == "training_mean" else "Piecewise-linear age"
        for scope, metric, value in (("pooled", "MAE", item.pooled_window_mae), ("pooled", "RMSE", item.pooled_window_rmse),
                                     ("pooled", "R2", item.pooled_window_r2), ("equal_bearing_macro", "MAE", item.macro_bearing_mae),
                                     ("equal_bearing_macro", "RMSE", item.macro_bearing_rmse)):
            rows.append({"scope": scope, "entity": "all", "method": method,
                         "method_type": "deterministic baseline", "metric": metric, "mean": value, "sample_std": np.nan,
                         "n_seeds": 1})
    for _, item in baseline_bearing.iterrows():
        method = "Training mean" if item.baseline == "training_mean" else "Piecewise-linear age"
        for metric in ("mae", "rmse"):
            rows.append({"scope": "per_bearing", "entity": item.bearing_id, "method": method, "method_type": "deterministic baseline",
                         "metric": metric.upper(), "mean": item[metric], "sample_std": np.nan, "n_seeds": 1})
    for _, item in baseline_condition.iterrows():
        method = "Training mean" if item.baseline == "training_mean" else "Piecewise-linear age"
        rows.append({"scope": "per_condition", "entity": item.condition, "method": method, "method_type": "deterministic baseline",
                     "metric": "MAE", "mean": item.mae, "sample_std": np.nan, "n_seeds": 1})

    neural_metrics: dict[tuple[str, str, str], list[float]] = {}
    for model in MODELS:
        for seed in SEEDS:
            prediction = pd.read_csv(MULTISEED / model.lower() / f"seed_{seed}" / "predictions.csv")
            for bearing, group in prediction.groupby("bearing_id", sort=True):
                result = metrics(group.true_rul, group.pred_rul)
                for metric, value in result.items():
                    neural_metrics.setdefault((model, "per_bearing", bearing, metric), []).append(value)
            for condition, group in prediction.groupby("condition", sort=True):
                result = metrics(group.true_rul, group.pred_rul)
                neural_metrics.setdefault((model, "per_condition", condition, "mae"), []).append(result["mae"])
            bearing_results = [metrics(group.true_rul, group.pred_rul) for _, group in prediction.groupby("bearing_id", sort=True)]
            neural_metrics.setdefault((model, "equal_bearing_macro", "all", "mae"), []).append(float(np.mean([x["mae"] for x in bearing_results])))
            neural_metrics.setdefault((model, "equal_bearing_macro", "all", "rmse"), []).append(float(np.mean([x["rmse"] for x in bearing_results])))

    for _, item in summary.iterrows():
        model = item.Model
        for label, field in (("MAE", "MAE mean±std"), ("RMSE", "RMSE mean±std"), ("R2", "R2 mean±std")):
            mean, sd = (float(x.strip()) for x in str(item[field]).split("+/-"))
            rows.append({"scope": "pooled", "entity": "all", "method": model, "method_type": "neural (three seeds)",
                         "metric": label, "mean": mean, "sample_std": sd, "n_seeds": 3})
    for (model, scope, entity, metric), values in neural_metrics.items():
        mean, sd = mean_sd(values)
        rows.append({"scope": scope, "entity": entity, "method": model, "method_type": "neural (three seeds)",
                     "metric": metric.upper(), "mean": mean, "sample_std": sd, "n_seeds": 3})

    table = pd.DataFrame(rows).sort_values(["scope", "entity", "metric", "method"])
    table.to_csv(OUT / "unified_comparison.csv", index=False)

    def subset(scope: str, metric: str, entity: str = "all") -> pd.DataFrame:
        return table[(table.scope == scope) & (table.metric == metric) & (table.entity == entity)]

    order = ["Training mean", "Piecewise-linear age", *MODELS]
    lines = [
        "# Unified Baseline and Neural Comparison", "",
        "All values use the fixed 3,164 test sequence rows. Baselines are one deterministic fit; neural values are mean +/- sample standard deviation across the three saved seeds. Pooled window metrics weight each overlapping sequence equally. Equal-bearing macro metrics calculate a metric per test bearing and then average the nine values equally; they are not independent-sample inference.", "",
        "## Pooled Window-Weighted Metrics", "", "| Method | MAE | RMSE | R-squared |", "| --- | ---: | ---: | ---: |",
    ]
    for method in order:
        values = []
        for metric in ("MAE", "RMSE", "R2"):
            value = subset("pooled", metric).query("method == @method").iloc[0]
            values.append(show(value["mean"], None if pd.isna(value.sample_std) else value.sample_std))
        lines.append(f"| {method} | {values[0]} | {values[1]} | {values[2]} |")
    lines += ["", "## Equal-Bearing Macro Errors", "", "| Method | Macro MAE | Macro RMSE |", "| --- | ---: | ---: |"]
    for method in order:
        values = []
        for metric in ("MAE", "RMSE"):
            value = subset("equal_bearing_macro", metric).query("method == @method").iloc[0]
            values.append(show(value["mean"], None if pd.isna(value.sample_std) else value.sample_std))
        lines.append(f"| {method} | {values[0]} | {values[1]} |")
    lines += ["", "## Per-Bearing Metrics", "", "| Bearing | Method | MAE | RMSE |", "| --- | --- | ---: | ---: |"]
    for bearing in sorted(table.loc[table.scope.eq("per_bearing"), "entity"].unique()):
        for method in order:
            mae = subset("per_bearing", "MAE", bearing).query("method == @method").iloc[0]
            rmse = subset("per_bearing", "RMSE", bearing).query("method == @method").iloc[0]
            lines.append(f"| {bearing} | {method} | {show(mae['mean'], None if pd.isna(mae.sample_std) else mae.sample_std)} | {show(rmse['mean'], None if pd.isna(rmse.sample_std) else rmse.sample_std)} |")
    lines += ["", "## Per-Condition MAE", "", "| Condition | Method | MAE |", "| --- | --- | ---: |"]
    for condition in sorted(table.loc[table.scope.eq("per_condition"), "entity"].unique()):
        for method in order:
            value = subset("per_condition", "MAE", condition).query("method == @method").iloc[0]
            lines.append(f"| {condition} | {method} | {show(value['mean'], None if pd.isna(value.sample_std) else value.sample_std)} |")
    lines += ["", "The piecewise-linear age baseline is intentionally unconstrained. Its fitted knot values include an increasing interval, so it is not a monotonic baseline.", ""]
    (OUT / "unified_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
