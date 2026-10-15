"""Final-pass M4 aggregation and offline trend/smoothness summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / "results" / "multiseed"
FINAL = ROOT / "results" / "final_experiment_summary"
MODELS = ["m1", "m2", "m3", "m4", "m5"]
SEEDS = [42, 123, 2024]


def mean_std(values: pd.Series | list[float]) -> tuple[float | None, float | None, str]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None, None, "N/A"
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return mean, std, f"{mean:.6f} +/- {std:.6f}"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in display.columns) + " |")
    return "\n".join(lines)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def condition_for_bearing(bearing: str) -> str:
    if bearing.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing.startswith("Bearing2_"):
        return "37.5Hz11kN"
    return "40Hz10kN"


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(y) == 0:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(float)
    yr = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def adjacent_stats(group: pd.DataFrame) -> dict[str, float | int]:
    pred = group.sort_values("end_position")["pred_rul"].to_numpy(float)
    diffs = pred[1:] - pred[:-1]
    violations = diffs[diffs > 0]
    return {
        "valid_adjacent_transitions": int(len(diffs)),
        "violating_transitions": int(len(violations)),
        "violation_rate": float(len(violations) / len(diffs)) if len(diffs) else 0.0,
        "mean_violation_magnitude": float(np.mean(violations)) if len(violations) else 0.0,
        "mean_abs_adjacent_change": float(np.mean(np.abs(diffs))) if len(diffs) else float("nan"),
    }


def load_prediction(model: str, seed: int) -> pd.DataFrame:
    path = MULTI / model / f"seed_{seed}" / "predictions.csv"
    frame = pd.read_csv(path)
    if "condition" not in frame.columns:
        frame["condition"] = frame["bearing_id"].map(condition_for_bearing)
    return frame


def load_metric(model: str, seed: int) -> dict[str, Any]:
    return json.loads((MULTI / model / f"seed_{seed}" / "metrics.json").read_text())


def build_seed_metrics() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            data = load_metric(model, seed)
            rows.append({k: v for k, v in data.items() if k != "per_condition" and not isinstance(v, dict)})
    return pd.DataFrame(rows)


def build_condition_metrics() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            data = load_metric(model, seed)
            for item in data["per_condition"]:
                rows.append({"model": model.upper(), "seed": seed, **item})
    return pd.DataFrame(rows)


def build_trend_smoothness() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    bearing_rows = []
    condition_rows = []
    for model in MODELS:
        for seed in SEEDS:
            frame = load_prediction(model, seed)
            all_diffs = []
            all_violations = []
            bearing_spearman = []
            bearing_smoothness = []
            for bearing, group in frame.groupby("bearing_id", sort=True):
                ordered = group.sort_values("end_position")
                pred = ordered["pred_rul"].to_numpy(float)
                positions = ordered["end_position"].to_numpy(float)
                rho = spearman_rho(positions, pred)
                adj = adjacent_stats(ordered)
                diffs = pred[1:] - pred[:-1]
                all_diffs.extend(diffs.tolist())
                all_violations.extend(diffs[diffs > 0].tolist())
                bearing_spearman.append(rho)
                bearing_smoothness.append(adj["mean_abs_adjacent_change"])
                bearing_rows.append(
                    {
                        "model": model.upper(),
                        "seed": seed,
                        "bearing_id": bearing,
                        "condition": condition_for_bearing(bearing),
                        "spearman_rho": rho,
                        **adj,
                    }
                )
            for condition, group in frame.groupby("condition", sort=True):
                cond_spearman = []
                cond_smooth = []
                cond_diffs = []
                cond_violations = []
                for _, bearing_group in group.groupby("bearing_id", sort=True):
                    ordered = bearing_group.sort_values("end_position")
                    pred = ordered["pred_rul"].to_numpy(float)
                    diffs = pred[1:] - pred[:-1]
                    cond_diffs.extend(diffs.tolist())
                    cond_violations.extend(diffs[diffs > 0].tolist())
                    cond_spearman.append(spearman_rho(ordered["end_position"].to_numpy(float), pred))
                    cond_smooth.append(float(np.mean(np.abs(diffs))) if len(diffs) else float("nan"))
                condition_rows.append(
                    {
                        "model": model.upper(),
                        "seed": seed,
                        "condition": condition,
                        "spearman_mean_across_bearings": float(np.nanmean(cond_spearman)),
                        "spearman_median_across_bearings": float(np.nanmedian(cond_spearman)),
                        "spearman_std_across_bearings": float(np.nanstd(cond_spearman, ddof=1)) if len(cond_spearman) > 1 else 0.0,
                        "mean_abs_adjacent_change": float(np.nanmean(cond_smooth)),
                        "median_abs_adjacent_change": float(np.nanmedian(cond_smooth)),
                        "std_abs_adjacent_change": float(np.nanstd(cond_smooth, ddof=1)) if len(cond_smooth) > 1 else 0.0,
                        "violation_rate": float(len(cond_violations) / len(cond_diffs)) if cond_diffs else 0.0,
                        "mean_violation_magnitude": float(np.mean(cond_violations)) if cond_violations else 0.0,
                    }
                )
            overall_rows.append(
                {
                    "model": model.upper(),
                    "seed": seed,
                    "violation_rate": float(len(all_violations) / len(all_diffs)) if all_diffs else 0.0,
                    "mean_violation_magnitude": float(np.mean(all_violations)) if all_violations else 0.0,
                    "spearman_mean_across_bearings": float(np.nanmean(bearing_spearman)),
                    "spearman_median_across_bearings": float(np.nanmedian(bearing_spearman)),
                    "spearman_std_across_bearings": float(np.nanstd(bearing_spearman, ddof=1)),
                    "mean_abs_adjacent_change": float(np.nanmean(bearing_smoothness)),
                    "median_abs_adjacent_change": float(np.nanmedian(bearing_smoothness)),
                    "std_abs_adjacent_change": float(np.nanstd(bearing_smoothness, ddof=1)),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(bearing_rows), pd.DataFrame(condition_rows)


def aggregate_seed_metrics(seed_df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "MAE": "mae",
        "RMSE": "rmse",
        "R2": "r2",
        "Macro-F1": "fault_macro_f1",
        "Mono violation": "mono_violation_rate",
        "Prediction bias": "mean_prediction_bias_pred_minus_true",
        "s_R": "s_R",
        "s_F": "s_F",
    }
    rows = []
    for model, group in seed_df.groupby("model", sort=False):
        row = {"Model": model}
        for label, col in metric_map.items():
            _, _, text = mean_std(group[col]) if col in group else (None, None, "N/A")
            mean, std, _ = mean_std(group[col]) if col in group else (None, None, "N/A")
            row[f"{label} mean"] = mean
            row[f"{label} std"] = std
            row[f"{label} mean±std"] = text
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_condition(condition_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), group in condition_df.groupby(["model", "condition"], sort=False):
        row = {"Model": model, "condition": condition}
        for label, col in {"MAE": "mae", "RMSE": "rmse", "R2": "r2"}.items():
            mean, std, text = mean_std(group[col])
            row[f"{label} mean"] = mean
            row[f"{label} std"] = std
            row[f"{label} mean±std"] = text
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_trend(trend_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = {
        "Violation Rate": "violation_rate",
        "Mean Violation Magnitude": "mean_violation_magnitude",
        "Spearman Trend rho": "spearman_mean_across_bearings",
        "Mean Adjacent Change": "mean_abs_adjacent_change",
    }
    for model, group in trend_df.groupby("model", sort=False):
        row = {"Model": model}
        for label, col in cols.items():
            mean, std, text = mean_std(group[col])
            row[f"{label} mean"] = mean
            row[f"{label} std"] = std
            row[f"{label} mean±std"] = text
        rows.append(row)
    return pd.DataFrame(rows)


def write_m4_aggregate(seed_df: pd.DataFrame, condition_df: pd.DataFrame) -> None:
    m4 = seed_df[seed_df["model"] == "M4"].copy()
    agg = aggregate_seed_metrics(m4)
    agg.to_csv(MULTI / "m4_aggregate_metrics.csv", index=False)
    table = agg[["Model", "MAE mean±std", "RMSE mean±std", "R2 mean±std", "Macro-F1 mean±std", "Mono violation mean±std", "s_R mean±std", "s_F mean±std"]]
    (MULTI / "m4_aggregate_metrics.md").write_text("# M4 Aggregate Metrics\n\n" + markdown_table(table) + "\n", encoding="utf-8")

    m4_cond = aggregate_condition(condition_df[condition_df["model"] == "M4"])
    m4_cond.to_csv(MULTI / "m4_per_condition_aggregate.csv", index=False)


def write_final_files(seed_df: pd.DataFrame, condition_df: pd.DataFrame, trend_df: pd.DataFrame, bearing_df: pd.DataFrame, trend_condition_df: pd.DataFrame) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    agg = aggregate_seed_metrics(seed_df)
    final_model = agg[["Model", "MAE mean±std", "RMSE mean±std", "R2 mean±std", "Macro-F1 mean±std", "Mono violation mean±std"]]
    final_model.to_csv(FINAL / "final_model_comparison.csv", index=False)
    (FINAL / "final_model_comparison.md").write_text("# Final Model Comparison\n\n" + markdown_table(final_model) + "\n", encoding="utf-8")

    cond = aggregate_condition(condition_df)
    cond.to_csv(FINAL / "final_condition_comparison.csv", index=False)

    trend_agg = aggregate_trend(trend_df)
    trend_table = trend_agg[["Model", "Violation Rate mean±std", "Mean Violation Magnitude mean±std", "Spearman Trend rho mean±std", "Mean Adjacent Change mean±std"]]
    trend_table.to_csv(FINAL / "final_trend_smoothness_comparison.csv", index=False)
    bearing_df.to_csv(FINAL / "final_trend_smoothness_per_bearing.csv", index=False)
    trend_condition_df.to_csv(FINAL / "final_trend_smoothness_per_condition.csv", index=False)

    best_mae = agg.sort_values("MAE mean").iloc[0]["Model"]
    best_rmse = agg.sort_values("RMSE mean").iloc[0]["Model"]
    best_r2 = agg.sort_values("R2 mean", ascending=False).iloc[0]["Model"]
    best_f1 = agg.dropna(subset=["Macro-F1 mean"]).sort_values("Macro-F1 mean", ascending=False).iloc[0]["Model"]
    lowest_mono = agg.sort_values("Mono violation mean").iloc[0]["Model"]
    best_spearman = trend_agg.sort_values("Spearman Trend rho mean").iloc[0]["Model"]
    lowest_smooth = trend_agg.sort_values("Mean Adjacent Change mean").iloc[0]["Model"]

    seed_cols = ["model", "seed", "mae", "rmse", "r2", "fault_macro_f1", "mono_violation_rate", "s_R", "s_F", "best_epoch", "training_time_seconds"]
    summary = [
        "# Final Seed Summary",
        "",
        "## Main Comparison",
        "",
        markdown_table(final_model),
        "",
        "## Trend and Smoothness",
        "",
        markdown_table(trend_table),
        "",
        "## Seed-Wise Metrics",
        "",
        markdown_table(seed_df[[c for c in seed_cols if c in seed_df.columns]]),
        "",
        f"- Best mean MAE: `{best_mae}`",
        f"- Best mean RMSE: `{best_rmse}`",
        f"- Best mean R2: `{best_r2}`",
        f"- Best mean Macro-F1: `{best_f1}`",
        f"- Lowest mean monotonicity violation rate: `{lowest_mono}`",
        f"- Best mean Spearman trend rho: `{best_spearman}`",
        f"- Lowest mean adjacent change: `{lowest_smooth}`",
    ]
    (FINAL / "final_seed_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    m2 = agg[agg["Model"] == "M2"].iloc[0]
    m3 = agg[agg["Model"] == "M3"].iloc[0]
    m4 = agg[agg["Model"] == "M4"].iloc[0]
    m5 = agg[agg["Model"] == "M5"].iloc[0]
    interpretation = [
        "# Final Interpretation",
        "",
        "These summaries use the fixed selected validation fold and three requested seeds only. No statistical significance test was run.",
        "",
        f"M4 does not add clear RUL value beyond M2 in this pass: M4 mean MAE is {m4['MAE mean']:.6f} versus M2 {m2['MAE mean']:.6f}, and M4 mean RMSE is {m4['RMSE mean']:.6f} versus M2 {m2['RMSE mean']:.6f}. M4 has a higher mean Macro-F1 than M2, but this should be treated cautiously because the selected fold has no Cage-class training bearing.",
        "",
        f"M5 improves RUL metrics relative to M4 in this pass: M5 mean MAE is {m5['MAE mean']:.6f} versus M4 {m4['MAE mean']:.6f}, while M5 mean RMSE is {m5['RMSE mean']:.6f} versus M4 {m4['RMSE mean']:.6f}. This does not make M5 the preferred overall model because M5 has lower mean Macro-F1 than M4 and worse strict monotonicity and adjacent-change diagnostics.",
        "",
        f"M3 remains the strongest RUL model by mean MAE ({m3['MAE mean']:.6f}), mean RMSE ({m3['RMSE mean']:.6f}), and mean R2 ({m3['R2 mean']:.6f}) among these five models.",
        "",
        f"Strict monotonicity metrics do not improve with the monotonicity-regularized models in these saved test predictions. The lowest mean violation rate is {lowest_mono}, so the monotonicity term should not be described as reducing test-time violations here.",
        "",
        "Spearman trend correlation is a trend-following diagnostic, not proof of physics consistency. It can give a different view than strict adjacent violation rate because it measures the overall ranking of predictions over time rather than every local increase.",
        "",
        f"The lowest mean adjacent prediction change is {lowest_smooth}. Use this as a smoothness diagnostic only; smoother predictions are not automatically more accurate.",
        "",
        "Fault Macro-F1 remains unreliable as a primary claim because the selected training fold contains no Cage-class bearing. Per-class fault conclusions should therefore be conservative.",
        "",
        "Cross-condition performance should be discussed carefully, especially for the 40Hz10kN condition where earlier diagnostics showed the hardest generalization behavior.",
        "",
        "Seed variability is material, particularly for M3 Macro-F1 and R2. The paper should emphasize mean +/- sample standard deviation rather than a single best seed.",
    ]
    (FINAL / "final_interpretation.md").write_text("\n".join(interpretation) + "\n", encoding="utf-8")

    summary_json = {
        "best_mean_mae": best_mae,
        "best_mean_rmse": best_rmse,
        "best_mean_r2": best_r2,
        "best_mean_macro_f1": best_f1,
        "lowest_mean_monotonicity_violation": lowest_mono,
        "best_mean_spearman_trend": best_spearman,
        "lowest_mean_adjacent_change": lowest_smooth,
    }
    (FINAL / "final_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def main() -> int:
    seed_df = build_seed_metrics()
    condition_df = build_condition_metrics()
    trend_df, bearing_df, trend_condition_df = build_trend_smoothness()
    write_m4_aggregate(seed_df, condition_df)
    write_final_files(seed_df, condition_df, trend_df, bearing_df, trend_condition_df)
    print(markdown_table(pd.read_csv(FINAL / "final_model_comparison.csv")))
    print()
    print(markdown_table(pd.read_csv(FINAL / "final_trend_smoothness_comparison.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
