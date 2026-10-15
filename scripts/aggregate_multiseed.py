"""Aggregate M1/M2/M3/M5 3-seed robustness results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "multiseed"
MODELS = ["m1", "m2", "m3", "m5"]
SEEDS = [42, 123, 2024]


def mean_std(series: pd.Series):
    values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(values) == 0:
        return None, None, "N/A"
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std, f"{mean:.6f} ± {std:.6f}"


def markdown_table(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(rows)


def main():
    rows = []
    cond_rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = BASE / model / f"seed_{seed}" / "metrics.json"
            data = json.loads(path.read_text())
            flat = {k: v for k, v in data.items() if k != "per_condition" and not isinstance(v, dict)}
            rows.append(flat)
            for item in data["per_condition"]:
                cond_rows.append({"model": model.upper(), "seed": seed, **item})
    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(cond_rows)
    df.to_csv(BASE / "seed_metrics.csv", index=False)
    cdf.to_csv(BASE / "per_condition_seed_metrics.csv", index=False)

    metric_map = {
        "MAE": "mae", "RMSE": "rmse", "R2": "r2", "Macro-F1": "fault_macro_f1", "Mono Violation": "mono_violation_rate",
        "Prediction Bias": "mean_prediction_bias_pred_minus_true",
    }
    agg_rows = []
    for model, group in df.groupby("model", sort=False):
        row = {"Model": model}
        for label, col in metric_map.items():
            mean, std, text = mean_std(group[col]) if col in group else (None, None, "N/A")
            row[f"{label} mean"] = mean
            row[f"{label} std"] = std
            row[f"{label} mean±std"] = text
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(BASE / "aggregate_metrics.csv", index=False)

    paper = agg[["Model", "MAE mean±std", "RMSE mean±std", "R2 mean±std", "Macro-F1 mean±std", "Mono Violation mean±std"]]
    (BASE / "aggregate_metrics.md").write_text("# Aggregate Metrics\n\n" + markdown_table(paper) + "\n", encoding="utf-8")

    pc_rows = []
    for (model, cond), group in cdf.groupby(["model", "condition"], sort=False):
        row = {"Model": model, "condition": cond}
        for label, col in {"MAE": "mae", "RMSE": "rmse", "R2": "r2"}.items():
            mean, std, text = mean_std(group[col])
            row[f"{label} mean"] = mean; row[f"{label} std"] = std; row[f"{label} mean±std"] = text
        pc_rows.append(row)
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(BASE / "per_condition_aggregate.csv", index=False)
    (BASE / "per_condition_aggregate.md").write_text("# Per-Condition Aggregate\n\n" + markdown_table(pc[["Model", "condition", "MAE mean±std", "RMSE mean±std", "R2 mean±std"]]) + "\n", encoding="utf-8")

    def mean_for(model, col):
        return float(df[df.model == model][col].mean())

    m3_better_m1_mae = all(
        float(df[(df.model == "M3") & (df.seed == s)].mae.iloc[0]) < float(df[(df.model == "M1") & (df.seed == s)].mae.iloc[0])
        for s in SEEDS
    )
    m3_better_m1_rmse = all(
        float(df[(df.model == "M3") & (df.seed == s)].rmse.iloc[0]) < float(df[(df.model == "M1") & (df.seed == s)].rmse.iloc[0])
        for s in SEEDS
    )
    m5_better_m3 = all(
        float(df[(df.model == "M5") & (df.seed == s)].mae.iloc[0]) < float(df[(df.model == "M3") & (df.seed == s)].mae.iloc[0])
        for s in SEEDS
    )
    best_mae = agg.sort_values("MAE mean").iloc[0].Model
    best_rmse = agg.sort_values("RMSE mean").iloc[0].Model
    best_r2 = agg.sort_values("R2 mean", ascending=False).iloc[0].Model
    best_f1 = agg.dropna(subset=["Macro-F1 mean"]).sort_values("Macro-F1 mean", ascending=False).iloc[0].Model
    best_mono = agg.sort_values("Mono Violation mean").iloc[0].Model

    lines = [
        "# Seed Stability Report", "",
        "## Paper-Ready Table", "", markdown_table(paper), "",
        "## Seed-Wise Results", "", markdown_table(df[["model", "seed", "mae", "rmse", "r2", "fault_macro_f1", "mono_violation_rate", "best_epoch", "training_time_seconds"]]), "",
        f"- M3 consistently outperforms M1 on MAE: `{m3_better_m1_mae}`",
        f"- M3 consistently outperforms M1 on RMSE: `{m3_better_m1_rmse}`",
        f"- M5 consistently outperforms M3 on MAE: `{m5_better_m3}`",
        f"- Best mean MAE: `{best_mae}`",
        f"- Best mean RMSE: `{best_rmse}`",
        f"- Best mean R2: `{best_r2}`",
        f"- Best mean fault Macro-F1: `{best_f1}`",
        f"- Lowest mean monotonicity violation rate: `{best_mono}`",
        "",
        "Conclusion stability: use the mean ± sample standard deviation table above; no best-seed cherry-picking was performed.",
    ]
    (BASE / "seed_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (BASE / "aggregate_summary.json").write_text(json.dumps({
        "best_mean_mae": best_mae, "best_mean_rmse": best_rmse, "best_mean_r2": best_r2,
        "best_mean_fault_macro_f1": best_f1, "lowest_mean_monotonicity_violation": best_mono,
        "m3_consistently_beats_m1_mae": m3_better_m1_mae,
        "m3_consistently_beats_m1_rmse": m3_better_m1_rmse,
        "m5_consistently_beats_m3_mae": m5_better_m3,
    }, indent=2), encoding="utf-8")
    print(markdown_table(paper))


if __name__ == "__main__":
    main()

