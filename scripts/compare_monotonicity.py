"""Offline monotonicity comparison for saved M1/M2/M3/M5 test predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = RESEARCH_ROOT / "results"


PREDICTION_FILES = {
    "M1": RESULTS_ROOT / "m1" / "m1_test_predictions.csv",
    "M2": RESULTS_ROOT / "m2" / "m2_test_predictions.csv",
    "M3": RESULTS_ROOT / "m3" / "m3_test_predictions.csv",
    "M5": RESULTS_ROOT / "m5" / "m5_test_predictions.csv",
}


def condition_for_bearing(bearing_id: str) -> str:
    if bearing_id.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing_id.startswith("Bearing2_"):
        return "37.5Hz11kN"
    if bearing_id.startswith("Bearing3_"):
        return "40Hz10kN"
    raise KeyError(bearing_id)


def violation_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    violating_magnitudes: list[float] = []
    total_pairs = 0
    for _, group in frame.groupby("bearing_id", sort=True):
        ordered = group.sort_values("end_position")
        predictions = ordered["pred_rul"].to_numpy(dtype=float)
        diffs = predictions[1:] - predictions[:-1]
        violating = diffs[diffs > 0.0]
        total_pairs += int(len(diffs))
        violating_magnitudes.extend(violating.tolist())
    violation_count = len(violating_magnitudes)
    return {
        "violation_rate": float(violation_count / total_pairs) if total_pairs else 0.0,
        "mean_violation_magnitude": float(np.mean(violating_magnitudes)) if violating_magnitudes else 0.0,
        "violating_transitions": int(violation_count),
        "valid_adjacent_transitions": int(total_pairs),
    }


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


def main() -> int:
    output_dir = RESULTS_ROOT / "fasttrack_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    bearing_rows: list[dict[str, object]] = []

    for model, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path)
        if "condition" not in frame.columns:
            frame["condition"] = frame["bearing_id"].map(condition_for_bearing)

        rows.append({"model": model, **violation_stats(frame)})

        for condition, group in frame.groupby("condition", sort=True):
            condition_rows.append({"model": model, "condition": condition, **violation_stats(group)})

        for bearing_id, group in frame.groupby("bearing_id", sort=True):
            bearing_rows.append(
                {
                    "model": model,
                    "bearing_id": bearing_id,
                    "condition": condition_for_bearing(bearing_id),
                    **violation_stats(group),
                }
            )

    summary = pd.DataFrame(rows)
    per_condition = pd.DataFrame(condition_rows)
    per_bearing = pd.DataFrame(bearing_rows)

    summary.to_csv(output_dir / "monotonicity_comparison.csv", index=False)
    per_condition.to_csv(output_dir / "monotonicity_comparison_by_condition.csv", index=False)
    per_bearing.to_csv(output_dir / "monotonicity_comparison_by_bearing.csv", index=False)

    lines = [
        "# Monotonicity Comparison",
        "",
        "Offline analysis using saved test prediction CSVs only. Predictions are grouped strictly by bearing and sorted chronologically by sequence end position. True RUL is not used for violation calculation.",
        "",
        "## Overall",
        "",
        markdown_table(summary[["model", "violation_rate", "mean_violation_magnitude", "violating_transitions", "valid_adjacent_transitions"]]),
        "",
        "## By Condition",
        "",
        markdown_table(per_condition[["model", "condition", "violation_rate", "mean_violation_magnitude", "violating_transitions", "valid_adjacent_transitions"]]),
        "",
        "## By Bearing",
        "",
        markdown_table(per_bearing[["model", "bearing_id", "condition", "violation_rate", "mean_violation_magnitude", "violating_transitions", "valid_adjacent_transitions"]]),
        "",
    ]
    (output_dir / "monotonicity_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(output_dir / "monotonicity_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

