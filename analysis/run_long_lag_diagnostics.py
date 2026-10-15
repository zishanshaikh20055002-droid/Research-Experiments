"""Offline within-bearing lagged upward-RUL diagnostics for saved predictions.

No training, labels, or predictions are modified. Strict upward violations use
``later_prediction > earlier_prediction`` with no tolerance for raw outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MULTISEED = ROOT / "results" / "multiseed"
M2U = Path(r"<USER_HOME>\OneDrive\Desktop\shuffle_control_results\outputs\M2-unshuffled")
SEEDS = (42, 123, 2024)
LAGS = (1, 10, 20)
EXPECTED_BEARINGS = (
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing2_3", "Bearing2_4",
    "Bearing2_5", "Bearing3_3", "Bearing3_4", "Bearing3_5",
)
KEYS = ("bearing_id", "start_position", "end_position")


def source_runs() -> list[tuple[str, str, Path]]:
    runs: list[tuple[str, str, Path]] = []
    for model in ("m1", "m2", "m3", "m4", "m5"):
        label = f"M{model[1:]}-U" if model == "m3" else model.upper()
        for seed in SEEDS:
            runs.append((label, str(seed), MULTISEED / model / f"seed_{seed}" / "predictions.csv"))
    for seed in SEEDS:
        runs.append(("M2-U", str(seed), M2U / f"seed_{seed}" / "predictions.csv"))
    return runs


def validate(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = set((*KEYS, "true_rul", "pred_rul")) - set(frame.columns)
    if missing:
        raise AssertionError(f"{source}: missing columns {sorted(missing)}")
    if len(frame) != 3164 or frame.duplicated(list(KEYS)).any():
        raise AssertionError(f"{source}: expected 3,164 unique test prediction rows")
    found = tuple(sorted(frame.bearing_id.unique()))
    if found != tuple(sorted(EXPECTED_BEARINGS)):
        raise AssertionError(f"{source}: unexpected bearing IDs {found}")
    if not np.isfinite(frame.pred_rul.to_numpy(float)).all():
        raise AssertionError(f"{source}: non-finite prediction")
    return frame.copy()


def bearing_lag_metrics(group: pd.DataFrame, lag: int) -> dict[str, float | int | bool]:
    ordered = group.sort_values("end_position", kind="mergesort")
    values = ordered.pred_rul.to_numpy(float)
    if len(values) <= lag:
        return {"eligible": False, "valid_transitions": 0, "violations": 0,
                "violation_rate": np.nan, "mean_positive_increase": np.nan}
    deltas = values[lag:] - values[:-lag]
    positive = deltas[deltas > 0.0]
    return {
        "eligible": True,
        "valid_transitions": int(len(deltas)),
        "violations": int(len(positive)),
        "violation_rate": float(len(positive) / len(deltas)),
        "mean_positive_increase": float(positive.mean()) if len(positive) else 0.0,
    }


def diagnostic_rows(method: str, seed: str, frame: pd.DataFrame, variant: str) -> tuple[list[dict], list[dict]]:
    per_bearing, summaries = [], []
    for lag in LAGS:
        lag_rows = []
        for bearing in EXPECTED_BEARINGS:
            metrics = bearing_lag_metrics(frame.loc[frame.bearing_id.eq(bearing)], lag)
            row = {"method": method, "seed": seed, "variant": variant, "lag": lag,
                   "bearing_id": bearing, **metrics}
            per_bearing.append(row)
            lag_rows.append(row)
        eligible = [row for row in lag_rows if row["eligible"]]
        transitions = sum(int(row["valid_transitions"]) for row in eligible)
        violations = sum(int(row["violations"]) for row in eligible)
        positive_sum = 0.0
        for bearing in EXPECTED_BEARINGS:
            ordered = frame.loc[frame.bearing_id.eq(bearing)].sort_values("end_position", kind="mergesort")
            values = ordered.pred_rul.to_numpy(float)
            if len(values) > lag:
                positive_sum += float(np.maximum(values[lag:] - values[:-lag], 0.0).sum())
        summaries.append({
            "method": method, "seed": seed, "variant": variant, "lag": lag,
            "eligible_bearing_count": len(eligible), "ineligible_bearing_count": len(lag_rows) - len(eligible),
            "valid_transition_count": transitions, "violating_transition_count": violations,
            "macro_violation_rate": float(np.mean([row["violation_rate"] for row in eligible])) if eligible else np.nan,
            "macro_mean_positive_increase": float(np.mean([row["mean_positive_increase"] for row in eligible])) if eligible else np.nan,
            "pooled_violation_rate": float(violations / transitions) if transitions else np.nan,
            "pooled_mean_positive_increase": float(positive_sum / violations) if violations else 0.0,
        })
    return per_bearing, summaries


def main() -> None:
    per_bearing_rows: list[dict] = []
    seed_rows: list[dict] = []
    sources = []
    for method, seed, source in source_runs():
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = validate(pd.read_csv(source), source)
        rows, summary = diagnostic_rows(method, seed, frame, "raw")
        per_bearing_rows.extend(rows)
        seed_rows.extend(summary)
        sources.append({"method": method, "seed": seed, "raw_prediction_source": str(source)})

    # Isotonic is a sanity check only: it should yield no strict upward violations.
    projected_root = OUT / "projected_predictions"
    isotonic_seed_rows: list[dict] = []
    for method, seed, _ in source_runs():
        directory = projected_root / ("M3" if method == "M3-U" else method)
        candidate = directory / f"{seed}_predictions_raw_and_isotonic.csv"
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing projected prediction file: {candidate}")
        saved = pd.read_csv(candidate)
        projected = saved.drop(columns=["pred_rul", "pred_rul_raw"], errors="ignore").rename(
            columns={"pred_rul_isotonic": "pred_rul"}
        )
        projected = validate(projected, candidate)
        rows, summary = diagnostic_rows(method, seed, projected, "isotonic_sanity")
        per_bearing_rows.extend(rows)
        isotonic_seed_rows.extend(summary)

    per_bearing = pd.DataFrame(per_bearing_rows)
    seed_summary = pd.DataFrame(seed_rows)
    aggregate = seed_summary.groupby(["method", "lag"], as_index=False).agg(
        seed_count=("seed", "count"),
        eligible_bearing_count=("eligible_bearing_count", "first"),
        ineligible_bearing_count=("ineligible_bearing_count", "first"),
        valid_transition_count=("valid_transition_count", "first"),
        macro_violation_rate_mean=("macro_violation_rate", "mean"),
        macro_violation_rate_sd=("macro_violation_rate", "std"),
        macro_mean_positive_increase_mean=("macro_mean_positive_increase", "mean"),
        macro_mean_positive_increase_sd=("macro_mean_positive_increase", "std"),
        pooled_violation_rate_mean=("pooled_violation_rate", "mean"),
        pooled_violation_rate_sd=("pooled_violation_rate", "std"),
        pooled_mean_positive_increase_mean=("pooled_mean_positive_increase", "mean"),
        pooled_mean_positive_increase_sd=("pooled_mean_positive_increase", "std"),
    )
    isotonic_summary = pd.DataFrame(isotonic_seed_rows)
    isotonic_check = isotonic_summary.groupby(["method", "lag"], as_index=False).agg(
        seed_count=("seed", "count"), valid_transition_count=("valid_transition_count", "first"),
        violating_transition_count=("violating_transition_count", "sum"),
        max_pooled_violation_rate=("pooled_violation_rate", "max"),
    )
    if not isotonic_check.violating_transition_count.eq(0).all():
        raise AssertionError("Isotonic sanity check found strict upward violations")

    per_bearing.to_csv(OUT / "long_lag_per_bearing.csv", index=False)
    seed_summary.to_csv(OUT / "long_lag_per_seed.csv", index=False)
    aggregate.to_csv(OUT / "long_lag_aggregate.csv", index=False)
    isotonic_check.to_csv(OUT / "long_lag_isotonic_sanity.csv", index=False)
    (OUT / "long_lag_sources.json").write_text(json.dumps({"sources": sources, "lags": list(LAGS)}, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Wrote {len(per_bearing)} per-bearing rows and {len(seed_summary)} raw per-seed summary rows")


if __name__ == "__main__":
    main()
