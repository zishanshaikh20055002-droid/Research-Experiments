from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PRED_OUT = OUT / "projected_predictions"
MULTISEED = ROOT / "results" / "multiseed"
M2U = Path(r"<USER_HOME>\OneDrive\Desktop\shuffle_control_results\outputs\M2-unshuffled")
META = ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz"
SEEDS = (42, 123, 2024)
TEST = (
    "Bearing1_3", "Bearing1_4", "Bearing1_5",
    "Bearing2_3", "Bearing2_4", "Bearing2_5",
    "Bearing3_3", "Bearing3_4", "Bearing3_5",
)
TRAIN = ("Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing3_1", "Bearing3_2")
KEYS = ["bearing_id", "start_position", "end_position"]
BOOTSTRAP_SEED = 20261015
N_BOOT = 1000
KNOT_QUANTILES = (0.0, 1 / 3, 2 / 3, 1.0)


def sequence_targets(frame: pd.DataFrame, bearings: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for bearing in bearings:
        group = frame.loc[frame.bearing_id.eq(bearing)].sort_values("chronological_position")
        for start in range(0, len(group) - 10 + 1):
            target = group.iloc[start + 9]
            rows.append({
                "bearing_id": bearing,
                "start_position": int(group.iloc[start].chronological_position),
                "end_position": int(target.chronological_position),
                "true_rul": float(target.rul),
                "condition": str(target.operating_condition),
            })
    return pd.DataFrame(rows)


def equal_bearing_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("bearing_id").size().to_dict()
    return np.asarray([1.0 / counts[b] for b in frame.bearing_id], dtype=float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    idx = int(np.searchsorted(np.cumsum(weights), q * weights.sum(), side="left"))
    return float(values[min(idx, len(values) - 1)])


def interpolation_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, float), knots[0], knots[-1])
    basis = np.zeros((len(clipped), len(knots)), dtype=float)
    interval = np.searchsorted(knots, clipped, side="right") - 1
    interval = np.clip(interval, 0, len(knots) - 2)
    left, right = knots[interval], knots[interval + 1]
    rw = (clipped - left) / (right - left)
    rr = np.arange(len(clipped))
    basis[rr, interval], basis[rr, interval + 1] = 1 - rw, rw
    return basis


def age_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    weights = equal_bearing_weights(train)
    x, y = train.end_position.to_numpy(float), train.true_rul.to_numpy(float)
    knots = np.asarray([weighted_quantile(x, weights, q) for q in KNOT_QUANTILES])
    design = interpolation_basis(x, knots)
    root_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
    out = test.copy()
    out["pred_rul"] = np.clip(interpolation_basis(out.end_position.to_numpy(float), knots) @ coef, 0, 1)
    return out, {"knots": knots.tolist(), "coefficients": coef.tolist(), "fit_bearings": list(TRAIN)}


def validate_rows(reference: pd.DataFrame, candidate: pd.DataFrame, source: Path) -> pd.DataFrame:
    if candidate.duplicated(KEYS).any():
        raise AssertionError(f"Duplicate keys: {source}")
    merged = reference[KEYS + ["true_rul", "condition"]].merge(
        candidate[KEYS + ["true_rul", "pred_rul"]], on=KEYS, how="outer",
        suffixes=("_ref", "_candidate"), indicator=True, validate="one_to_one",
    )
    if len(merged) != 3164 or not merged._merge.eq("both").all():
        raise AssertionError(f"Prediction rows do not match fixed test set: {source}")
    if not np.allclose(merged.true_rul_ref, merged.true_rul_candidate, atol=1e-7, rtol=0):
        raise AssertionError(f"Target mismatch: {source}")
    return merged.rename(columns={"true_rul_ref": "true_rul"})[KEYS + ["condition", "true_rul", "pred_rul"]]


def project(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pred_rul_raw"] = out.pred_rul
    out["pred_rul_isotonic"] = np.nan
    for _, group in out.groupby("bearing_id", sort=False):
        ordered = group.sort_values("end_position")
        fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
            ordered.end_position.to_numpy(float), ordered.pred_rul.to_numpy(float)
        )
        out.loc[ordered.index, "pred_rul_isotonic"] = fitted
    if out.pred_rul_isotonic.isna().any():
        raise AssertionError("Isotonic projection left missing predictions")
    return out


def metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    y, p = frame.true_rul.to_numpy(float), frame[column].to_numpy(float)
    per = []
    for _, group in frame.groupby("bearing_id", sort=True):
        e = group[column].to_numpy(float) - group.true_rul.to_numpy(float)
        per.append((np.mean(np.abs(e)), np.sqrt(np.mean(e**2))))
    return {
        "pooled_mae": float(np.mean(np.abs(p - y))),
        "pooled_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "macro_mae": float(np.mean([x[0] for x in per])),
        "macro_rmse": float(np.mean([x[1] for x in per])),
    }


def load_methods(reference: pd.DataFrame, age: pd.DataFrame) -> dict[str, list[tuple[str, pd.DataFrame, Path | None]]]:
    methods: dict[str, list[tuple[str, pd.DataFrame, Path | None]]] = {"age_baseline": [("deterministic", age, None)]}
    for model in ("m1", "m2", "m3", "m4", "m5"):
        methods[model.upper()] = []
        for seed in SEEDS:
            source = MULTISEED / model / f"seed_{seed}" / "predictions.csv"
            methods[model.upper()].append((str(seed), validate_rows(reference, pd.read_csv(source), source), source))
    methods["M2-U"] = []
    for seed in SEEDS:
        source = M2U / f"seed_{seed}" / "predictions.csv"
        methods["M2-U"].append((str(seed), validate_rows(reference, pd.read_csv(source), source), source))
    return methods


def bootstrap(methods: dict[str, list[tuple[str, pd.DataFrame, Path | None]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.choice(TEST, size=len(TEST), replace=True) for _ in range(N_BOOT)]
    replicate_rows = []
    for method, versions in methods.items():
        for variant, column in (("raw", "pred_rul_raw"), ("isotonic", "pred_rul_isotonic")):
            projected = [project(frame) for _, frame, _ in versions]
            for rep, sampled in enumerate(draws):
                seed_metrics = []
                for frame in projected:
                    pieces = [frame.loc[frame.bearing_id.eq(b)] for b in sampled]
                    pooled = pd.concat(pieces, ignore_index=True)
                    bearing_mae, bearing_rmse = [], []
                    for piece in pieces:
                        e = piece[column].to_numpy(float) - piece.true_rul.to_numpy(float)
                        bearing_mae.append(np.mean(np.abs(e)))
                        bearing_rmse.append(np.sqrt(np.mean(e**2)))
                    e = pooled[column].to_numpy(float) - pooled.true_rul.to_numpy(float)
                    seed_metrics.append((np.mean(bearing_mae), np.mean(bearing_rmse), np.mean(np.abs(e)), np.sqrt(np.mean(e**2))))
                values = np.mean(np.asarray(seed_metrics), axis=0)
                replicate_rows.append({"method": method, "variant": variant, "replicate": rep,
                                       "macro_mae": values[0], "macro_rmse": values[1],
                                       "pooled_mae": values[2], "pooled_rmse": values[3]})
    reps = pd.DataFrame(replicate_rows)
    summaries = []
    for (method, variant), group in reps.groupby(["method", "variant"], sort=True):
        row = {"method": method, "variant": variant, "bootstrap_unit": "bearing", "replicates": N_BOOT,
               "seed_aggregation": "metric per seed, then arithmetic mean across available seeds"}
        for metric_name in ("macro_mae", "macro_rmse", "pooled_mae", "pooled_rmse"):
            values = group[metric_name].to_numpy(float)
            row[f"{metric_name}_bootstrap_mean"] = float(values.mean())
            row[f"{metric_name}_ci_low"] = float(np.percentile(values, 2.5))
            row[f"{metric_name}_ci_high"] = float(np.percentile(values, 97.5))
        summaries.append(row)
    return pd.DataFrame(summaries), reps


def main() -> None:
    PRED_OUT.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(META).sort_values(["bearing_id", "chronological_position"]).reset_index(drop=True)
    train, test = sequence_targets(metadata, TRAIN), sequence_targets(metadata, TEST)
    if len(test) != 3164 or test.bearing_id.nunique() != 9:
        raise AssertionError("Fixed test rows are not 3,164 windows from nine bearings")
    age, age_config = age_predictions(train, test)
    methods = load_methods(test, age)

    m1_single = validate_rows(test, pd.read_csv(ROOT / "results" / "m1" / "m1_test_predictions.csv"), ROOT / "results" / "m1" / "m1_test_predictions.csv")
    check = project(m1_single)
    raw_check = metrics(check, "pred_rul_raw")["pooled_mae"]
    iso_check = metrics(check, "pred_rul_isotonic")["pooled_mae"]
    if not (abs(raw_check - 0.306988443) < 1e-6 and abs(iso_check - 0.298858289) < 1e-6):
        raise AssertionError(f"M1 isotonic validation failed: {raw_check}, {iso_check}")

    metric_rows = []
    for method, versions in methods.items():
        method_dir = PRED_OUT / method.replace("/", "_")
        method_dir.mkdir(parents=True)
        for version, frame, source in versions:
            projected = project(frame)
            projected.to_csv(method_dir / f"{version}_predictions_raw_and_isotonic.csv", index=False)
            for variant, column in (("raw", "pred_rul_raw"), ("isotonic", "pred_rul_isotonic")):
                metric_rows.append({"method": method, "seed": version, "variant": variant,
                                    "source": str(source) if source else "training-only age fit", **metrics(projected, column)})

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT / "isotonic_metrics_by_seed.csv", index=False)
    aggregate = metric_df.groupby(["method", "variant"], as_index=False).agg(
        seed_count=("seed", "count"), pooled_mae_mean=("pooled_mae", "mean"), pooled_mae_sd=("pooled_mae", "std"),
        pooled_rmse_mean=("pooled_rmse", "mean"), pooled_rmse_sd=("pooled_rmse", "std"),
        macro_mae_mean=("macro_mae", "mean"), macro_mae_sd=("macro_mae", "std"),
        macro_rmse_mean=("macro_rmse", "mean"), macro_rmse_sd=("macro_rmse", "std"),
    )
    aggregate.to_csv(OUT / "isotonic_metrics_aggregate.csv", index=False)
    ci, reps = bootstrap(methods)
    ci.to_csv(OUT / "bearing_bootstrap_ci.csv", index=False)
    reps.to_csv(OUT / "bearing_bootstrap_replicates.csv", index=False)
    (OUT / "age_baseline_fit.json").write_text(json.dumps(age_config, indent=2), encoding="utf-8")
    (OUT / "analysis_config.json").write_text(json.dumps({
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": N_BOOT,
        "bootstrap_unit": "whole test bearing", "test_bearings": list(TEST),
        "isotonic": "fit separately to each bearing's prediction vs end_position; increasing=False; no true labels used",
        "m1_validation": {"raw_pooled_mae": raw_check, "isotonic_pooled_mae": iso_check},
        "status": "exploratory; fixed test set previously inspected",
    }, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Wrote {len(reps)} paired bootstrap rows and {len(ci)} CI rows to {OUT}")


if __name__ == "__main__":
    main()
