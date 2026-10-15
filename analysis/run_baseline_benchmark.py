"""LOBO benchmark of non-neural RUL baselines on frozen XJTU-SY features.

This script reads existing processed features only and writes fresh benchmark
artifacts alongside itself. It never reads raw waveforms or alters prior output.
"""
from __future__ import annotations

import gzip
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(r"<PROJECT_ROOT>")
OUT = Path(__file__).parent
PROCESSED = PROJECT / "data" / "processed"
LENGTH, STRIDE, SEED = 10, 1, 20261013

MODELS = {
    "training_mean": {"kind": "constant", "description": "Mean frozen RUL across training windows."},
    "ridge_final_step": {"kind": "pipeline", "estimator": "StandardScaler + Ridge(alpha=1.0)", "alpha": 1.0},
    "extra_trees_final_step": {"kind": "extra_trees", "n_estimators": 150, "min_samples_leaf": 2, "max_features": 1.0, "random_state": SEED, "n_jobs": -1},
    "hist_gradient_boosting_final_step": {"kind": "hgb", "max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": SEED},
    "extra_trees_flattened_window": {"kind": "extra_trees", "n_estimators": 150, "min_samples_leaf": 2, "max_features": 1.0, "random_state": SEED, "n_jobs": -1},
    "recording_index_only_ridge": {"kind": "pipeline", "estimator": "StandardScaler + Ridge(alpha=1.0) on final recording index only", "alpha": 1.0},
}


def load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    with gzip.open(PROCESSED / "xjtu_features_metadata.csv.gz", "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    data = np.load(PROCESSED / "xjtu_features.npz", allow_pickle=False)
    features = [str(x) for x in data["feature_names"].tolist()]
    if len(frame) != len(data["X"]) or not np.allclose(frame["rul"].to_numpy(np.float32), data["rul"]):
        raise RuntimeError("Processed metadata/NPZ RUL alignment mismatch; stopping.")
    if not np.isfinite(frame[features].to_numpy(float)).all():
        raise RuntimeError("Non-finite processed features; stopping.")
    return frame.sort_values(["bearing_id", "chronological_position"]).reset_index(drop=True), data["X"], features


def windows(frame: pd.DataFrame, x: np.ndarray, bearing: str) -> dict[str, np.ndarray]:
    group = frame[frame.bearing_id.eq(bearing)].sort_values("chronological_position")
    positions = group.chronological_position.to_numpy(int)
    if not np.array_equal(positions, np.arange(len(group))):
        raise RuntimeError(f"{bearing}: chronological positions not contiguous")
    row_ids = group.index.to_numpy(int)
    original_indices = group.recording_index.to_numpy(int)
    if not np.all(np.diff(original_indices) > 0):
        raise RuntimeError(f"{bearing}: original recording indices not strictly increasing")
    if len(group) < LENGTH:
        raise RuntimeError(f"{bearing}: too few recordings")
    ends = np.arange(LENGTH - 1, len(group), STRIDE)
    return {
        "final": x[row_ids[ends]],
        "flat": x[row_ids[np.array([np.arange(end - LENGTH + 1, end + 1) for end in ends])]].reshape(len(ends), -1),
        "y": group.rul.to_numpy(float)[ends],
        "end_position": positions[ends],
        "recording_index_end": original_indices[ends],
        "start_position": positions[ends - LENGTH + 1],
    }


def build_model(name: str):
    cfg = MODELS[name]
    if name == "ridge_final_step":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=cfg["alpha"]))])
    if name == "recording_index_only_ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=cfg["alpha"]))])
    if cfg["kind"] == "extra_trees":
        return ExtraTreesRegressor(**{k: v for k, v in cfg.items() if k in {"n_estimators", "min_samples_leaf", "max_features", "random_state", "n_jobs"}})
    if cfg["kind"] == "hgb":
        return HistGradientBoostingRegressor(**{k: v for k, v in cfg.items() if k in {"max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization", "random_state"}})
    raise ValueError(name)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(mean_squared_error(y, pred) ** 0.5), "r2": float(r2_score(y, pred)), "prediction_min": float(np.min(pred)), "prediction_max": float(np.max(pred))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, x, features = load()
    bearings = sorted(frame.bearing_id.unique())
    all_windows = {bearing: windows(frame, x, bearing) for bearing in bearings}
    validation = []
    for bearing, bundle in all_windows.items():
        validation.append({"bearing_id": bearing, "recordings": int((frame.bearing_id == bearing).sum()), "windows": len(bundle["y"]), "target_at_final_step": bool(np.allclose(bundle["y"], frame[frame.bearing_id.eq(bearing)].sort_values("chronological_position").rul.to_numpy()[LENGTH-1:])), "chronological_ordering": True, "finite_features": bool(np.isfinite(bundle["final"]).all()), "rul_in_bounds": bool(np.all((bundle["y"] >= 0) & (bundle["y"] <= 1)))})
    if not all(x["target_at_final_step"] and x["finite_features"] and x["rul_in_bounds"] for x in validation):
        raise RuntimeError("Data validation failed; stopping.")
    pd.DataFrame(validation).to_csv(OUT / "fold_manifest.csv", index=False)
    predictions, bearing_rows, model_rows = [], [], []
    for test_bearing in bearings:
        train_bearings = [b for b in bearings if b != test_bearing]
        train_final = np.concatenate([all_windows[b]["final"] for b in train_bearings])
        train_flat = np.concatenate([all_windows[b]["flat"] for b in train_bearings])
        train_y = np.concatenate([all_windows[b]["y"] for b in train_bearings])
        test = all_windows[test_bearing]
        if test_bearing in train_bearings:
            raise RuntimeError("Bearing overlap")
        for name in MODELS:
            if name == "training_mean":
                start = time.perf_counter(); pred = np.full(len(test["y"]), train_y.mean()); fit_s = time.perf_counter() - start; infer_s = 0.0
            else:
                model = build_model(name)
                if name == "extra_trees_flattened_window":
                    train_input, test_input = train_flat, test["flat"]
                elif name == "recording_index_only_ridge":
                    train_input = np.concatenate([all_windows[b]["recording_index_end"] for b in train_bearings]).reshape(-1, 1)
                    test_input = test["recording_index_end"].reshape(-1, 1)
                else:
                    train_input, test_input = train_final, test["final"]
                start = time.perf_counter(); model.fit(train_input, train_y); fit_s = time.perf_counter() - start
                start = time.perf_counter(); pred = model.predict(test_input); infer_s = time.perf_counter() - start
            result = metrics(test["y"], pred)
            bearing_rows.append({"model": name, "held_out_bearing": test_bearing, "operating_condition": frame.loc[frame.bearing_id.eq(test_bearing), "operating_condition"].iloc[0], "n_windows": len(pred), **result, "fit_cpu_seconds": fit_s, "inference_cpu_seconds": infer_s})
            predictions.extend({"model": name, "held_out_bearing": test_bearing, "operating_condition": frame.loc[frame.bearing_id.eq(test_bearing), "operating_condition"].iloc[0], "start_position": int(s), "end_position": int(e), "recording_index_end": int(i), "true_rul": float(y), "prediction": float(p)} for s, e, i, y, p in zip(test["start_position"], test["end_position"], test["recording_index_end"], test["y"], pred))
    by_bearing = pd.DataFrame(bearing_rows)
    pred_frame = pd.DataFrame(predictions)
    by_bearing.to_csv(OUT / "baseline_metrics_by_bearing.csv", index=False)
    pred_frame.to_csv(OUT / "baseline_predictions_lobo.csv", index=False)
    for name, group in pred_frame.groupby("model"):
        y, p = group.true_rul.to_numpy(), group.prediction.to_numpy()
        bm = by_bearing[by_bearing.model.eq(name)]
        model_rows.append({"model": name, "pooled_window_mae": float(mean_absolute_error(y, p)), "pooled_window_rmse": float(mean_squared_error(y, p) ** 0.5), "equal_bearing_macro_mae": float(bm.mae.mean()), "equal_bearing_macro_rmse": float(bm.rmse.mean()), "pooled_r2_diagnostic": float(r2_score(y, p)), "prediction_min": float(p.min()), "prediction_max": float(p.max()), "mean_fit_cpu_seconds_per_fold": float(bm.fit_cpu_seconds.mean()), "mean_inference_cpu_seconds_per_fold": float(bm.inference_cpu_seconds.mean())})
    models = pd.DataFrame(model_rows).sort_values("pooled_window_mae")
    models.to_csv(OUT / "baseline_metrics_by_model.csv", index=False)
    cfg = {"protocol": {"target": "frozen normalized relative RUL; final step of 10-step window", "split": "leave-one-physical-bearing-out over all 15 bearings", "sequence_length": LENGTH, "stride": STRIDE, "held_out_bearing_excluded_from_fit_scaler": True, "exploratory": True}, "models": MODELS, "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform()}}
    (OUT / "model_configurations.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (OUT / "run_manifest.json").write_text(json.dumps({"processed_npz": str(PROCESSED / "xjtu_features.npz"), "processed_metadata": str(PROCESSED / "xjtu_features_metadata.csv.gz"), "n_recordings": len(frame), "n_features": len(features), "n_bearings": len(bearings), "total_windows": int(sum(len(v["y"]) for v in all_windows.values())), "seed": SEED, "models": list(MODELS), "raw_data_touched": False}, indent=2), encoding="utf-8")
    (OUT / "DATA_VALIDATION_REPORT.md").write_text("# Data Validation\n\n- Processed metadata and NPZ RUL arrays align row-for-row.\n- 32 features are finite.\n- Every bearing is numerically ordered, has contiguous chronological positions, and produces `N-9` windows.\n- Each target equals frozen RUL at the window final step.\n- No window crosses a bearing boundary.\n- Bearing3_4 preserves original numeric index gap; index-only baseline uses observed final index only.\n", encoding="utf-8")
    report = ["# Baseline Benchmark v1", "", "## Protocol", "", "- Exploratory leave-one-physical-bearing-out evaluation over all 15 bearings.", "- All models use the frozen normalized-relative RUL target at the final timestep of each 10-step window.", "- No held-out bearing was included in model fitting or scaler fitting. Hyperparameters were fixed before execution; no inner tuning was run.", "- Pooled results weight windows; macro results give each bearing equal weight. Windows are not independent bearing trajectories.", "", "## Results", "", models.to_markdown(index=False, floatfmt=".6f"), "", "## Interpretation limits", "", "- These LOBO results are exploratory because earlier fixed-split test results were repeatedly inspected.", "- The index-only model uses only the observable final recording index, not normalized age or total bearing length. It remains a target/age diagnostic, not a vibration model.", "- Historical fixed-split M1-M5 and baseline values are not merged with LOBO ranks because the held-out sets and aggregation differ.", "", "## Decision", "", "- Inspect `baseline_metrics_by_model.csv` and per-bearing rows before claiming useful vibration signal. A model must improve over the training-mean and index-only references under both pooled and macro views to support that conclusion.", "- Tree models are not directly comparable to historical BiLSTM results because this protocol differs; only qualitative, explicitly separated context is valid.", "- Proceed to a controlled physics-on versus physics-off neural experiment only after a grouped inner-validation plan and a pre-registered evaluation choice; do not use these LOBO results for tuning."]
    (OUT / "BASELINE_BENCHMARK_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
