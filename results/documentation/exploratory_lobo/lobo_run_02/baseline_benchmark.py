"""Leakage-free leave-one-bearing-out baseline benchmark for frozen XJTU-SY features.

This script deliberately creates outputs only beside itself. It neither changes the
processed dataset nor trains any neural or physics-guided model.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEQUENCE_LENGTH = 10
SEED = 42
RUN_DIR = Path(__file__).resolve().parent
PROJECT = RUN_DIR.parents[1]
DATA_DIR = PROJECT / "data" / "processed"
NPZ_PATH = DATA_DIR / "xjtu_features.npz"
METADATA_PATH = DATA_DIR / "xjtu_features_metadata.csv.gz"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_std": float(np.std(y_pred, ddof=0)),
    }


def build_windows(
    frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build windows per bearing in validated chronological order.

    The target is the RUL on each window's final recording: x[t:t+10] -> rul[t+9].
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    bearings: list[str] = []
    conditions: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index_ends: list[int] = []

    for bearing_id, bearing in frame.groupby("bearing_id", sort=True):
        bearing = bearing.sort_values("chronological_position", kind="stable").reset_index(drop=True)
        positions = bearing["chronological_position"].to_numpy(dtype=int)
        indices = bearing["recording_index"].to_numpy(dtype=int)
        if not np.array_equal(positions, np.arange(len(bearing))):
            raise ValueError(f"{bearing_id}: chronological positions are not consecutive")
        if np.any(np.diff(indices) <= 0):
            raise ValueError(f"{bearing_id}: recording indices are not strictly increasing")
        if len(bearing) < SEQUENCE_LENGTH:
            raise ValueError(f"{bearing_id}: fewer than {SEQUENCE_LENGTH} recordings")

        source_rows = bearing["_source_row"].to_numpy(dtype=int)
        values = feature_matrix[source_rows]
        ruls = target_vector[source_rows]
        condition = str(bearing["operating_condition"].iloc[0])
        for start in range(0, len(bearing) - SEQUENCE_LENGTH + 1):
            end = start + SEQUENCE_LENGTH - 1
            sequences.append(values[start : end + 1])
            targets.append(float(ruls[end]))
            bearings.append(str(bearing_id))
            conditions.append(condition)
            starts.append(int(positions[start]))
            ends.append(int(positions[end]))
            index_ends.append(int(indices[end]))

    result = {
        "X_window": np.asarray(sequences, dtype=np.float64),
        "y": np.asarray(targets, dtype=np.float64),
        "bearing_id": np.asarray(bearings, dtype=str),
        "operating_condition": np.asarray(conditions, dtype=str),
        "start_position": np.asarray(starts, dtype=int),
        "end_position": np.asarray(ends, dtype=int),
        "recording_index_end": np.asarray(index_ends, dtype=int),
    }
    result["X_final"] = result["X_window"][:, -1, :]
    result["X_flat"] = result["X_window"].reshape(len(result["y"]), -1)
    return result


def model_specs() -> dict[str, dict]:
    return {
        "training_mean": {
            "input": "none", "description": "Constant mean of training-fold sequence targets only.",
        },
        "ridge_final_step": {
            "input": "32 final-step standardized features",
            "factory": lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        },
        "extra_trees_final_step": {
            "input": "32 final-step features",
            "factory": lambda: ExtraTreesRegressor(
                n_estimators=150, random_state=SEED, min_samples_leaf=2,
                max_features=1.0, n_jobs=-1,
            ),
        },
        "random_forest_final_step": {
            "input": "32 final-step features",
            "factory": lambda: RandomForestRegressor(
                n_estimators=150, random_state=SEED, min_samples_leaf=2,
                max_features=1.0, n_jobs=-1,
            ),
        },
        "hist_gradient_boosting_final_step": {
            "input": "32 final-step features",
            "factory": lambda: HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED,
            ),
        },
        "extra_trees_flattened_window": {
            "input": "flattened 10 x 32 chronological window (320 values)",
            "factory": lambda: ExtraTreesRegressor(
                n_estimators=150, random_state=SEED, min_samples_leaf=2,
                max_features=1.0, n_jobs=-1,
            ),
        },
        "recording_index_ridge_diagnostic": {
            "input": "observable original recording index at sequence end only",
            "factory": lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            "description": "Diagnostic only; no normalized age, total length, terminal index, or future observation is supplied.",
        },
    }


def inputs_for(name: str, data: dict[str, np.ndarray]) -> np.ndarray:
    if name == "ridge_final_step":
        return data["X_final"]
    if name in {"extra_trees_final_step", "random_forest_final_step", "hist_gradient_boosting_final_step"}:
        return data["X_final"]
    if name == "extra_trees_flattened_window":
        return data["X_flat"]
    if name == "recording_index_ridge_diagnostic":
        return data["recording_index_end"].reshape(-1, 1).astype(np.float64)
    raise KeyError(name)


def md_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([line, separator, *body])


def main() -> None:
    npz = np.load(NPZ_PATH, allow_pickle=False)
    metadata = pd.read_csv(METADATA_PATH)
    metadata["_source_row"] = np.arange(len(metadata), dtype=int)
    feature_columns = npz["feature_names"].astype(str).tolist()
    X = npz["X"].astype(np.float64)
    y_npz = npz["rul"].astype(np.float64)

    validation: list[str] = []
    if len(metadata) != len(X) or len(y_npz) != len(X):
        raise ValueError("Processed feature matrix, target vector, and metadata lengths disagree")
    validation.append(f"Processed rows agree: X={X.shape}, metadata={len(metadata)}, rul={len(y_npz)}.")
    if feature_columns != [column for column in metadata.columns if column in feature_columns]:
        raise ValueError("Feature column order differs between NPZ and metadata")
    if not np.isfinite(X).all() or not np.isfinite(y_npz).all():
        raise ValueError("NPZ contains NaN or Inf")
    # NPZ targets are float32 while CSV targets are text-decoded float64.
    # Row keys are exact; a <=3e-8 representation difference is expected.
    if not np.allclose(metadata["rul"].to_numpy(dtype=float), y_npz, atol=1e-7, rtol=0.0):
        raise ValueError("Metadata RUL values differ from NPZ RUL values")
    validation.append("Feature values and RUL values are finite; NPZ and metadata RUL values agree within float32 representation tolerance (1e-7).")

    data = build_windows(metadata, X, y_npz)
    bearings = sorted(metadata["bearing_id"].unique().tolist())
    expected_windows = len(metadata) - len(bearings) * (SEQUENCE_LENGTH - 1)
    if len(data["y"]) != expected_windows:
        raise ValueError(f"Expected {expected_windows} windows, produced {len(data['y'])}")
    validation.append(
        f"Built {len(data['y'])} windows using the NPZ feature matrix and target vector: x[t:t+{SEQUENCE_LENGTH}] -> RUL[t+{SEQUENCE_LENGTH - 1}] across {len(bearings)} bearings."
    )
    if not np.all((data["y"] >= 0.0) & (data["y"] <= 1.0)):
        raise ValueError("Window targets fall outside [0, 1]")
    validation.append("All sequence targets are normalized RUL values in [0, 1]; windows never span bearing identities.")
    b34 = metadata.loc[metadata["bearing_id"] == "Bearing3_4", "recording_index"].to_numpy(dtype=int)
    if 1105 in b34 or not np.any(np.diff(b34) > 1):
        raise ValueError("Expected Bearing3_4 numeric gap at 1105 was not observed")
    validation.append("Bearing3_4 retains its original numeric-index gap at 1105; no file is fabricated or renumbered.")

    specs = model_specs()
    config_for_json = {
        "benchmark": "Exploratory leave-one-bearing-out baseline benchmark on the frozen processed XJTU-SY feature dataset",
        "sequence_definition": "Ten recordings, stride 1; target is normalized RUL at the final recording.",
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "random_seed": SEED,
        "models": {
            name: {key: value for key, value in spec.items() if key != "factory"} | {
                "parameters": {
                    "ridge_alpha": 1.0 if "ridge" in name else None,
                    "n_estimators": 150 if "trees" in name or "forest" in name else None,
                    "min_samples_leaf": 2 if "trees" in name or "forest" in name else None,
                    "hgb_learning_rate": 0.05 if "hist_gradient" in name else None,
                    "hgb_max_iter": 150 if "hist_gradient" in name else None,
                    "hgb_max_leaf_nodes": 15 if "hist_gradient" in name else None,
                    "hgb_l2_regularization": 1.0 if "hist_gradient" in name else None,
                },
            }
            for name, spec in specs.items()
        },
    }
    (RUN_DIR / "model_configurations.json").write_text(json.dumps(config_for_json, indent=2), encoding="utf-8")

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    bearing_metric_rows: list[dict] = []
    all_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in specs}
    total_started = time.perf_counter()

    for fold_number, heldout in enumerate(bearings, start=1):
        test_mask = data["bearing_id"] == heldout
        train_mask = ~test_mask
        train_bearings = sorted(set(data["bearing_id"][train_mask]))
        if heldout in train_bearings or set(train_bearings).intersection({heldout}):
            raise ValueError(f"Fold leakage detected for {heldout}")
        fold_rows.append({
            "fold": fold_number,
            "heldout_bearing": heldout,
            "training_bearings": ";".join(train_bearings),
            "n_training_bearings": len(train_bearings),
            "n_train_windows": int(train_mask.sum()),
            "n_test_windows": int(test_mask.sum()),
            "heldout_in_training": False,
            "scaler_policy": "Fitted only on this fold's 14 training bearings for Ridge models; tree models need no scaler.",
        })

        y_train, y_test = data["y"][train_mask], data["y"][test_mask]
        for name, spec in specs.items():
            started = time.perf_counter()
            if name == "training_mean":
                prediction = np.full(len(y_test), float(np.mean(y_train)), dtype=np.float64)
            else:
                estimator = spec["factory"]()
                x_all = inputs_for(name, data)
                estimator.fit(x_all[train_mask], y_train)
                prediction = estimator.predict(x_all[test_mask]).astype(np.float64)
            elapsed = time.perf_counter() - started
            result = metrics(y_test, prediction)
            bearing_metric_rows.append({
                "model": name,
                "heldout_bearing": heldout,
                "operating_condition": data["operating_condition"][test_mask][0],
                "n_test_windows": int(test_mask.sum()),
                **result,
                "fit_predict_seconds": elapsed,
            })
            all_predictions[name].append((y_test, prediction))
            for local_index, row_index in enumerate(np.flatnonzero(test_mask)):
                prediction_rows.append({
                    "model": name,
                    "heldout_bearing": heldout,
                    "operating_condition": data["operating_condition"][row_index],
                    "start_position": int(data["start_position"][row_index]),
                    "end_position": int(data["end_position"][row_index]),
                    "recording_index_end": int(data["recording_index_end"][row_index]),
                    "true_rul": float(data["y"][row_index]),
                    "prediction": float(prediction[local_index]),
                })
        print(f"Completed LOBO fold {fold_number}/{len(bearings)}: {heldout}", flush=True)

    metrics_rows: list[dict] = []
    bearing_metrics = pd.DataFrame(bearing_metric_rows)
    for name in specs:
        y_true = np.concatenate([pair[0] for pair in all_predictions[name]])
        y_pred = np.concatenate([pair[1] for pair in all_predictions[name]])
        pooled = metrics(y_true, y_pred)
        per_bearing = bearing_metrics[bearing_metrics["model"] == name]
        metrics_rows.append({
            "model": name,
            "evaluation_protocol": "LOBO pooled windows across 15 held-out bearings",
            "n_folds": len(bearings),
            "n_predictions": len(y_true),
            **pooled,
            "macro_bearing_mae": float(per_bearing["mae"].mean()),
            "macro_bearing_rmse": float(per_bearing["rmse"].mean()),
            "macro_bearing_r2": float(per_bearing["r2"].mean()),
        })

    metrics_frame = pd.DataFrame(metrics_rows).sort_values("mae", kind="stable")
    predictions_frame = pd.DataFrame(prediction_rows)
    folds_frame = pd.DataFrame(fold_rows)
    metrics_frame.to_csv(RUN_DIR / "baseline_metrics_by_model.csv", index=False)
    bearing_metrics.sort_values(["model", "heldout_bearing"], kind="stable").to_csv(
        RUN_DIR / "baseline_metrics_by_bearing.csv", index=False
    )
    predictions_frame.sort_values(["model", "heldout_bearing", "end_position"], kind="stable").to_csv(
        RUN_DIR / "baseline_predictions_lobo.csv", index=False
    )
    folds_frame.to_csv(RUN_DIR / "fold_manifest.csv", index=False)

    validation.extend([
        "Each of 15 folds holds out exactly one bearing; every held-out bearing appears once and never occurs in its fold's fit set.",
        "For each model, prediction and target rows use the same held-out sequence row and final-step target.",
        "Ridge scaling is implemented as a Pipeline, fitted only when estimator.fit receives fold training data. No scaler is fit for trees or constants.",
        "No validation or historical fixed-test split is used in fitting, feature selection, hyperparameter selection, or model selection.",
    ])
    validation_report = "# Data Validation Report\n\n" + "\n".join(f"- {line}" for line in validation) + "\n\n"
    validation_report += "## Frozen inputs\n\n"
    validation_report += f"- Features: `{NPZ_PATH}`\n- Metadata: `{METADATA_PATH}`\n"
    validation_report += "- The benchmark is a new exploratory LOBO protocol and is not directly comparable with the historical fixed nine-bearing test split.\n"
    (RUN_DIR / "DATA_VALIDATION_REPORT.md").write_text(validation_report, encoding="utf-8")

    summary_for_report = metrics_frame[["model", "n_predictions", "mae", "rmse", "r2", "macro_bearing_mae", "macro_bearing_rmse"]].copy()
    for column in ["mae", "rmse", "r2", "macro_bearing_mae", "macro_bearing_rmse"]:
        summary_for_report[column] = summary_for_report[column].map(lambda value: f"{value:.6f}")
    historical = PROJECT / "results" / "baselines" / "baseline_metrics.csv"
    report = "# Baseline Benchmark v1: Exploratory LOBO Results\n\n"
    report += "## Scope\n\n"
    report += "This is a leakage-free leave-one-bearing-out (LOBO) exploratory baseline benchmark on the frozen 32-feature processed dataset. It does not train a neural or physics-guided model. It must not be numerically compared as if it shared the historical fixed nine-bearing test split.\n\n"
    report += "## Protocol\n\n"
    report += "For each of 15 folds, all windows of one bearing are held out and all windows from the other 14 bearings are used to fit the baseline. Windows are ten chronological recordings, stride one, with the final recording RUL as the target. Ridge uses a fold-local StandardScaler. The recording-index diagnostic receives only the observable original recording index at the window end; it receives neither total bearing length nor a normalized age.\n\n"
    report += "## Pooled and Equal-Bearing Metrics\n\n"
    report += "Pooled metrics give each window equal weight. Macro metrics average the 15 per-bearing metrics and therefore give each bearing equal weight. Window overlap means these are not 9,080 independent bearing trajectories.\n\n"
    report += md_table(summary_for_report) + "\n\n"
    report += "## Interpretation Rules\n\n"
    report += "- Vibration-feature models are considered descriptively useful only if they improve both pooled and macro MAE over the training-mean and recording-index-only references; this is not a significance test.\n"
    report += "- A competitive recording-index-only diagnostic would indicate that sampling-progress information can be predictive under this protocol; it would not demonstrate deployable RUL performance because real deployment may not expose comparable observation histories.\n"
    report += "- Physics-guided neural work is not justified or rejected by this one baseline screen alone. Any next experiment should be preregistered against an untouched bearing-level protocol.\n\n"
    report += "## Historical-Result Boundary\n\n"
    report += f"Historical fixed-split baseline results, if present, are retained separately at `{historical}`. They are not pooled with this LOBO output and were not used to choose models here.\n"
    (RUN_DIR / "BASELINE_BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")

    run_manifest = {
        "status": "completed",
        "started_and_finished_in_one_invocation": True,
        "project": str(PROJECT),
        "run_directory": str(RUN_DIR),
        "inputs": {"npz": str(NPZ_PATH), "metadata": str(METADATA_PATH)},
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "protocol": {"type": "leave-one-bearing-out", "n_bearings": len(bearings), "n_windows": len(data["y"]), "sequence_length": SEQUENCE_LENGTH, "seed": SEED},
        "elapsed_seconds": time.perf_counter() - total_started,
        "historical_artifacts_read_only": [str(PROJECT / "audit_2026-10-13"), str(PROJECT / "baseline_benchmark_2026-10-13")],
        "outputs": [
            "BASELINE_BENCHMARK_REPORT.md", "baseline_metrics_by_model.csv", "baseline_metrics_by_bearing.csv",
            "baseline_predictions_lobo.csv", "fold_manifest.csv", "model_configurations.json",
            "run_manifest.json", "DATA_VALIDATION_REPORT.md", "baseline_benchmark.py",
        ],
    }
    (RUN_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": run_manifest["elapsed_seconds"], "rows": len(predictions_frame), "windows": len(data["y"])}, indent=2))


if __name__ == "__main__":
    main()
