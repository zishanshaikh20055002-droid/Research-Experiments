"""Regression metrics for RUL prediction."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(errors)) / denominator) if denominator > 0.0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "predictions_outside_0_1": int(np.sum((y_pred < 0.0) | (y_pred > 1.0))),
    }

