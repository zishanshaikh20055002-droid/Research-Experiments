"""Validation checks for preprocessed XJTU-SY feature data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import FAULT_LABELS, TEST_BEARINGS, TRAIN_BEARINGS, validation_folds


def validate_preprocessed_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    stride: int,
    scaler_fitted: bool,
) -> dict[str, object]:
    checks: dict[str, object] = {}

    rul_values = frame["rul"].to_numpy(dtype=float)
    checks["rul_within_0_1"] = bool(np.all((rul_values >= 0.0) & (rul_values <= 1.0)))

    expected_faults = frame["bearing_id"].map(FAULT_LABELS)
    checks["fault_labels_match_mapping"] = bool((expected_faults == frame["fault_label"]).all())

    feature_values = frame[feature_columns].to_numpy(dtype=float)
    checks["features_all_finite"] = bool(np.isfinite(feature_values).all())

    train_set = set(TRAIN_BEARINGS)
    test_set = set(TEST_BEARINGS)
    checks["train_test_bearing_disjoint"] = train_set.isdisjoint(test_set)
    checks["no_test_bearing_in_training_pool"] = bool(
        set(frame.loc[frame["split"] == "train_pool", "bearing_id"]).isdisjoint(test_set)
    )

    fold_checks = []
    for fold in validation_folds():
        fold_train = set(fold.train_bearings)
        fold_val = {fold.validation_bearing}
        fold_test = set(fold.test_bearings)
        fold_checks.append(
            {
                "fold": fold.name,
                "train_bearings": sorted(fold_train),
                "validation_bearing": fold.validation_bearing,
                "test_bearings": sorted(fold_test),
                "train_val_disjoint": fold_train.isdisjoint(fold_val),
                "train_test_disjoint": fold_train.isdisjoint(fold_test),
                "val_test_disjoint": fold_val.isdisjoint(fold_test),
            }
        )
    checks["validation_folds"] = fold_checks
    checks["all_validation_folds_leakage_free"] = all(
        item["train_val_disjoint"] and item["train_test_disjoint"] and item["val_test_disjoint"]
        for item in fold_checks
    )

    sequence_counts: dict[str, int] = {}
    sequence_boundary_ok = True
    chronological_ok = True
    for bearing_id, group in frame.groupby("bearing_id", sort=False):
        ordered = group.sort_values("chronological_position")
        positions = ordered["chronological_position"].to_numpy(dtype=int)
        recording_indices = ordered["recording_index"].to_numpy(dtype=int)
        chronological_ok = chronological_ok and bool(np.array_equal(positions, np.arange(len(ordered))))
        chronological_ok = chronological_ok and bool(np.all(np.diff(recording_indices) > 0))

        count = 0
        for start in range(0, max(0, len(ordered) - sequence_length + 1), stride):
            window = ordered.iloc[start : start + sequence_length]
            if window["bearing_id"].nunique() != 1:
                sequence_boundary_ok = False
            count += 1
        sequence_counts[bearing_id] = count

    checks["chronological_ordering_ok"] = chronological_ok
    checks["sequence_boundary_check"] = sequence_boundary_ok
    checks["sequence_length"] = sequence_length
    checks["sequence_stride"] = stride
    checks["sequence_counts_by_bearing"] = sequence_counts
    checks["scaler_fitted"] = scaler_fitted
    checks["normalization_not_fitted_on_test"] = not scaler_fitted

    checks["passed"] = all(
        [
            checks["rul_within_0_1"],
            checks["fault_labels_match_mapping"],
            checks["features_all_finite"],
            checks["train_test_bearing_disjoint"],
            checks["no_test_bearing_in_training_pool"],
            checks["all_validation_folds_leakage_free"],
            checks["chronological_ordering_ok"],
            checks["sequence_boundary_check"],
            checks["normalization_not_fitted_on_test"],
        ]
    )
    return checks

