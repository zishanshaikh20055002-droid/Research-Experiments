"""Bearing labels and split definitions for XJTU-SY."""

from __future__ import annotations

from dataclasses import dataclass


CONDITION_BEARINGS = {
    "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
    "37.5Hz11kN": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
    "40Hz10kN": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
}

FAULT_LABELS = {
    "Bearing1_1": "Outer",
    "Bearing1_2": "Outer",
    "Bearing1_3": "Outer",
    "Bearing2_2": "Outer",
    "Bearing2_4": "Outer",
    "Bearing2_5": "Outer",
    "Bearing3_1": "Outer",
    "Bearing3_5": "Outer",
    "Bearing2_1": "Inner",
    "Bearing3_3": "Inner",
    "Bearing3_4": "Inner",
    "Bearing1_4": "Cage",
    "Bearing2_3": "Cage",
    "Bearing1_5": "Mixed",
    "Bearing3_2": "Mixed",
}

FAULT_TO_ID = {"Outer": 0, "Inner": 1, "Cage": 2, "Mixed": 3}
ID_TO_FAULT = {value: key for key, value in FAULT_TO_ID.items()}

TRAIN_BEARINGS = [
    "Bearing1_1",
    "Bearing1_2",
    "Bearing2_1",
    "Bearing2_2",
    "Bearing3_1",
    "Bearing3_2",
]

TEST_BEARINGS = [
    "Bearing1_3",
    "Bearing1_4",
    "Bearing1_5",
    "Bearing2_3",
    "Bearing2_4",
    "Bearing2_5",
    "Bearing3_3",
    "Bearing3_4",
    "Bearing3_5",
]


@dataclass(frozen=True)
class ValidationFold:
    name: str
    train_bearings: list[str]
    validation_bearing: str
    test_bearings: list[str]


def bearing_short_id(bearing_id: str) -> str:
    return bearing_id.replace("Bearing", "")


def condition_for_bearing(bearing_id: str) -> str:
    for condition, bearings in CONDITION_BEARINGS.items():
        if bearing_id in bearings:
            return condition
    raise KeyError(f"Unknown bearing: {bearing_id}")


def fault_label_for_bearing(bearing_id: str) -> str:
    return FAULT_LABELS[bearing_id]


def validation_folds() -> list[ValidationFold]:
    return [
        ValidationFold(
            name=f"val_{bearing_id}",
            train_bearings=[candidate for candidate in TRAIN_BEARINGS if candidate != bearing_id],
            validation_bearing=bearing_id,
            test_bearings=list(TEST_BEARINGS),
        )
        for bearing_id in TRAIN_BEARINGS
    ]


def assert_known_label_mapping() -> None:
    expected = {bearing for bearings in CONDITION_BEARINGS.values() for bearing in bearings}
    missing = expected.difference(FAULT_LABELS)
    extra = set(FAULT_LABELS).difference(expected)
    if missing or extra:
        raise ValueError(f"Fault label mapping mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")

