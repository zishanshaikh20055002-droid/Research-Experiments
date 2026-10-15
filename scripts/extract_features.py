"""Extract compact XJTU-SY bearing features without touching raw CSV files."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.features import (  # noqa: E402
    EXPECTED_COLUMNS,
    assert_feature_vector_complete,
    extract_recording_features,
    feature_names,
    read_recording,
)
from src.labels import (  # noqa: E402
    CONDITION_BEARINGS,
    FAULT_TO_ID,
    TEST_BEARINGS,
    TRAIN_BEARINGS,
    assert_known_label_mapping,
    condition_for_bearing,
    fault_label_for_bearing,
    validation_folds,
)
from src.preprocessing_checks import validate_preprocessed_frame  # noqa: E402


FILENAME_RE = re.compile(r"^(\d+)\.csv$", re.IGNORECASE)


def numeric_sort_key(path: Path) -> tuple[int, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return sys.maxsize, path.name.lower()
    return int(match.group(1)), path.name


def locate_dataset() -> Path:
    inspection_json = RESEARCH_ROOT / "results" / "dataset_inspection_summary.json"
    if inspection_json.exists():
        summary = json.loads(inspection_json.read_text(encoding="utf-8"))
        path = Path(summary["dataset_path"])
        if path.exists():
            return path

    data_dir = REPO_ROOT / "data"
    for path in data_dir.rglob("*"):
        if path.is_dir() and all((path / condition).is_dir() for condition in CONDITION_BEARINGS):
            return path
    raise FileNotFoundError(f"Could not locate XJTU-SY dataset under {data_dir}")


def read_config(path: Path | None) -> dict[str, Any]:
    defaults = {
        "dataset": {
            "sampling_frequency_hz": 25600,
            "expected_rows_per_csv": 32768,
        },
        "sequence": {
            "length": 10,
            "stride": 1,
        },
    }
    if path is None or not path.exists():
        return defaults
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for section, values in defaults.items():
            loaded.setdefault(section, {})
            for key, value in values.items():
                loaded[section].setdefault(key, value)
        return loaded
    except Exception:
        return defaults


def missing_numeric_indices(csv_files: list[Path]) -> list[int]:
    indices = sorted(
        int(match.group(1))
        for path in csv_files
        if (match := FILENAME_RE.match(path.name)) is not None
    )
    if not indices:
        return []
    expected = set(range(1, max(indices) + 1))
    return sorted(expected.difference(indices))


def split_for_bearing(bearing_id: str) -> str:
    if bearing_id in TRAIN_BEARINGS:
        return "train_pool"
    if bearing_id in TEST_BEARINGS:
        return "test"
    raise KeyError(f"Unknown split for {bearing_id}")


def process_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_known_label_mapping()
    dataset_root = locate_dataset()
    expected_rows = int(config["dataset"]["expected_rows_per_csv"])
    sampling_frequency_hz = float(config["dataset"]["sampling_frequency_hz"])
    names = feature_names()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_records: dict[str, list[int]] = {}

    for condition, bearings in CONDITION_BEARINGS.items():
        for bearing_id in bearings:
            bearing_dir = dataset_root / condition / bearing_id
            csv_files = sorted(bearing_dir.glob("*.csv"), key=numeric_sort_key)
            missing = missing_numeric_indices(csv_files)
            if missing:
                missing_records[bearing_id] = missing
                warnings.append(
                    f"{bearing_id}: missing numeric CSV indices {missing}; existing files are kept in numeric order."
                )

            valid_count = len(csv_files)
            if valid_count < 2:
                raise ValueError(f"{bearing_id} must contain at least two recordings for normalized RUL")

            for chronological_position, csv_path in enumerate(csv_files):
                match = FILENAME_RE.match(csv_path.name)
                if not match:
                    warnings.append(f"{bearing_id}: skipped non-numeric CSV file {csv_path.name}")
                    continue

                recording_index = int(match.group(1))
                frame = read_recording(str(csv_path), expected_rows=expected_rows)
                features = extract_recording_features(frame, sampling_frequency_hz)
                assert_feature_vector_complete(features, names)

                denominator = valid_count - 1
                rul = (denominator - chronological_position) / denominator
                fault_label = fault_label_for_bearing(bearing_id)
                rows.append(
                    {
                        "bearing_id": bearing_id,
                        "bearing_short_id": bearing_id.replace("Bearing", ""),
                        "operating_condition": condition_for_bearing(bearing_id),
                        "recording_file": csv_path.name,
                        "recording_index": recording_index,
                        "chronological_position": chronological_position,
                        "recordings_in_bearing": valid_count,
                        "rul": float(rul),
                        "fault_label": fault_label,
                        "fault_label_id": FAULT_TO_ID[fault_label],
                        "split": split_for_bearing(bearing_id),
                        **features,
                    }
                )
                if len(rows) % 250 == 0:
                    print(f"Processed {len(rows)} recordings...", flush=True)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["bearing_id", "chronological_position"]).reset_index(drop=True)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_root),
        "sampling_frequency_hz": sampling_frequency_hz,
        "expected_rows_per_csv": expected_rows,
        "expected_columns": EXPECTED_COLUMNS,
        "feature_columns": names,
        "feature_count": len(names),
        "missing_records": missing_records,
        "warnings": warnings,
        "train_bearings": TRAIN_BEARINGS,
        "test_bearings": TEST_BEARINGS,
        "validation_strategy": "leave_one_training_bearing_out",
        "validation_folds": [
            {
                "fold": fold.name,
                "train_bearings": fold.train_bearings,
                "validation_bearing": fold.validation_bearing,
                "test_bearings": fold.test_bearings,
            }
            for fold in validation_folds()
        ],
        "normalization": {
            "status": "not_fitted_in_preprocessing",
            "policy": (
                "Fit scalers only on training bearings inside each validation fold; apply "
                "the fitted scaler to validation and test data."
            ),
        },
    }
    return frame, metadata


def build_summary(frame: pd.DataFrame, metadata: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    feature_columns = metadata["feature_columns"]
    rul = frame["rul"].astype(float)
    return {
        "generated_at_utc": metadata["generated_at_utc"],
        "processed_samples": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "feature_matrix_shape": [int(len(frame)), int(len(feature_columns))],
        "samples_per_bearing": {
            key: int(value)
            for key, value in frame.groupby("bearing_id", sort=True).size().to_dict().items()
        },
        "class_sample_counts": {
            key: int(value)
            for key, value in frame.groupby("fault_label", sort=True).size().to_dict().items()
        },
        "class_bearing_counts": {
            key: int(value)
            for key, value in frame.groupby("fault_label")["bearing_id"].nunique().sort_index().to_dict().items()
        },
        "rul_statistics": {
            "min": float(rul.min()),
            "max": float(rul.max()),
            "mean": float(rul.mean()),
            "std": float(rul.std(ddof=1)),
            "median": float(rul.median()),
        },
        "missing_records": metadata["missing_records"],
        "warnings": metadata["warnings"],
        "feature_nan_count": int(frame[feature_columns].isna().sum().sum()),
        "feature_inf_count": int(np.isinf(frame[feature_columns].to_numpy(dtype=float)).sum()),
        "validation_checks": checks,
    }


def write_outputs(frame: pd.DataFrame, metadata: dict[str, Any], checks: dict[str, Any]) -> None:
    processed_dir = RESEARCH_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESEARCH_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = metadata["feature_columns"]
    feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32)
    target_rul = frame["rul"].to_numpy(dtype=np.float32)
    target_fault = frame["fault_label_id"].to_numpy(dtype=np.int64)

    np.savez_compressed(
        processed_dir / "xjtu_features.npz",
        X=feature_matrix,
        rul=target_rul,
        fault_label_id=target_fault,
        feature_names=np.asarray(feature_columns),
        bearing_id=frame["bearing_id"].to_numpy(dtype=str),
        chronological_position=frame["chronological_position"].to_numpy(dtype=np.int64),
        recording_index=frame["recording_index"].to_numpy(dtype=np.int64),
    )

    with gzip.open(processed_dir / "xjtu_features_metadata.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)

    metadata_path = processed_dir / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checks_path = processed_dir / "preprocessing_checks.json"
    checks_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    summary = build_summary(frame, metadata, checks)
    summary_path = results_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "preprocessing_summary.md").write_text(markdown_summary(summary, metadata), encoding="utf-8")


def markdown_summary(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# XJTU-SY Preprocessing Summary",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- Processed samples: `{summary['processed_samples']}`",
        f"- Feature matrix shape: `{summary['feature_matrix_shape']}`",
        f"- Feature count: `{summary['feature_count']}`",
        "",
        "## Feature List",
        "",
    ]
    lines.extend(f"- `{name}`" for name in metadata["feature_columns"])
    lines.extend(["", "## Samples Per Bearing", ""])
    for bearing_id, count in summary["samples_per_bearing"].items():
        lines.append(f"- {bearing_id}: `{count}`")
    lines.extend(["", "## Fault Class Distribution", ""])
    for label, count in summary["class_sample_counts"].items():
        bearings = summary["class_bearing_counts"][label]
        lines.append(f"- {label}: `{count}` samples from `{bearings}` bearings")
    lines.extend(["", "## RUL Statistics", ""])
    for key, value in summary["rul_statistics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Missing Records", ""])
    if summary["missing_records"]:
        for bearing_id, missing in summary["missing_records"].items():
            lines.append(f"- {bearing_id}: `{missing}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Validation Checks", ""])
    for key, value in summary["validation_checks"].items():
        if key in {"validation_folds", "sequence_counts_by_bearing"}:
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=RESEARCH_ROOT / "configs" / "experiment.yaml",
        help="Experiment YAML config path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    frame, metadata = process_dataset(config)
    checks = validate_preprocessed_frame(
        frame,
        feature_columns=metadata["feature_columns"],
        sequence_length=int(config["sequence"]["length"]),
        stride=int(config["sequence"]["stride"]),
        scaler_fitted=False,
    )
    if not checks["passed"]:
        raise RuntimeError(f"Preprocessing validation checks failed: {checks}")

    write_outputs(frame, metadata, checks)
    print(f"Processed samples: {len(frame)}")
    print(f"Feature matrix shape: {len(frame)} x {len(metadata['feature_columns'])}")
    print(f"Outputs: {RESEARCH_ROOT / 'data' / 'processed'}")
    print(f"Summary: {RESEARCH_ROOT / 'results' / 'preprocessing_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

