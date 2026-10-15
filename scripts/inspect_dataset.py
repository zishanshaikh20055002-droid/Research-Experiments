"""Inspect the XJTU-SY bearing dataset without modifying raw files.

The script locates the dataset under the repository data directory, validates
all bearing CSV files, and writes Markdown/JSON summaries into the isolated
research results directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_CONDITIONS = {
    "35Hz12kN": ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"],
    "37.5Hz11kN": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
    "40Hz10kN": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
}
EXPECTED_COLUMNS = ["Horizontal_vibration_signals", "Vertical_vibration_signals"]
EXPECTED_ROWS = 32768
FILENAME_RE = re.compile(r"^(\d+)\.csv$", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def research_root() -> Path:
    return Path(__file__).resolve().parents[1]


def locate_xjtu_dataset(data_dir: Path) -> Path:
    candidates: list[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_dir():
            continue
        condition_dirs = [path / condition for condition in EXPECTED_CONDITIONS]
        if all(condition_dir.is_dir() for condition_dir in condition_dirs):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Could not locate XJTU-SY dataset under {data_dir}")

    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def numeric_csv_sort_key(path: Path) -> tuple[int, str]:
    match = FILENAME_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name.lower()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(path),
        "rows": 0,
        "columns": [],
        "column_count": 0,
        "has_expected_columns": False,
        "malformed": False,
        "malformed_reason": None,
        "nan_count": 0,
        "inf_count": 0,
        "non_numeric_count": 0,
        "empty_value_count": 0,
    }
    try:
        header_frame = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        result["columns"] = list(header_frame.columns)
        result["column_count"] = len(header_frame.columns)
        result["has_expected_columns"] = result["columns"] == EXPECTED_COLUMNS

        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=np.float64)
        result["rows"] = int(len(frame))

        values = frame.to_numpy(dtype=float, na_value=np.nan)
        result["nan_count"] = int(np.isnan(values).sum())
        result["inf_count"] = int(np.isinf(values).sum())
    except ValueError as exc:
        result["malformed"] = True
        result["malformed_reason"] = repr(exc)
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
            result["columns"] = list(frame.columns)
            result["column_count"] = len(frame.columns)
            result["has_expected_columns"] = result["columns"] == EXPECTED_COLUMNS
            result["rows"] = int(len(frame))
            stripped = frame.apply(lambda column: column.str.strip())
            result["empty_value_count"] = int((stripped == "").sum().sum())
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            original_missing = frame.isna() | (stripped == "")
            result["non_numeric_count"] = int((numeric.isna() & ~original_missing).sum().sum())
            values = numeric.to_numpy(dtype=float, na_value=np.nan)
            result["nan_count"] = int(np.isnan(values).sum())
            result["inf_count"] = int(np.isinf(values).sum())
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001 - this is an inspection report.
        result["malformed"] = True
        result["malformed_reason"] = repr(exc)
    return result


def detect_gpu() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "devices": [],
        "nvidia_smi": None,
    }
    try:
        import torch  # type: ignore

        gpu["torch_available"] = True
        gpu["cuda_available"] = bool(torch.cuda.is_available())
        gpu["cuda_device_count"] = int(torch.cuda.device_count()) if gpu["cuda_available"] else 0
        if gpu["cuda_available"]:
            for index in range(gpu["cuda_device_count"]):
                props = torch.cuda.get_device_properties(index)
                gpu["devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - absence of torch is report data.
        gpu["torch_error"] = repr(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu["nvidia_smi"] = completed.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            gpu["nvidia_smi_error"] = repr(exc)
    return gpu


def summarize_dataset(dataset_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root()),
        "dataset_path": str(dataset_root),
        "expected_rows_per_csv": EXPECTED_ROWS,
        "expected_columns": EXPECTED_COLUMNS,
        "bearings": [],
        "missing_bearing_folders": [],
        "unexpected_bearing_folders": [],
        "total_recordings": 0,
        "total_csv_bytes": 0,
        "approx_raw_data_size_gb": 0.0,
        "malformed_csvs": [],
        "shape_problems": [],
        "column_problems": [],
        "value_problems": [],
        "missing_files": [],
        "duplicate_or_suspicious_files": [],
        "filename_ordering_notes": [],
        "gpu": detect_gpu(),
    }

    expected_bearings = {
        bearing: condition
        for condition, bearings in EXPECTED_CONDITIONS.items()
        for bearing in bearings
    }
    seen_bearings: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)

    for condition, bearings in EXPECTED_CONDITIONS.items():
        condition_dir = dataset_root / condition
        actual_dirs = [p.name for p in condition_dir.iterdir() if p.is_dir()] if condition_dir.is_dir() else []
        for actual in actual_dirs:
            if actual not in bearings:
                summary["unexpected_bearing_folders"].append(str(condition_dir / actual))

        for bearing in bearings:
            bearing_dir = condition_dir / bearing
            if not bearing_dir.is_dir():
                summary["missing_bearing_folders"].append(str(bearing_dir))
                continue
            seen_bearings.add(bearing)

            csv_files = sorted(bearing_dir.glob("*.csv"), key=numeric_csv_sort_key)
            indices = [numeric_csv_sort_key(path)[0] for path in csv_files if FILENAME_RE.match(path.name)]
            expected_indices = list(range(1, max(indices) + 1)) if indices else []
            missing_indices = sorted(set(expected_indices) - set(indices))
            non_numeric = [path.name for path in csv_files if not FILENAME_RE.match(path.name)]
            lexicographic = [path.name for path in sorted(csv_files, key=lambda p: p.name.lower())]
            numeric = [path.name for path in csv_files]
            if lexicographic != numeric:
                summary["filename_ordering_notes"].append(
                    {
                        "bearing": bearing,
                        "note": "Lexicographic order differs from numeric chronological order; use numeric filename sorting.",
                        "first_numeric": numeric[:5],
                        "first_lexicographic": lexicographic[:5],
                    }
                )

            representative: dict[str, Any] | None = None
            row_counts: Counter[int] = Counter()
            column_signatures: Counter[tuple[str, ...]] = Counter()

            for csv_path in csv_files:
                file_size = csv_path.stat().st_size
                summary["total_csv_bytes"] += file_size
                hashes[file_sha256(csv_path)].append(str(csv_path))

                csv_result = inspect_csv(csv_path)
                if representative is None:
                    representative = csv_result
                row_counts[int(csv_result["rows"])] += 1
                column_signatures[tuple(csv_result["columns"])] += 1

                if csv_result["malformed"]:
                    summary["malformed_csvs"].append(csv_result)
                if csv_result["rows"] != EXPECTED_ROWS or csv_result["column_count"] != 2:
                    summary["shape_problems"].append(csv_result)
                if not csv_result["has_expected_columns"]:
                    summary["column_problems"].append(csv_result)
                if (
                    csv_result["nan_count"]
                    or csv_result["inf_count"]
                    or csv_result["non_numeric_count"]
                    or csv_result["empty_value_count"]
                ):
                    summary["value_problems"].append(csv_result)

            if missing_indices:
                summary["missing_files"].append(
                    {
                        "bearing": bearing,
                        "missing_numeric_csv_indices": missing_indices,
                    }
                )
            if non_numeric:
                summary["duplicate_or_suspicious_files"].append(
                    {
                        "bearing": bearing,
                        "non_numeric_csv_names": non_numeric,
                    }
                )

            bearing_summary = {
                "bearing": bearing,
                "condition": condition,
                "path": str(bearing_dir),
                "recording_count": len(csv_files),
                "first_file": csv_files[0].name if csv_files else None,
                "last_file": csv_files[-1].name if csv_files else None,
                "missing_numeric_csv_indices": missing_indices,
                "non_numeric_csv_names": non_numeric,
                "row_count_distribution": dict(sorted(row_counts.items())),
                "column_signatures": {", ".join(signature): count for signature, count in column_signatures.items()},
                "representative_csv": representative,
            }
            summary["bearings"].append(bearing_summary)
            summary["total_recordings"] += len(csv_files)

    for bearing, condition in expected_bearings.items():
        if bearing not in seen_bearings:
            summary["missing_bearing_folders"].append(str(dataset_root / condition / bearing))

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    for paths in duplicate_groups:
        summary["duplicate_or_suspicious_files"].append({"identical_sha256_files": paths})

    summary["approx_raw_data_size_gb"] = round(summary["total_csv_bytes"] / (1024**3), 3)
    return summary


def markdown_summary(summary: dict[str, Any]) -> str:
    def table_cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "# XJTU-SY Dataset Inspection Summary",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- Repository root: `{summary['repository_root']}`",
        f"- Dataset path: `{summary['dataset_path']}`",
        f"- Total recordings: `{summary['total_recordings']}`",
        f"- Approximate raw CSV size: `{summary['approx_raw_data_size_gb']} GB`",
        "",
        "## Bearings",
        "",
        "| Bearing | Condition | Recordings | Row counts | Column signatures | Missing files |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for bearing in summary["bearings"]:
        row_counts = ", ".join(f"{rows}: {count}" for rows, count in bearing["row_count_distribution"].items())
        columns = "; ".join(
            f"{signature}: {count}" for signature, count in bearing["column_signatures"].items()
        )
        missing = ", ".join(map(str, bearing["missing_numeric_csv_indices"])) or "None"
        lines.append(
            f"| {table_cell(bearing['bearing'])} | {table_cell(bearing['condition'])} | "
            f"{bearing['recording_count']} | {table_cell(row_counts)} | "
            f"{table_cell(columns)} | {table_cell(missing)} |"
        )

    def issue_count(name: str) -> str:
        return f"- {name}: `{len(summary[name])}`"

    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            issue_count("missing_bearing_folders"),
            issue_count("unexpected_bearing_folders"),
            issue_count("missing_files"),
            issue_count("malformed_csvs"),
            issue_count("shape_problems"),
            issue_count("column_problems"),
            issue_count("value_problems"),
            issue_count("duplicate_or_suspicious_files"),
            issue_count("filename_ordering_notes"),
            "",
            "## Proposed Split",
            "",
            "- Train: Bearing1_1, Bearing1_2, Bearing2_1, Bearing2_2, Bearing3_1, Bearing3_2",
            "- Test: Bearing1_3, Bearing1_4, Bearing1_5, Bearing2_3, Bearing2_4, Bearing2_5, Bearing3_3, Bearing3_4, Bearing3_5",
            "- Validation: leave-one-training-bearing-out over the six training bearings; never sample-split within a bearing.",
            "",
            "## GPU",
            "",
            f"- PyTorch available: `{summary['gpu'].get('torch_available')}`",
            f"- CUDA available via PyTorch: `{summary['gpu'].get('cuda_available')}`",
            f"- CUDA device count: `{summary['gpu'].get('cuda_device_count')}`",
        ]
    )
    for device in summary["gpu"].get("devices", []):
        lines.append(
            f"- CUDA device {device['index']}: {device['name']} ({device['total_memory_gb']} GB)"
        )
    if summary["gpu"].get("nvidia_smi"):
        lines.append(f"- nvidia-smi: `{summary['gpu']['nvidia_smi']}`")
    if summary["gpu"].get("torch_error"):
        lines.append(f"- PyTorch detection error: `{summary['gpu']['torch_error']}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    root = repo_root()
    dataset_root = locate_xjtu_dataset(root / "data")
    summary = summarize_dataset(dataset_root)

    results_dir = research_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "dataset_inspection_summary.json"
    md_path = results_dir / "dataset_inspection_summary.md"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(markdown_summary(summary), encoding="utf-8")

    print(f"Dataset path: {summary['dataset_path']}")
    print(f"Bearings found: {len(summary['bearings'])}")
    print(f"Total recordings: {summary['total_recordings']}")
    print(f"Approximate raw CSV size: {summary['approx_raw_data_size_gb']} GB")
    print(f"Markdown summary: {md_path}")
    print(f"JSON summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
