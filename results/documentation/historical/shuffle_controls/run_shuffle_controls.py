"""Run matched unshuffled M2/M4 controls through the final multiseed trainer."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "shuffle_controls"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_multiseed as trainer  # noqa: E402


SEEDS = [42, 123, 2024]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cfg = trainer.Config()
    meta = json.loads((ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text())
    frame = trainer.load_processed_metadata(ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    features = meta["feature_columns"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_loader = trainer.DataLoader

    def force_unshuffled(dataset, *args, **kwargs):
        requested = kwargs.get("shuffle", False)
        if requested:
            kwargs["shuffle"] = False
        return original_loader(dataset, *args, **kwargs)

    trainer.DataLoader = force_unshuffled

    manifest = {
        "purpose": "Exploratory matched-batching sensitivity analysis on previously inspected test data",
        "seeds": SEEDS,
        "device": str(device),
        "torch_version": torch.__version__,
        "controls": ["M2-unshuffled", "M4-unshuffled"],
        "changed_setting": "training DataLoader shuffle=True to shuffle=False",
        "validation_test_loaders": "shuffle=False, unchanged",
        "config": cfg.__dict__,
        "fold": {
            "train_bearings": ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing3_1", "Bearing3_2"],
            "validation_bearing": cfg.validation_bearing,
            "test_bearings": trainer.TEST_BEARINGS,
        },
        "feature_count": len(features),
        "sequence_length": cfg.sequence_length,
        "stride": cfg.stride,
        "source_entry_points": {
            "M2/M3/M5": str(ROOT / "scripts" / "run_multiseed.py"),
            "M4": str(ROOT / "scripts" / "run_m4_multiseed.py"),
        },
        "source_sha256": {
            "run_multiseed.py": sha256(ROOT / "scripts" / "run_multiseed.py"),
            "run_m4_multiseed.py": sha256(ROOT / "scripts" / "run_m4_multiseed.py"),
        },
        "objective_by_model": {
            "M2-unshuffled": "MSE(RUL) + CrossEntropy(fault)",
            "M4-unshuffled": "exp(-s_R)*MSE + s_R + exp(-s_F)*CrossEntropy + s_F",
        },
        "comparison_objective_differences": {
            "M2-unshuffled_vs_M3": "monotonicity penalty present only in M3",
            "M4-unshuffled_vs_M5": "monotonicity penalty present only in M5",
        },
        "runs": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "shuffle_control_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for model_name in ("m2", "m4"):
        for seed in SEEDS:
            target = OUT / f"{model_name.upper()}-unshuffled" / f"seed_{seed}"
            print(f"Starting {model_name.upper()}-unshuffled seed {seed} on {device}", flush=True)
            metrics = trainer.run_multi(model_name, seed, frame, features, cfg, device, target)
            manifest["runs"].append({
                "model": f"{model_name.upper()}-unshuffled",
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "best_validation_loss": metrics["best_validation_loss"],
                "training_time_seconds": metrics["training_time_seconds"],
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
                "output_directory": str(target),
            })
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"Completed {model_name.upper()}-unshuffled seed {seed}: "
                f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}",
                flush=True,
            )

    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"All six controls complete. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
