"""Run the preregistered unshuffled M3 lambda sweep in a compatible PyTorch environment.

This script intentionally does not install dependencies. It reuses the audited
final multiseed implementation and changes only ``Config.lambda_mono``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 123, 2024)
LAMBDAS = (1.0, 10.0)


def load_runner(project_root: Path):
    path = project_root / "scripts" / "run_multiseed.py"
    spec = importlib.util.spec_from_file_location("audited_run_multiseed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audited runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validation_evaluation(runner, model, frame, features, cfg, device, out: Path):
    import torch
    from torch.utils.data import DataLoader

    _, _, validation, _, _ = runner.make_data(frame, features, cfg)
    loader = DataLoader(runner.MultiDataset(validation), batch_size=cfg.batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.no_grad():
        for xb, *_ in loader:
            pred_rul, _ = model(xb.to(device))
            predictions.append(pred_rul.cpu().numpy().reshape(-1))
    pred = np.concatenate(predictions)
    table = pd.DataFrame({
        "bearing_id": validation.bearing_id,
        "start_position": validation.start_position,
        "end_position": validation.end_position,
        "true_rul": validation.y,
        "pred_rul": pred,
    })
    table.to_csv(out / "validation_predictions.csv", index=False)
    return {
        **runner.regression_metrics(validation.y, pred),
        **runner.mono_metrics(table),
        "sequence_count": int(len(table)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch import failed; no run started: {exc!r}") from exc

    runner = load_runner(project_root)
    metadata = json.loads((project_root / "data" / "processed" / "preprocessing_metadata.json").read_text())
    frame = runner.load_processed_metadata(project_root / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    features = metadata["feature_columns"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "status": "running",
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "source_runner": str(project_root / "scripts" / "run_multiseed.py"),
        "selection_rule": "lowest mean validation RUL MAE across completed seeds",
        "early_stopping_rule": "validation total loss, matching historical M3-U",
        "run_order": [{"seed": seed, "lambda_mono": value} for seed in SEEDS for value in LAMBDAS],
        "runs": [],
    }
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    original_finish = runner.finish

    for seed in SEEDS:
        for lambda_mono in LAMBDAS:
            run_out = output / f"lambda_{lambda_mono:g}" / f"seed_{seed}"
            run_out.mkdir(parents=True, exist_ok=False)
            cfg = runner.Config(lambda_mono=lambda_mono, patience=8)
            config = {**cfg.__dict__, "seed": seed, "model": "M3-U", "train_shuffle": False}
            (run_out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            started = time.perf_counter()

            def finish_with_validation(model_name, seed_value, out, model, weights, test_b, cfg_value,
                                       feature_names, scaler, rows, best_epoch, best, train_time, device_value):
                metrics = original_finish(model_name, seed_value, out, model, weights, test_b, cfg_value,
                                          feature_names, scaler, rows, best_epoch, best, train_time, device_value)
                metrics["validation"] = validation_evaluation(
                    runner, model, frame, feature_names, cfg_value, device_value, out
                )
                (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                return metrics

            runner.finish = finish_with_validation
            try:
                metrics = runner.run_multi("m3", seed, frame, features, cfg, device, run_out)
            finally:
                runner.finish = original_finish
            elapsed = time.perf_counter() - started
            record = {
                "seed": seed,
                "lambda_mono": lambda_mono,
                "output": str(run_out),
                "elapsed_seconds": elapsed,
                "best_epoch": metrics["best_epoch"],
                "best_validation_total_loss": metrics["best_validation_loss"],
                "validation_mae": metrics["validation"]["mae"],
                "validation_rmse": metrics["validation"]["rmse"],
                "validation_violation_rate": metrics["validation"]["mono_violation_rate"],
            }
            (run_out / "run_log.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            manifest["runs"].append(record)
            (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)

    runs = pd.DataFrame(manifest["runs"])
    summary = runs.groupby("lambda_mono", as_index=False).agg(
        seed_count=("seed", "count"),
        validation_mae_mean=("validation_mae", "mean"),
        validation_mae_sd=("validation_mae", "std"),
        validation_violation_rate_mean=("validation_violation_rate", "mean"),
        validation_violation_rate_sd=("validation_violation_rate", "std"),
    )
    summary.to_csv(output / "validation_selection_summary.csv", index=False)
    selected = float(summary.sort_values("validation_mae_mean").iloc[0].lambda_mono)
    manifest.update({"status": "complete", "selected_lambda": selected, "completed_unix": time.time()})
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
