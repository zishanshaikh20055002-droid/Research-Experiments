"""Train M1: BiLSTM baseline for normalized RUL regression only."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.dataset import (  # noqa: E402
    RULSequenceDataset,
    SequenceBundle,
    fit_scaler,
    load_processed_metadata,
    make_sequences,
    transform_features,
)
from src.labels import TEST_BEARINGS, TRAIN_BEARINGS, validation_folds  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.models import BiLSTMRULRegressor  # noqa: E402


@dataclass(frozen=True)
class M1Config:
    model: str = "M1"
    seed: int = 42
    sequence_length: int = 10
    stride: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    grad_clip_norm: float = 1.0
    loss: str = "mse"
    optimizer: str = "adam"
    num_workers: int = 0


def read_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def build_config(config_path: Path | None) -> M1Config:
    raw = read_yaml_config(config_path)
    sequence = raw.get("sequence", {})
    m1 = raw.get("m1", {})
    return M1Config(
        seed=int(m1.get("seed", 42)),
        sequence_length=int(sequence.get("length", m1.get("sequence_length", 10))),
        stride=int(sequence.get("stride", m1.get("stride", 1))),
        hidden_size=int(m1.get("hidden_size", 64)),
        num_layers=int(m1.get("num_layers", 1)),
        dropout=float(m1.get("dropout", 0.0)),
        batch_size=int(m1.get("batch_size", 32)),
        learning_rate=float(m1.get("learning_rate", 1e-3)),
        max_epochs=int(m1.get("max_epochs", 100)),
        patience=int(m1.get("patience", 15)),
        grad_clip_norm=float(m1.get("grad_clip_norm", 1.0)),
        loss=str(m1.get("loss", "mse")).lower(),
        num_workers=int(m1.get("num_workers", 0)),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
        "nvidia_smi": None,
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            status["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                }
            )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            import subprocess

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
            status["nvidia_smi"] = completed.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            status["nvidia_smi_error"] = repr(exc)
    return status


def loss_function(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported M1 loss: {name}")


def dataloader(bundle: SequenceBundle, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        RULSequenceDataset(bundle),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def dry_run(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.train()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return {
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu().item()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            prediction = model(x)
            loss = criterion(prediction, y)
            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            output = model(x.to(device, non_blocking=True)).detach().cpu().numpy().reshape(-1)
            predictions.append(output)
    return np.concatenate(predictions)


def plot_loss(history: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("M1 Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["true_rul"], predictions["pred_rul"], s=10, alpha=0.45)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("True normalized RUL")
    plt.ylabel("Predicted normalized RUL")
    plt.title("M1 Test Predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_chronological(predictions: pd.DataFrame, output_path: Path, max_bearings: int = 4) -> None:
    bearings = list(predictions["bearing_id"].drop_duplicates())[:max_bearings]
    fig, axes = plt.subplots(len(bearings), 1, figsize=(9, max(3, 2.4 * len(bearings))), sharex=False)
    if len(bearings) == 1:
        axes = [axes]
    for axis, bearing_id in zip(axes, bearings):
        group = predictions[predictions["bearing_id"] == bearing_id].sort_values("end_position")
        axis.plot(group["end_position"], group["true_rul"], label="True RUL", linewidth=1.5)
        axis.plot(group["end_position"], group["pred_rul"], label="Predicted RUL", linewidth=1.2)
        axis.set_title(str(bearing_id))
        axis.set_ylabel("RUL")
        axis.set_ylim(-0.05, 1.05)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Chronological sequence end position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_fold(
    fold: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: M1Config,
    device: torch.device,
    output_dir: Path,
    dry_run_required: bool,
) -> dict[str, Any]:
    scaler = fit_scaler(frame, feature_columns, fold.train_bearings)
    transformed = transform_features(frame, feature_columns, scaler)
    train_bundle = make_sequences(frame, transformed, fold.train_bearings, config.sequence_length, config.stride)
    val_bundle = make_sequences(frame, transformed, [fold.validation_bearing], config.sequence_length, config.stride)

    train_loader = dataloader(train_bundle, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = dataloader(val_bundle, config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = BiLSTMRULRegressor(
        input_size=len(feature_columns),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = loss_function(config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    dry_info = dry_run(model, criterion, train_loader, device) if dry_run_required else None

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss = run_epoch(model, val_loader, criterion, device)
        history_rows.append({"fold": fold.name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_time_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for {fold.name}")

    checkpoint_path = output_dir / f"m1_{fold.name}_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "fold": {
                "name": fold.name,
                "train_bearings": fold.train_bearings,
                "validation_bearing": fold.validation_bearing,
                "test_bearings": fold.test_bearings,
            },
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        checkpoint_path,
    )

    return {
        "fold": fold.name,
        "validation_bearing": fold.validation_bearing,
        "train_sequence_count": int(len(train_bundle.y)),
        "validation_sequence_count": int(len(val_bundle.y)),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_rows)),
        "training_time_seconds": float(training_time_seconds),
        "history": history_rows,
        "checkpoint_path": str(checkpoint_path),
        "dry_run": dry_info,
    }


def evaluate_selected(
    selected: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: M1Config,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint = torch.load(selected["checkpoint_path"], map_location="cpu")
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    transformed = ((features - scaler_mean) / scaler_scale).astype(np.float32)

    test_bundle = make_sequences(frame, transformed, TEST_BEARINGS, config.sequence_length, config.stride)
    test_loader = dataloader(test_bundle, config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = BiLSTMRULRegressor(
        input_size=len(feature_columns),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict(model, test_loader, device)

    predictions_frame = pd.DataFrame(
        {
            "bearing_id": test_bundle.bearing_id,
            "start_position": test_bundle.start_position,
            "end_position": test_bundle.end_position,
            "recording_index_end": test_bundle.recording_index_end,
            "true_rul": test_bundle.y,
            "pred_rul": predictions,
        }
    )
    metrics = regression_metrics(test_bundle.y, predictions)
    metrics["test_sequence_count"] = int(len(test_bundle.y))
    predictions_frame.to_csv(output_dir / "m1_test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "m1_metrics.csv", index=False)
    (output_dir / "m1_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return predictions_frame, metrics


def experiment_report(
    output_dir: Path,
    config: M1Config,
    torch_info: dict[str, Any],
    selected: dict[str, Any],
    metrics: dict[str, Any],
    fold_results: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    sequence_counts = {
        result["fold"]: {"train": result["train_sequence_count"], "validation": result["validation_sequence_count"]}
        for result in fold_results
    }
    report = {
        "model": "M1",
        "architecture": {
            "encoder": "BiLSTM",
            "input_size": 32,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "bidirectional": True,
            "head": "Linear(128->64), ReLU, Linear(64->1), Sigmoid",
            "tasks": ["RUL regression only"],
        },
        "hyperparameters": asdict(config),
        "torch": torch_info,
        "split": {"training_pool": TRAIN_BEARINGS, "test": TEST_BEARINGS, "validation": "leave-one-training-bearing-out"},
        "sequence_counts": sequence_counts,
        "selected_fold": selected["fold"],
        "training_time_seconds_selected_fold": selected["training_time_seconds"],
        "best_validation_loss": selected["best_val_loss"],
        "best_epoch": selected["best_epoch"],
        "epochs_completed_selected_fold": selected["epochs_completed"],
        "test_metrics": metrics,
        "warnings": warnings,
    }
    (output_dir / "m1_experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# M1 Baseline Experiment Report",
        "",
        "M1 is a single-task BiLSTM RUL regressor. It does not include fault classification, monotonicity loss, or adaptive task weighting.",
        "",
        "## Architecture",
        "",
        f"- Input tensor: `(batch, {config.sequence_length}, 32)`",
        f"- Encoder: 1-layer bidirectional LSTM, hidden size `{config.hidden_size}`",
        "- Output head: linear, ReLU, linear, sigmoid",
        "- Output: one normalized RUL scalar in `[0, 1]`",
        f"- Loss: `{config.loss}`",
        "",
        "## Split",
        "",
        f"- Training pool: `{TRAIN_BEARINGS}`",
        f"- Test bearings: `{TEST_BEARINGS}`",
        "- Validation: leave-one-training-bearing-out over training pool",
        "",
        "## Results",
        "",
        f"- Selected fold: `{selected['fold']}`",
        f"- Best validation loss: `{selected['best_val_loss']}`",
        f"- Best epoch: `{selected['best_epoch']}`",
        f"- Epochs completed: `{selected['epochs_completed']}`",
        f"- Training time, selected fold seconds: `{selected['training_time_seconds']}`",
        f"- Test MAE: `{metrics['mae']}`",
        f"- Test RMSE: `{metrics['rmse']}`",
        f"- Test R2: `{metrics['r2']}`",
        f"- Prediction range: `{metrics['prediction_min']}` to `{metrics['prediction_max']}`",
        f"- Predictions outside [0,1]: `{metrics['predictions_outside_0_1']}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    (output_dir / "m1_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RESEARCH_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--model", default="M1", choices=["M1"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args.config)
    set_seed(config.seed)

    metadata_path = RESEARCH_ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz"
    preprocessing_metadata_path = RESEARCH_ROOT / "data" / "processed" / "preprocessing_metadata.json"
    preprocessing_metadata = json.loads(preprocessing_metadata_path.read_text(encoding="utf-8"))
    feature_columns = preprocessing_metadata["feature_columns"]
    frame = load_processed_metadata(metadata_path)

    output_dir = RESEARCH_ROOT / "results" / "m1"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    torch_info = torch_status()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings = list(preprocessing_metadata.get("warnings", []))
    if not torch.cuda.is_available():
        warnings.append("CUDA is not available to PyTorch; M1 trained on CPU.")

    print(f"PyTorch: {torch_info['torch_version']}")
    print(f"CUDA available: {torch_info['cuda_available']}")
    print(f"Device: {device}")

    fold_results: list[dict[str, Any]] = []
    for index, fold in enumerate(validation_folds()):
        print(f"Training {fold.name} with validation bearing {fold.validation_bearing}", flush=True)
        result = train_fold(
            fold=fold,
            frame=frame,
            feature_columns=feature_columns,
            config=config,
            device=device,
            output_dir=output_dir,
            dry_run_required=index == 0,
        )
        print(f"{fold.name}: best val {result['best_val_loss']:.6f} at epoch {result['best_epoch']}", flush=True)
        fold_results.append(result)

    selected = min(fold_results, key=lambda item: item["best_val_loss"])
    shutil.copyfile(selected["checkpoint_path"], output_dir / "m1_best.pt")

    histories = [pd.DataFrame(result["history"]) for result in fold_results]
    history_frame = pd.concat(histories, ignore_index=True)
    history_frame.to_csv(output_dir / "m1_training_history.csv", index=False)
    selected_history = history_frame[history_frame["fold"] == selected["fold"]]
    plot_loss(selected_history, plots_dir / "m1_loss_curve.png")

    predictions_frame, metrics = evaluate_selected(selected, frame, feature_columns, config, device, output_dir)
    plot_scatter(predictions_frame, plots_dir / "m1_pred_vs_true_scatter.png")
    plot_chronological(predictions_frame, plots_dir / "m1_chronological_test_curves.png")

    config_record = {
        "config": asdict(config),
        "torch": torch_info,
        "selected_fold": selected["fold"],
        "dry_run": fold_results[0]["dry_run"],
    }
    (output_dir / "m1_config_used.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    fold_results_light = [{key: value for key, value in result.items() if key != "history"} for result in fold_results]
    (output_dir / "m1_fold_results.json").write_text(json.dumps(fold_results_light, indent=2), encoding="utf-8")

    experiment_report(output_dir, config, torch_info, selected, metrics, fold_results, warnings)

    print(f"Selected fold: {selected['fold']}")
    print(f"Best validation loss: {selected['best_val_loss']}")
    print(f"Test MAE: {metrics['mae']}")
    print(f"Test RMSE: {metrics['rmse']}")
    print(f"Test R2: {metrics['r2']}")
    print(f"Prediction range: {metrics['prediction_min']} to {metrics['prediction_max']}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

