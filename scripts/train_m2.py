"""Train M2: clean multi-task RUL + fault classification baseline."""

from __future__ import annotations

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as m1_train  # noqa: E402
from src.dataset import SequenceBundle, fit_scaler, load_processed_metadata, make_sequences, transform_features  # noqa: E402
from src.labels import FAULT_TO_ID, ID_TO_FAULT, TEST_BEARINGS, validation_folds  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.models import BiLSTMMultiTaskRegressorClassifier  # noqa: E402


M1_REFERENCE = {"mae": 0.306988, "rmse": 0.401279, "r2": -1.008124}


@dataclass(frozen=True)
class M2Config:
    experiment: str = "M2"
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
    selected_validation_bearing: str = "Bearing2_2"
    num_workers: int = 0


class MultiTaskSequenceDataset(Dataset):
    def __init__(self, bundle: SequenceBundle):
        self.x = torch.as_tensor(bundle.x, dtype=torch.float32)
        self.rul = torch.as_tensor(bundle.y, dtype=torch.float32).view(-1, 1)
        self.fault = torch.as_tensor(bundle.fault_label_id, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.rul[index], self.fault[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataloader(bundle: SequenceBundle, config: M2Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        MultiTaskSequenceDataset(bundle),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    rul_loss_fn: nn.Module,
    fault_loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"rul_loss": 0.0, "fault_loss": 0.0, "total_loss": 0.0, "count": 0.0}
    for x, rul, fault in loader:
        x = x.to(device, non_blocking=True)
        rul = rul.to(device, non_blocking=True)
        fault = fault.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            pred_rul, fault_logits = model(x)
            rul_loss = rul_loss_fn(pred_rul, rul)
            fault_loss = fault_loss_fn(fault_logits, fault)
            total_loss = rul_loss + fault_loss
            if train_mode:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        batch_size = int(x.shape[0])
        totals["rul_loss"] += float(rul_loss.detach().cpu().item()) * batch_size
        totals["fault_loss"] += float(fault_loss.detach().cpu().item()) * batch_size
        totals["total_loss"] += float(total_loss.detach().cpu().item()) * batch_size
        totals["count"] += batch_size
    count = max(totals.pop("count"), 1.0)
    return {key: value / count for key, value in totals.items()}


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    rul_predictions: list[np.ndarray] = []
    fault_predictions: list[np.ndarray] = []
    fault_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for x, _, _ in loader:
            pred_rul, fault_logits = model(x.to(device, non_blocking=True))
            probabilities = torch.softmax(fault_logits, dim=1)
            rul_predictions.append(pred_rul.detach().cpu().numpy().reshape(-1))
            fault_predictions.append(torch.argmax(probabilities, dim=1).detach().cpu().numpy())
            fault_probabilities.append(probabilities.detach().cpu().numpy())
    return (
        np.concatenate(rul_predictions),
        np.concatenate(fault_predictions),
        np.concatenate(fault_probabilities),
    )


def fault_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2, 3]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class = {
        ID_TO_FAULT[label]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def condition_for_bearing(bearing_id: str) -> str:
    if bearing_id.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing_id.startswith("Bearing2_"):
        return "37.5Hz11kN"
    if bearing_id.startswith("Bearing3_"):
        return "40Hz10kN"
    raise KeyError(bearing_id)


def grouped_rul_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in predictions.groupby(group_column):
        metrics = regression_metrics(group["true_rul"].to_numpy(), group["pred_rul"].to_numpy())
        rows.append({group_column: group_value, "sequence_count": int(len(group)), **metrics})
    return pd.DataFrame(rows).sort_values(group_column)


def plot_losses(history: pd.DataFrame, plots_dir: Path) -> None:
    specs = [
        ("total_loss", "Total Loss", "m2_total_loss.png"),
        ("rul_loss", "RUL MSE Loss", "m2_rul_loss.png"),
        ("fault_loss", "Fault Cross-Entropy Loss", "m2_fault_loss.png"),
    ]
    for key, title, filename in specs:
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history[f"train_{key}"], label="Train")
        plt.plot(history["epoch"], history[f"val_{key}"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"M2 {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()


def plot_rul_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["true_rul"], predictions["pred_rul"], s=10, alpha=0.45)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("True normalized RUL")
    plt.ylabel("Predicted normalized RUL")
    plt.title("M2 Test RUL Predictions")
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


def plot_confusion(confusion: list[list[int]], output_path: Path) -> None:
    labels = [ID_TO_FAULT[index] for index in range(4)]
    matrix = np.asarray(confusion)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(4), labels, rotation=30)
    plt.yticks(range(4), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("M2 Fault Confusion Matrix")
    for row in range(4):
        for column in range(4):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def class_distribution(frame: pd.DataFrame, bundle: SequenceBundle, bearings: list[str]) -> dict[str, Any]:
    bearing_rows = frame[frame["bearing_id"].isin(bearings)].drop_duplicates("bearing_id")
    bearing_counts = bearing_rows.groupby("fault_label")["bearing_id"].nunique().to_dict()
    sequence_counts: dict[str, int] = {}
    for label_id, label in ID_TO_FAULT.items():
        sequence_counts[label] = int(np.sum(bundle.fault_label_id == label_id))
    return {
        "bearing_counts": {label: int(bearing_counts.get(label, 0)) for label in FAULT_TO_ID},
        "sequence_counts": sequence_counts,
    }


def write_report(
    output_dir: Path,
    config: M2Config,
    train_bundle: SequenceBundle,
    val_bundle: SequenceBundle,
    test_bundle: SequenceBundle,
    distribution: dict[str, Any],
    rul: dict[str, Any],
    fault: dict[str, Any],
    per_condition: pd.DataFrame,
    per_bearing: pd.DataFrame,
    history: pd.DataFrame,
    training_time: float,
    warnings: list[str],
) -> None:
    comparison = {
        "m1": M1_REFERENCE,
        "m2": {"mae": rul["mae"], "rmse": rul["rmse"], "r2": rul["r2"]},
        "delta_m2_minus_m1": {
            "mae": rul["mae"] - M1_REFERENCE["mae"],
            "rmse": rul["rmse"] - M1_REFERENCE["rmse"],
            "r2": rul["r2"] - M1_REFERENCE["r2"],
        },
    }
    improves = comparison["delta_m2_minus_m1"]["mae"] < 0 and comparison["delta_m2_minus_m1"]["rmse"] < 0
    report = {
        "experiment": "M2",
        "architecture": {
            "shared_encoder": "BiLSTM hidden_size=64 num_layers=1 bidirectional=True",
            "shared_representation_dim": 128,
            "rul_head": "Linear(128,64) -> ReLU -> Linear(64,1) -> Sigmoid",
            "fault_head": "Linear(128,64) -> ReLU -> Linear(64,4)",
            "loss": "MSELoss + CrossEntropyLoss",
        },
        "hyperparameters": asdict(config),
        "sequence_counts": {
            "train": int(len(train_bundle.y)),
            "validation": int(len(val_bundle.y)),
            "test": int(len(test_bundle.y)),
        },
        "fault_distribution": distribution,
        "rul_metrics": rul,
        "fault_metrics": fault,
        "per_condition_rul_metrics": per_condition.to_dict(orient="records"),
        "per_bearing_rul_metrics": per_bearing.to_dict(orient="records"),
        "training_time_seconds": training_time,
        "epochs_completed": int(history["epoch"].max()),
        "best_validation_total_loss": float(history["val_total_loss"].min()),
        "comparison_with_m1": comparison,
        "m2_improves_rul_vs_m1": improves,
        "warnings": warnings,
    }
    (output_dir / "m2_experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "m2_metrics.json").write_text(
        json.dumps({"rul": rul, "fault": fault, "comparison_with_m1": comparison}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# M2 Experiment Report",
        "",
        "M2 is a clean multi-task baseline with shared BiLSTM encoder, one RUL regression head, and one fault classification head. It does not use monotonicity loss, uncertainty weighting, physics loss, PCGrad, GradNorm, or domain adaptation.",
        "",
        "## Results",
        "",
        f"- Train/validation/test sequences: `{len(train_bundle.y)}` / `{len(val_bundle.y)}` / `{len(test_bundle.y)}`",
        f"- Training time seconds: `{training_time}`",
        f"- Test MAE: `{rul['mae']}`",
        f"- Test RMSE: `{rul['rmse']}`",
        f"- Test R2: `{rul['r2']}`",
        f"- Fault accuracy: `{fault['accuracy']}`",
        f"- Fault macro-F1: `{fault['macro_f1']}`",
        f"- M2 improves RUL vs M1: `{improves}`",
        "",
        "## Comparison With M1",
        "",
        f"- M1: `{M1_REFERENCE}`",
        f"- M2: `{comparison['m2']}`",
        f"- Delta M2 - M1: `{comparison['delta_m2_minus_m1']}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")
    (output_dir / "m2_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = M2Config()
    set_seed(config.seed)
    output_dir = RESEARCH_ROOT / "results" / "m2"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_metadata = json.loads(
        (RESEARCH_ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text(encoding="utf-8")
    )
    feature_columns = preprocessing_metadata["feature_columns"]
    frame = load_processed_metadata(RESEARCH_ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    selected_fold = next(fold for fold in validation_folds() if fold.validation_bearing == config.selected_validation_bearing)

    scaler = fit_scaler(frame, feature_columns, selected_fold.train_bearings)
    transformed = transform_features(frame, feature_columns, scaler)
    train_bundle = make_sequences(frame, transformed, selected_fold.train_bearings, config.sequence_length, config.stride)
    val_bundle = make_sequences(frame, transformed, [selected_fold.validation_bearing], config.sequence_length, config.stride)
    test_bundle = make_sequences(frame, transformed, TEST_BEARINGS, config.sequence_length, config.stride)

    if not np.isfinite(transformed).all():
        raise RuntimeError("Scaled features contain NaN or Inf")
    if set(selected_fold.train_bearings) & {selected_fold.validation_bearing} or set(selected_fold.train_bearings) & set(TEST_BEARINGS):
        raise RuntimeError("Bearing split overlap detected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings = list(preprocessing_metadata.get("warnings", []))
    if not torch.cuda.is_available():
        warnings.append("CUDA is not available to PyTorch; M2 trained on CPU.")
    torch_info = m1_train.torch_status()
    print(f"PyTorch: {torch_info['torch_version']}")
    print(f"CUDA available: {torch_info['cuda_available']}")
    print(f"Device: {device}")
    print(f"Train bearings: {selected_fold.train_bearings}")
    print(f"Validation bearing: {selected_fold.validation_bearing}")
    print(f"Test bearings: {TEST_BEARINGS}")
    print(f"Sequence counts train/val/test: {len(train_bundle.y)}/{len(val_bundle.y)}/{len(test_bundle.y)}")

    train_loader = dataloader(train_bundle, config, shuffle=True)
    val_loader = dataloader(val_bundle, config, shuffle=False)
    test_loader = dataloader(test_bundle, config, shuffle=False)
    model = BiLSTMMultiTaskRegressorClassifier(input_size=len(feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rul_loss_fn = nn.MSELoss()
    fault_loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_losses = run_epoch(model, train_loader, rul_loss_fn, fault_loss_fn, device, optimizer, config.grad_clip_norm)
        val_losses = run_epoch(model, val_loader, rul_loss_fn, fault_loss_fn, device, None, config.grad_clip_norm)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_losses.items()})
        row.update({f"val_{key}": value for key, value in val_losses.items()})
        history_rows.append(row)
        print(
            f"epoch {epoch:03d} train total={train_losses['total_loss']:.5f} "
            f"val total={val_losses['total_loss']:.5f} "
            f"val rul={val_losses['rul_loss']:.5f} val fault={val_losses['fault_loss']:.5f}",
            flush=True,
        )
        if val_losses["total_loss"] < best_val - 1e-8:
            best_val = val_losses["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    training_time = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No M2 checkpoint captured")

    checkpoint_path = output_dir / "m2_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "fold": {
                "train_bearings": selected_fold.train_bearings,
                "validation_bearing": selected_fold.validation_bearing,
                "test_bearings": TEST_BEARINGS,
            },
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "m2_training_history.csv", index=False)
    plot_losses(history, plots_dir)

    pred_rul, pred_fault, fault_probs = predict(model, test_loader, device)
    predictions = pd.DataFrame(
        {
            "bearing_id": test_bundle.bearing_id,
            "condition": [condition_for_bearing(item) for item in test_bundle.bearing_id],
            "start_position": test_bundle.start_position,
            "end_position": test_bundle.end_position,
            "recording_index_end": test_bundle.recording_index_end,
            "true_rul": test_bundle.y,
            "pred_rul": pred_rul,
            "true_fault_id": test_bundle.fault_label_id,
            "pred_fault_id": pred_fault,
            "true_fault_label": [ID_TO_FAULT[int(item)] for item in test_bundle.fault_label_id],
            "pred_fault_label": [ID_TO_FAULT[int(item)] for item in pred_fault],
        }
    )
    for class_id, label in ID_TO_FAULT.items():
        predictions[f"prob_{label}"] = fault_probs[:, class_id]
    predictions.to_csv(output_dir / "m2_test_predictions.csv", index=False)

    rul = regression_metrics(test_bundle.y, pred_rul)
    rul["mean_prediction_bias_pred_minus_true"] = float(np.mean(pred_rul - test_bundle.y))
    rul["test_sequence_count"] = int(len(test_bundle.y))
    fault = fault_metrics(test_bundle.fault_label_id, pred_fault)
    per_condition = grouped_rul_metrics(predictions, "condition")
    per_bearing = grouped_rul_metrics(predictions, "bearing_id")
    per_bearing["condition"] = per_bearing["bearing_id"].map(condition_for_bearing)
    per_condition.to_csv(output_dir / "m2_per_condition_rul_metrics.csv", index=False)
    per_bearing.to_csv(output_dir / "m2_per_bearing_rul_metrics.csv", index=False)
    pd.DataFrame(fault["confusion_matrix"]).to_csv(output_dir / "m2_confusion_matrix.csv", index=False)

    distribution = {
        "train": class_distribution(frame, train_bundle, selected_fold.train_bearings),
        "validation": class_distribution(frame, val_bundle, [selected_fold.validation_bearing]),
        "test": class_distribution(frame, test_bundle, TEST_BEARINGS),
    }
    (output_dir / "m2_config_used.json").write_text(
        json.dumps({"config": asdict(config), "torch": torch_info, "fold": selected_fold.__dict__}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "m2_fault_distribution.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")
    (output_dir / "m2_scaler_parameters.json").write_text(
        json.dumps(
            {
                "fit_bearings": selected_fold.train_bearings,
                "validation_bearing": selected_fold.validation_bearing,
                "test_bearings": TEST_BEARINGS,
                "feature_mean": dict(zip(feature_columns, map(float, scaler.mean_))),
                "feature_std": dict(zip(feature_columns, map(float, scaler.scale_))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_rul_scatter(predictions, plots_dir / "m2_pred_vs_true_rul_scatter.png")
    plot_chronological(predictions, plots_dir / "m2_chronological_test_curves.png")
    plot_confusion(fault["confusion_matrix"], plots_dir / "m2_confusion_matrix.png")

    write_report(
        output_dir,
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        distribution,
        rul,
        fault,
        per_condition,
        per_bearing,
        history,
        training_time,
        warnings,
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation total loss: {best_val}")
    print(f"RUL MAE/RMSE/R2: {rul['mae']}/{rul['rmse']}/{rul['r2']}")
    print(f"Fault accuracy/macro-F1: {fault['accuracy']}/{fault['macro_f1']}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

