"""Fast-track M3 and M5 experiments for the XJTU-SY ablation study."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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


M1 = {"mae": 0.306988, "rmse": 0.401279, "r2": -1.008124}
M2 = {"mae": 0.3137107426063325, "rmse": 0.3932937344852861, "r2": -0.9289969468171038, "macro_f1": 0.11245001671283134}
M1_CONDITION = {
    "35Hz12kN": {"mae": 0.10282459158572506, "rmse": 0.14353681241534164, "r2": 0.7151602995141819},
    "37.5Hz11kN": {"mae": 0.1562410123925026, "rmse": 0.20868325015340167, "r2": 0.45344778024002597},
    "40Hz10kN": {"mae": 0.4063712759267536, "rmse": 0.48536968571922934, "r2": -1.8914436548435964},
}
M2_CONDITION = {
    "35Hz12kN": {"mae": 0.1265777557996697, "rmse": 0.17589945130937182, "r2": 0.5722373170194478},
    "37.5Hz11kN": {"mae": 0.18364381914451222, "rmse": 0.2370138531616812, "r2": 0.2949759373831987},
    "40Hz10kN": {"mae": 0.4011574576604232, "rmse": 0.46703869578890206, "r2": -1.6771651713075255},
}


@dataclass(frozen=True)
class FastConfig:
    seed: int = 42
    sequence_length: int = 10
    stride: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 40
    patience: int = 8
    grad_clip_norm: float = 1.0
    lambda_mono: float = 0.1
    selected_validation_bearing: str = "Bearing2_2"


class MultiTaskDataset(Dataset):
    def __init__(self, bundle: SequenceBundle):
        self.x = torch.as_tensor(bundle.x, dtype=torch.float32)
        self.rul = torch.as_tensor(bundle.y, dtype=torch.float32).view(-1, 1)
        self.fault = torch.as_tensor(bundle.fault_label_id, dtype=torch.long)
        self.bearing = np.asarray(bundle.bearing_id)
        self.end_position = torch.as_tensor(bundle.end_position, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.rul[index], self.fault[index], self.bearing[index], self.end_position[index]


class M5LossWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s_rul = nn.Parameter(torch.zeros(()))
        self.s_fault = nn.Parameter(torch.zeros(()))

    def forward(self, rul_loss: torch.Tensor, fault_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.s_rul) * rul_loss + self.s_rul + torch.exp(-self.s_fault) * fault_loss + self.s_fault


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def condition_for_bearing(bearing_id: str) -> str:
    if bearing_id.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing_id.startswith("Bearing2_"):
        return "37.5Hz11kN"
    if bearing_id.startswith("Bearing3_"):
        return "40Hz10kN"
    raise KeyError(bearing_id)


def loader(bundle: SequenceBundle, config: FastConfig, shuffle: bool) -> DataLoader:
    return DataLoader(MultiTaskDataset(bundle), batch_size=config.batch_size, shuffle=shuffle, num_workers=0)


def monotonicity_loss(pred_rul: torch.Tensor, bearings: tuple[str, ...] | list[str], positions: torch.Tensor) -> torch.Tensor:
    values = pred_rul.view(-1)
    penalties: list[torch.Tensor] = []
    unique_bearings = sorted(set(str(item) for item in bearings))
    for bearing in unique_bearings:
        indices = [index for index, item in enumerate(bearings) if str(item) == bearing]
        if len(indices) < 2:
            continue
        index_tensor = torch.as_tensor(indices, device=values.device, dtype=torch.long)
        order = torch.argsort(positions.to(values.device)[index_tensor])
        ordered_values = values[index_tensor][order]
        penalties.append(torch.relu(ordered_values[1:] - ordered_values[:-1]))
    if not penalties:
        return values.sum() * 0.0
    return torch.cat(penalties).mean()


def run_epoch(model, weights, data_loader, device, config, optimizer, mode: str):
    is_train = optimizer is not None
    model.train(is_train)
    if weights is not None:
        weights.train(is_train)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    totals = {"rul_loss": 0.0, "fault_loss": 0.0, "mono_loss": 0.0, "total_loss": 0.0, "count": 0.0}
    for x, rul, fault, bearings, positions in data_loader:
        x, rul, fault, positions = x.to(device), rul.to(device), fault.to(device), positions.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred_rul, logits = model(x)
            rul_loss = mse(pred_rul, rul)
            fault_loss = ce(logits, fault)
            mono_loss = monotonicity_loss(pred_rul, bearings, positions)
            if mode == "m5":
                task_loss = weights(rul_loss, fault_loss)
            else:
                task_loss = rul_loss + fault_loss
            total_loss = task_loss + config.lambda_mono * mono_loss
            if is_train:
                total_loss.backward()
                params = list(model.parameters()) + (list(weights.parameters()) if weights is not None else [])
                nn.utils.clip_grad_norm_(params, config.grad_clip_norm)
                optimizer.step()
        n = int(x.shape[0])
        totals["rul_loss"] += float(rul_loss.detach().cpu()) * n
        totals["fault_loss"] += float(fault_loss.detach().cpu()) * n
        totals["mono_loss"] += float(mono_loss.detach().cpu()) * n
        totals["total_loss"] += float(total_loss.detach().cpu()) * n
        totals["count"] += n
    count = max(totals.pop("count"), 1.0)
    return {k: v / count for k, v in totals.items()}


def predict(model, data_loader, device):
    model.eval()
    pr, pf, probs = [], [], []
    with torch.no_grad():
        for x, *_ in data_loader:
            rul, logits = model(x.to(device))
            p = torch.softmax(logits, dim=1)
            pr.append(rul.cpu().numpy().reshape(-1))
            pf.append(torch.argmax(p, dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.concatenate(pr), np.concatenate(pf), np.concatenate(probs)


def fault_metrics(y_true, y_pred):
    labels = [0, 1, 2, 3]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {ID_TO_FAULT[label]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mono_metrics(predictions: pd.DataFrame):
    total = violations = 0
    magnitudes = []
    for _, group in predictions.groupby("bearing_id"):
        values = group.sort_values("end_position")["pred_rul"].to_numpy(float)
        diffs = values[1:] - values[:-1]
        bad = diffs[diffs > 0]
        total += len(diffs)
        violations += len(bad)
        magnitudes.extend(bad.tolist())
    return {"violation_rate": float(violations / total) if total else 0.0, "mean_violation_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0, "violation_count": int(violations), "adjacent_pairs": int(total)}


def grouped_rul(predictions: pd.DataFrame, group_col: str):
    rows = []
    for value, group in predictions.groupby(group_col):
        rows.append({group_col: value, "sequence_count": int(len(group)), **regression_metrics(group.true_rul.to_numpy(), group.pred_rul.to_numpy())})
    return pd.DataFrame(rows).sort_values(group_col)


def plot_all(history, predictions, fault, output_dir: Path, model_name: str, s_history=None):
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    for key, title in [("total_loss", "Total Loss"), ("rul_loss", "RUL Loss"), ("fault_loss", "Fault Loss")]:
        plt.figure(figsize=(8, 5))
        plt.plot(history.epoch, history[f"train_{key}"], label="Train")
        plt.plot(history.epoch, history[f"val_{key}"], label="Validation")
        plt.title(f"{model_name.upper()} {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / f"{model_name}_{key}.png", dpi=160)
        plt.close()
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions.true_rul, predictions.pred_rul, s=10, alpha=0.45)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.tight_layout()
    plt.savefig(plots / f"{model_name}_pred_vs_true_rul.png", dpi=160)
    plt.close()
    bearings = list(predictions.bearing_id.drop_duplicates())[:4]
    fig, axes = plt.subplots(len(bearings), 1, figsize=(9, max(3, 2.4 * len(bearings))))
    if len(bearings) == 1:
        axes = [axes]
    for ax, bearing in zip(axes, bearings):
        group = predictions[predictions.bearing_id == bearing].sort_values("end_position")
        ax.plot(group.end_position, group.true_rul, label="True")
        ax.plot(group.end_position, group.pred_rul, label="Pred")
        ax.set_title(bearing)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots / f"{model_name}_chronological_rul.png", dpi=160)
    plt.close(fig)
    matrix = np.asarray(fault["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    labels = [ID_TO_FAULT[i] for i in range(4)]
    plt.xticks(range(4), labels, rotation=30)
    plt.yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(plots / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close()
    if s_history is not None and not s_history.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(s_history.epoch, s_history.s_rul, label="s_R")
        plt.plot(s_history.epoch, s_history.s_fault, label="s_F")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Learned scalar")
        plt.tight_layout()
        plt.savefig(plots / f"{model_name}_uncertainty_parameters.png", dpi=160)
        plt.close()


def class_distribution(frame, bundle, bearings):
    rows = frame[frame.bearing_id.isin(bearings)].drop_duplicates("bearing_id")
    bc = rows.groupby("fault_label").bearing_id.nunique().to_dict()
    sc = {ID_TO_FAULT[i]: int(np.sum(bundle.fault_label_id == i)) for i in range(4)}
    return {"bearing_counts": {label: int(bc.get(label, 0)) for label in FAULT_TO_ID}, "sequence_counts": sc}


def run_experiment(mode: str, frame, feature_columns, preprocessing_metadata, config: FastConfig, device):
    output_dir = RESEARCH_ROOT / "results" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    fold = next(f for f in validation_folds() if f.validation_bearing == config.selected_validation_bearing)
    scaler = fit_scaler(frame, feature_columns, fold.train_bearings)
    transformed = transform_features(frame, feature_columns, scaler)
    train_bundle = make_sequences(frame, transformed, fold.train_bearings, config.sequence_length, config.stride)
    val_bundle = make_sequences(frame, transformed, [fold.validation_bearing], config.sequence_length, config.stride)
    test_bundle = make_sequences(frame, transformed, TEST_BEARINGS, config.sequence_length, config.stride)
    train_loader = loader(train_bundle, config, shuffle=False)
    val_loader = loader(val_bundle, config, shuffle=False)
    test_loader = loader(test_bundle, config, shuffle=False)
    model = BiLSTMMultiTaskRegressorClassifier(input_size=len(feature_columns)).to(device)
    weights = M5LossWeights().to(device) if mode == "m5" else None
    params = list(model.parameters()) + (list(weights.parameters()) if weights is not None else [])
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    best_val = float("inf")
    best_state = None
    best_weights = None
    best_epoch = 0
    stale = 0
    hist = []
    s_hist = []
    start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        tr = run_epoch(model, weights, train_loader, device, config, optimizer, mode)
        va = run_epoch(model, weights, val_loader, device, config, None, mode)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        hist.append(row)
        if weights is not None:
            s_hist.append({"epoch": epoch, "s_rul": float(weights.s_rul.detach().cpu()), "s_fault": float(weights.s_fault.detach().cpu())})
        print(f"{mode.upper()} epoch {epoch:03d} train={tr['total_loss']:.5f} val={va['total_loss']:.5f} val_mono={va['mono_loss']:.5f}", flush=True)
        if va["total_loss"] < best_val - 1e-8:
            best_val = va["total_loss"]
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_weights = {k: v.detach().cpu().clone() for k, v in weights.state_dict().items()} if weights is not None else None
        else:
            stale += 1
        if stale >= config.patience:
            break
    training_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    if weights is not None:
        weights.load_state_dict(best_weights)
    history = pd.DataFrame(hist)
    s_history = pd.DataFrame(s_hist)
    history.to_csv(output_dir / f"{mode}_training_history.csv", index=False)
    if not s_history.empty:
        s_history.to_csv(output_dir / f"{mode}_uncertainty_history.csv", index=False)
    pred_rul, pred_fault, probs = predict(model, test_loader, device)
    predictions = pd.DataFrame({
        "bearing_id": test_bundle.bearing_id,
        "condition": [condition_for_bearing(x) for x in test_bundle.bearing_id],
        "start_position": test_bundle.start_position,
        "end_position": test_bundle.end_position,
        "true_rul": test_bundle.y,
        "pred_rul": pred_rul,
        "true_fault_id": test_bundle.fault_label_id,
        "pred_fault_id": pred_fault,
        "true_fault_label": [ID_TO_FAULT[int(x)] for x in test_bundle.fault_label_id],
        "pred_fault_label": [ID_TO_FAULT[int(x)] for x in pred_fault],
    })
    for class_id, label in ID_TO_FAULT.items():
        predictions[f"prob_{label}"] = probs[:, class_id]
    predictions.to_csv(output_dir / f"{mode}_test_predictions.csv", index=False)
    rul = regression_metrics(test_bundle.y, pred_rul)
    rul["mean_prediction_bias_pred_minus_true"] = float(np.mean(pred_rul - test_bundle.y))
    rul["mean_true_rul"] = float(np.mean(test_bundle.y))
    rul["mean_predicted_rul"] = float(np.mean(pred_rul))
    fault = fault_metrics(test_bundle.fault_label_id, pred_fault)
    mono = mono_metrics(predictions)
    per_condition = grouped_rul(predictions, "condition")
    per_condition.to_csv(output_dir / f"{mode}_per_condition_rul_metrics.csv", index=False)
    pd.DataFrame(fault["confusion_matrix"]).to_csv(output_dir / f"{mode}_confusion_matrix.csv", index=False)
    distribution = {
        "train": class_distribution(frame, train_bundle, fold.train_bearings),
        "validation": class_distribution(frame, val_bundle, [fold.validation_bearing]),
        "test": class_distribution(frame, test_bundle, TEST_BEARINGS),
    }
    metrics = {"rul": rul, "fault": fault, "monotonicity": mono, "per_condition_rul": per_condition.to_dict("records"), "fault_distribution": distribution}
    if weights is not None:
        metrics["learned_uncertainty"] = {"s_R": float(weights.s_rul.detach().cpu()), "s_F": float(weights.s_fault.detach().cpu())}
    (output_dir / f"{mode}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({
        "model_state_dict": best_state,
        "weights_state_dict": best_weights,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, output_dir / f"{mode}_best.pt")
    plot_all(history, predictions, fault, output_dir, mode, s_history if mode == "m5" else None)
    report = {
        "model": mode.upper(),
        "loss": "MSE + CE + lambda_mono*L_mono" if mode == "m3" else "exp(-s_R)*MSE+s_R + exp(-s_F)*CE+s_F + lambda_mono*L_mono",
        "lambda_mono": config.lambda_mono,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_val,
        "training_time_seconds": training_time,
        "sequence_counts": {"train": len(train_bundle.y), "validation": len(val_bundle.y), "test": len(test_bundle.y)},
        **metrics,
        "warnings": preprocessing_metadata.get("warnings", []) + ["CUDA unavailable; trained on CPU." if not torch.cuda.is_available() else ""],
    }
    (output_dir / f"{mode}_experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        f"# {mode.upper()} Experiment Report",
        "",
        f"- Loss: `{report['loss']}`",
        f"- Lambda mono: `{config.lambda_mono}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Training time seconds: `{training_time}`",
        f"- RUL MAE/RMSE/R2: `{rul['mae']}` / `{rul['rmse']}` / `{rul['r2']}`",
        f"- Fault macro-F1: `{fault['macro_f1']}`",
        f"- Mono violation rate: `{mono['violation_rate']}`",
        f"- Mean violation magnitude: `{mono['mean_violation_magnitude']}`",
        "",
        "Known limitation: the adopted split has no Cage-class bearing in the training subset, so Cage classification is not learnable in the conventional supervised sense.",
    ]
    (output_dir / f"{mode}_experiment_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{mode.upper()} done: MAE={rul['mae']:.6f} RMSE={rul['rmse']:.6f} R2={rul['r2']:.6f} macroF1={fault['macro_f1']:.6f} mono={mono['violation_rate']:.6f}", flush=True)
    return report


def write_comparison(m3, m5):
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.columns:
            display[column] = display[column].map(
                lambda value: f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            )
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in headers) + " |")
        return "\n".join(lines)

    out = RESEARCH_ROOT / "results" / "fasttrack_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"Model": "M1", "MAE": M1["mae"], "RMSE": M1["rmse"], "R2": M1["r2"], "Fault Macro-F1": None, "Mono Violation": None},
        {"Model": "M2", "MAE": M2["mae"], "RMSE": M2["rmse"], "R2": M2["r2"], "Fault Macro-F1": M2["macro_f1"], "Mono Violation": None},
        {"Model": "M3", "MAE": m3["rul"]["mae"], "RMSE": m3["rul"]["rmse"], "R2": m3["rul"]["r2"], "Fault Macro-F1": m3["fault"]["macro_f1"], "Mono Violation": m3["monotonicity"]["violation_rate"]},
        {"Model": "M5", "MAE": m5["rul"]["mae"], "RMSE": m5["rul"]["rmse"], "R2": m5["rul"]["r2"], "Fault Macro-F1": m5["fault"]["macro_f1"], "Mono Violation": m5["monotonicity"]["violation_rate"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "m1_m2_m3_m5_comparison.csv", index=False)
    cond_rows = []
    for model, vals in [("M1", M1_CONDITION), ("M2", M2_CONDITION)]:
        for cond, met in vals.items():
            cond_rows.append({"Model": model, "condition": cond, **met})
    for model, report in [("M3", m3), ("M5", m5)]:
        for row in report["per_condition_rul"]:
            cond_rows.append({"Model": model, "condition": row["condition"], "mae": row["mae"], "rmse": row["rmse"], "r2": row["r2"]})
    cdf = pd.DataFrame(cond_rows)
    cdf.to_csv(out / "per_condition_comparison.csv", index=False)
    best_rul = df.sort_values("RMSE").iloc[0]["Model"]
    best_mono = df.dropna(subset=["Mono Violation"]).sort_values("Mono Violation").iloc[0]["Model"]
    best_fault = df.dropna(subset=["Fault Macro-F1"]).sort_values("Fault Macro-F1", ascending=False).iloc[0]["Model"]
    md = ["# Fast-Track Comparison", "", markdown_table(df), "", "## Per-Condition RUL", "", markdown_table(cdf), "", f"- Best numeric RUL by RMSE: `{best_rul}`", f"- Best monotonicity: `{best_mono}`", f"- Best fault Macro-F1: `{best_fault}`", "", "Known limitation: the selected split has no Cage-class training bearing; Cage classification is not learnable supervised under this protocol."]
    (out / "m1_m2_m3_m5_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "m1_m2_m3_m5_comparison.json").write_text(json.dumps({"comparison": rows, "per_condition": cond_rows, "best_rul_rmse": best_rul, "best_monotonicity": best_mono, "best_fault_macro_f1": best_fault}, indent=2), encoding="utf-8")


def main():
    config = FastConfig()
    set_seed(config.seed)
    preprocessing_metadata = json.loads((RESEARCH_ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text())
    frame = load_processed_metadata(RESEARCH_ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    feature_columns = preprocessing_metadata["feature_columns"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; torch={torch.__version__}; lambda_mono={config.lambda_mono}", flush=True)
    print("Running M3", flush=True)
    m3 = run_experiment("m3", frame, feature_columns, preprocessing_metadata, config, device)
    print("Running M5", flush=True)
    set_seed(config.seed)
    m5 = run_experiment("m5", frame, feature_columns, preprocessing_metadata, config, device)
    write_comparison(m3, m5)
    print("Fast-track M3/M5 complete. Stopping.", flush=True)


if __name__ == "__main__":
    main()
