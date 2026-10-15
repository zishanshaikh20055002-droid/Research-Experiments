"""Run fixed 3-seed robustness experiments for M1, M2, M3, M4, and M5."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SequenceBundle, fit_scaler, load_processed_metadata, make_sequences, transform_features  # noqa: E402
from src.labels import ID_TO_FAULT, TEST_BEARINGS, validation_folds  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.models import BiLSTMMultiTaskRegressorClassifier, BiLSTMRULRegressor  # noqa: E402


SEEDS = [42, 123, 2024]
MODELS = ["m1", "m2", "m3", "m5"]


@dataclass(frozen=True)
class Config:
    sequence_length: int = 10
    stride: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    lambda_mono: float = 0.1
    validation_bearing: str = "Bearing2_2"


class RULDataset(Dataset):
    def __init__(self, bundle: SequenceBundle):
        self.x = torch.as_tensor(bundle.x, dtype=torch.float32)
        self.y = torch.as_tensor(bundle.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class MultiDataset(Dataset):
    def __init__(self, bundle: SequenceBundle):
        self.x = torch.as_tensor(bundle.x, dtype=torch.float32)
        self.rul = torch.as_tensor(bundle.y, dtype=torch.float32).view(-1, 1)
        self.fault = torch.as_tensor(bundle.fault_label_id, dtype=torch.long)
        self.bearing = np.asarray(bundle.bearing_id)
        self.pos = torch.as_tensor(bundle.end_position, dtype=torch.long)

    def __len__(self):
        return len(self.rul)

    def __getitem__(self, index):
        return self.x[index], self.rul[index], self.fault[index], self.bearing[index], self.pos[index]


class M5Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_rul = nn.Parameter(torch.zeros(()))
        self.s_fault = nn.Parameter(torch.zeros(()))

    def forward(self, rul_loss, fault_loss):
        return torch.exp(-self.s_rul) * rul_loss + self.s_rul + torch.exp(-self.s_fault) * fault_loss + self.s_fault


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def condition_for_bearing(bearing: str) -> str:
    if bearing.startswith("Bearing1_"):
        return "35Hz12kN"
    if bearing.startswith("Bearing2_"):
        return "37.5Hz11kN"
    return "40Hz10kN"


def mono_loss(pred, bearings, positions):
    values = pred.view(-1)
    losses = []
    for bearing in sorted(set(str(x) for x in bearings)):
        idx = [i for i, x in enumerate(bearings) if str(x) == bearing]
        if len(idx) < 2:
            continue
        index = torch.as_tensor(idx, device=values.device)
        order = torch.argsort(positions.to(values.device)[index])
        ordered = values[index][order]
        losses.append(torch.relu(ordered[1:] - ordered[:-1]))
    return torch.cat(losses).mean() if losses else values.sum() * 0.0


def mono_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    mags = []
    pairs = 0
    for _, group in df.groupby("bearing_id"):
        vals = group.sort_values("end_position").pred_rul.to_numpy(float)
        diffs = vals[1:] - vals[:-1]
        bad = diffs[diffs > 0]
        pairs += len(diffs)
        mags.extend(bad.tolist())
    return {
        "mono_violation_rate": float(len(mags) / pairs) if pairs else 0.0,
        "mono_mean_violation_magnitude": float(np.mean(mags)) if mags else 0.0,
        "mono_violating_transitions": int(len(mags)),
        "mono_valid_adjacent_transitions": int(pairs),
    }


def fault_metrics(y_true, y_pred) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "fault_accuracy": float(accuracy_score(y_true, y_pred)),
        "fault_macro_precision": float(mp),
        "fault_macro_recall": float(mr),
        "fault_macro_f1": float(mf),
        "fault_per_class": {
            ID_TO_FAULT[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(4)
        },
    }


def per_condition(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for cond, group in df.groupby("condition"):
        rows.append({"condition": cond, **regression_metrics(group.true_rul.to_numpy(), group.pred_rul.to_numpy())})
    return rows


def make_data(frame, features, cfg: Config):
    fold = next(f for f in validation_folds() if f.validation_bearing == cfg.validation_bearing)
    scaler = fit_scaler(frame, features, fold.train_bearings)
    x = transform_features(frame, features, scaler)
    return (
        fold,
        make_sequences(frame, x, fold.train_bearings, cfg.sequence_length, cfg.stride),
        make_sequences(frame, x, [fold.validation_bearing], cfg.sequence_length, cfg.stride),
        make_sequences(frame, x, TEST_BEARINGS, cfg.sequence_length, cfg.stride),
        scaler,
    )


def run_m1(seed, frame, features, cfg, device, out):
    set_seed(seed)
    fold, train_b, val_b, test_b, scaler = make_data(frame, features, cfg)
    train_l = DataLoader(RULDataset(train_b), batch_size=cfg.batch_size, shuffle=True)
    val_l = DataLoader(RULDataset(val_b), batch_size=cfg.batch_size)
    model = BiLSTMRULRegressor(input_size=len(features)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best, stale, best_state, best_epoch, rows = float("inf"), 0, None, 0, []
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); tr = 0; n = 0
        for xb, yb in train_l:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); pred = model(xb); loss = loss_fn(pred, yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
            tr += float(loss.detach().cpu()) * len(yb); n += len(yb)
        model.eval(); va = 0; vn = 0
        with torch.no_grad():
            for xb, yb in val_l:
                pred = model(xb.to(device)); loss = loss_fn(pred.cpu(), yb)
                va += float(loss) * len(yb); vn += len(yb)
        tr /= n; va /= vn; rows.append({"epoch": epoch, "train_total_loss": tr, "val_total_loss": va, "train_rul_loss": tr, "val_rul_loss": va})
        if va < best - 1e-8:
            best, stale, best_state, best_epoch = va, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}, epoch
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return finish("m1", seed, out, model, None, test_b, cfg, features, scaler, rows, best_epoch, best, train_time, device)


def run_multi(model_name, seed, frame, features, cfg, device, out):
    set_seed(seed)
    fold, train_b, val_b, test_b, scaler = make_data(frame, features, cfg)
    train_l = DataLoader(MultiDataset(train_b), batch_size=cfg.batch_size, shuffle=(model_name in {"m2", "m4"}))
    val_l = DataLoader(MultiDataset(val_b), batch_size=cfg.batch_size, shuffle=False)
    model = BiLSTMMultiTaskRegressorClassifier(input_size=len(features)).to(device)
    weights = M5Weights().to(device) if model_name in {"m4", "m5"} else None
    params = list(model.parameters()) + (list(weights.parameters()) if weights else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    best, stale, best_state, best_w, best_epoch, rows = float("inf"), 0, None, None, 0, []
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        tr = epoch_multi(model, weights, train_l, mse, ce, opt, cfg, device, model_name)
        va = epoch_multi(model, weights, val_l, mse, ce, None, cfg, device, model_name)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        if weights:
            row.update({"s_rul": float(weights.s_rul.detach().cpu()), "s_fault": float(weights.s_fault.detach().cpu())})
        rows.append(row)
        if va["total_loss"] < best - 1e-8:
            best, stale, best_epoch = va["total_loss"], 0, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_w = {k: v.cpu().clone() for k, v in weights.state_dict().items()} if weights else None
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    if weights:
        weights.load_state_dict(best_w)
    return finish(model_name, seed, out, model, weights, test_b, cfg, features, scaler, rows, best_epoch, best, train_time, device)


def epoch_multi(model, weights, data, mse, ce, opt, cfg, device, name):
    is_train = opt is not None
    model.train(is_train)
    totals = {"rul_loss": 0.0, "fault_loss": 0.0, "mono_loss": 0.0, "total_loss": 0.0, "n": 0}
    for xb, yr, yf, bearings, pos in data:
        xb, yr, yf, pos = xb.to(device), yr.to(device), yf.to(device), pos.to(device)
        if is_train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pr, logits = model(xb)
            rl, fl = mse(pr, yr), ce(logits, yf)
            ml = mono_loss(pr, bearings, pos) if name in {"m3", "m5"} else pr.sum() * 0.0
            task = (torch.exp(-weights.s_rul) * rl + weights.s_rul + torch.exp(-weights.s_fault) * fl + weights.s_fault) if name in {"m4", "m5"} else rl + fl
            loss = task + (cfg.lambda_mono * ml if name in {"m3", "m5"} else 0.0)
            if is_train:
                loss.backward()
                params = list(model.parameters()) + (list(weights.parameters()) if weights else [])
                nn.utils.clip_grad_norm_(params, cfg.grad_clip); opt.step()
        n = len(yr)
        totals["rul_loss"] += float(rl.detach().cpu()) * n; totals["fault_loss"] += float(fl.detach().cpu()) * n
        totals["mono_loss"] += float(ml.detach().cpu()) * n; totals["total_loss"] += float(loss.detach().cpu()) * n; totals["n"] += n
    n = totals.pop("n")
    return {k: v / n for k, v in totals.items()}


def finish(model_name, seed, out, model, weights, test_b, cfg, features, scaler, rows, best_epoch, best, train_time, device):
    test_l = DataLoader(MultiDataset(test_b), batch_size=cfg.batch_size, shuffle=False)
    model.eval(); pr_all=[]; pf_all=[]; probs_all=[]
    with torch.no_grad():
        for xb, *_ in test_l:
            if model_name == "m1":
                pr = model(xb.to(device)); logits = None
                pf = np.full(len(xb), -1); probs = np.zeros((len(xb), 4))
            else:
                pr, logits = model(xb.to(device)); probs_t = torch.softmax(logits, 1)
                pf = torch.argmax(probs_t, 1).cpu().numpy(); probs = probs_t.cpu().numpy()
            pr_all.append(pr.cpu().numpy().reshape(-1)); pf_all.append(pf); probs_all.append(probs)
    pred = np.concatenate(pr_all); pf = np.concatenate(pf_all); probs = np.concatenate(probs_all)
    df = pd.DataFrame({
        "bearing_id": test_b.bearing_id, "condition": [condition_for_bearing(x) for x in test_b.bearing_id],
        "start_position": test_b.start_position, "end_position": test_b.end_position,
        "true_rul": test_b.y, "pred_rul": pred, "true_fault_id": test_b.fault_label_id, "pred_fault_id": pf,
    })
    for i in range(4):
        df[f"prob_{ID_TO_FAULT[i]}"] = probs[:, i]
    rul = regression_metrics(test_b.y, pred)
    rul["mean_prediction_bias_pred_minus_true"] = float(np.mean(pred - test_b.y)); rul["prediction_min"] = rul["prediction_min"]; rul["prediction_max"] = rul["prediction_max"]
    mono = mono_metrics(df)
    fault = fault_metrics(test_b.fault_label_id, pf) if model_name != "m1" else {"fault_accuracy": None, "fault_macro_precision": None, "fault_macro_recall": None, "fault_macro_f1": None}
    metrics = {
        "model": model_name.upper(), "seed": seed, "best_epoch": best_epoch, "best_validation_loss": best, "training_time_seconds": train_time,
        **rul, **fault, **mono, "per_condition": per_condition(df),
    }
    if weights:
        metrics["s_R"] = float(weights.s_rul.detach().cpu()); metrics["s_F"] = float(weights.s_fault.detach().cpu())
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "training_history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "weights_state_dict": weights.state_dict() if weights else None, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "features": features, "config": cfg.__dict__}, out / "checkpoint.pt")
    return metrics


def main():
    cfg = Config()
    meta = json.loads((ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text())
    frame = load_processed_metadata(ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    features = meta["feature_columns"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / "results" / "multiseed"
    print(f"Device: {device}; torch={torch.__version__}", flush=True)
    all_metrics = []
    for model in MODELS:
        for seed in SEEDS:
            out = base / model / f"seed_{seed}"
            print(f"Running {model.upper()} seed {seed}", flush=True)
            if model == "m1":
                metrics = run_m1(seed, frame, features, cfg, device, out)
            else:
                metrics = run_multi(model, seed, frame, features, cfg, device, out)
            all_metrics.append(metrics)
            print(f"{model.upper()} seed {seed}: MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f} F1={metrics.get('fault_macro_f1')}", flush=True)
    (base / "run_manifest.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("Multiseed runs complete. Run aggregate_multiseed.py next.", flush=True)


if __name__ == "__main__":
    main()
