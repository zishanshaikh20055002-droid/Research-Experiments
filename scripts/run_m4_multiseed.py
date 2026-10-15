"""Run only the M4 3-seed final-pass experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_multiseed import Config, SEEDS, run_multi  # noqa: E402
from src.dataset import load_processed_metadata  # noqa: E402


def main() -> int:
    cfg = Config()
    meta = json.loads((ROOT / "data" / "processed" / "preprocessing_metadata.json").read_text())
    frame = load_processed_metadata(ROOT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
    features = meta["feature_columns"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / "results" / "multiseed" / "m4"
    print(f"Device: {device}; torch={torch.__version__}", flush=True)
    all_metrics = []
    for seed in SEEDS:
        out = base / f"seed_{seed}"
        print(f"Running M4 seed {seed}", flush=True)
        metrics = run_multi("m4", seed, frame, features, cfg, device, out)
        all_metrics.append(metrics)
        print(
            "M4 seed {seed}: MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f} "
            "F1={f1} s_R={sr:.6f} s_F={sf:.6f}".format(
                seed=seed,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                f1=metrics.get("fault_macro_f1"),
                sr=metrics["s_R"],
                sf=metrics["s_F"],
            ),
            flush=True,
        )
    (base / "m4_run_manifest.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print("M4 multiseed run complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
