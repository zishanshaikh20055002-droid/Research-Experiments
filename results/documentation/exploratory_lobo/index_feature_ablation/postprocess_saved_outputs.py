"""Add deterministic summaries to completed post-hoc ablation outputs; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT = RUN_DIR.parents[1]

metrics_path = RUN_DIR / "metrics_by_model.csv"
metrics = pd.read_csv(metrics_path)
metrics = metrics[metrics["seed"].astype(str) != "mean_across_fixed_seeds"].copy()
value_columns = ["mae", "rmse", "r2", "prediction_min", "prediction_max", "prediction_mean", "prediction_std", "macro_bearing_mae", "macro_bearing_rmse", "mean_fit_predict_seconds_per_fold"]
group_columns = ["model", "aggregation", "operating_condition"]
summaries = metrics.groupby(group_columns, as_index=False)[value_columns].mean()
summaries.insert(1, "seed", "mean_across_fixed_seeds")
pd.concat([metrics, summaries], ignore_index=True).to_csv(metrics_path, index=False)

diagnostic_path = RUN_DIR / "fold_extrapolation_diagnostics.csv"
diagnostic = pd.read_csv(diagnostic_path)
metadata = pd.read_csv(PROJECT / "data" / "processed" / "xjtu_features_metadata.csv.gz")
source_indices = metadata.groupby("bearing_id", sort=True)["recording_index"].apply(lambda values: ";".join(str(int(v)) for v in values)).rename("source_recording_indices")
diagnostic = diagnostic.drop(columns=["source_recording_indices"], errors="ignore").merge(source_indices, left_on="heldout_bearing", right_index=True, how="left")
diagnostic.to_csv(diagnostic_path, index=False)
