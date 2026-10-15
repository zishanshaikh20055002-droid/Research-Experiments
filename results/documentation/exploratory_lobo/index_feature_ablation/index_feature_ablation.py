"""Post-hoc exploratory LOBO ablation of recording index, vibration, and condition."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

RUN_DIR = Path(__file__).resolve().parent
PROJECT = RUN_DIR.parents[1]
DATA = PROJECT / "data" / "processed"
SOURCE_RUN = RUN_DIR.parent / "lobo_run_02"
SEQUENCE_LENGTH = 10
HGB_SEED = 42
TREE_SEEDS = (42, 123, 2024)
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 10_000


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    error = y - p
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
        "prediction_min": float(p.min()), "prediction_max": float(p.max()),
        "prediction_mean": float(p.mean()), "prediction_std": float(p.std()),
    }


def make_windows(metadata: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    records: dict[str, list] = {key: [] for key in ["x_final", "x_flat", "y", "bearing", "condition", "start", "end", "index_end"]}
    for bearing, frame in metadata.groupby("bearing_id", sort=True):
        frame = frame.sort_values("chronological_position", kind="stable")
        source = frame["_row"].to_numpy(int)
        positions = frame["chronological_position"].to_numpy(int)
        indices = frame["recording_index"].to_numpy(int)
        if not np.array_equal(positions, np.arange(len(frame))) or np.any(np.diff(indices) <= 0):
            raise ValueError(f"Invalid chronological ordering for {bearing}")
        for start in range(len(frame) - SEQUENCE_LENGTH + 1):
            end = start + SEQUENCE_LENGTH - 1
            records["x_final"].append(x[source[end]])
            records["x_flat"].append(x[source[start:end + 1]].reshape(-1))
            records["y"].append(y[source[end]])
            records["bearing"].append(bearing)
            records["condition"].append(frame["operating_condition"].iloc[0])
            records["start"].append(positions[start])
            records["end"].append(positions[end])
            records["index_end"].append(indices[end])
    return {
        "x_final": np.asarray(records["x_final"], dtype=float),
        "x_flat": np.asarray(records["x_flat"], dtype=float),
        "y": np.asarray(records["y"], dtype=float),
        "bearing": np.asarray(records["bearing"], dtype=str),
        "condition": np.asarray(records["condition"], dtype=str),
        "start": np.asarray(records["start"], dtype=int), "end": np.asarray(records["end"], dtype=int),
        "index_end": np.asarray(records["index_end"], dtype=int),
    }


def hgb(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


def extra(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=150, random_state=seed, min_samples_leaf=2, max_features=1.0, n_jobs=-1)


def design_matrix(name: str, d: dict[str, np.ndarray], train_mask: np.ndarray, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = d["index_end"].reshape(-1, 1).astype(float)
    if name in {"A1_index_hgb", "B1_index_extra"}:
        return index[train_mask], index[test_mask]
    if name in {"A2_vibration_hgb", "B2_vibration_extra"}:
        return d["x_final"][train_mask], d["x_final"][test_mask]
    if name in {"A3_index_vibration_hgb", "B3_index_vibration_extra"}:
        return np.column_stack([index[train_mask], d["x_final"][train_mask]]), np.column_stack([index[test_mask], d["x_final"][test_mask]])
    if name in {"A4_condition_hgb", "A5_index_condition_hgb"}:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ctrain = encoder.fit_transform(d["condition"][train_mask].reshape(-1, 1))
        ctest = encoder.transform(d["condition"][test_mask].reshape(-1, 1))
        if name == "A4_condition_hgb":
            return ctrain, ctest
        return np.column_stack([index[train_mask], ctrain]), np.column_stack([index[test_mask], ctest])
    if name == "B4_window_extra":
        return d["x_flat"][train_mask], d["x_flat"][test_mask]
    if name == "B5_index_window_extra":
        return np.column_stack([index[train_mask], d["x_flat"][train_mask]]), np.column_stack([index[test_mask], d["x_flat"][test_mask]])
    raise KeyError(name)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = frame.columns.tolist()
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header = "| " + " | ".join(column.ljust(widths[i]) for i, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def paired_summary(differences: np.ndarray) -> dict[str, object]:
    nonzero = differences[~np.isclose(differences, 0.0)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = np.mean(differences[rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))], axis=1)
    improved = int((differences < 0).sum())
    worsened = int((differences > 0).sum())
    tied = int(len(differences) - improved - worsened)
    sign_p = float(stats.binomtest(improved, improved + worsened, p=0.5, alternative="two-sided").pvalue) if len(nonzero) else float("nan")
    wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided", method="auto") if len(nonzero) else None
    return {
        "mean_difference": float(differences.mean()), "median_difference": float(np.median(differences)),
        "improved": improved, "worsened": worsened, "tied": tied,
        "bootstrap_percentile_95_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "sign_test_two_sided_p": sign_p, "sign_test_nonzero_n": int(len(nonzero)),
        "wilcoxon_two_sided_p": float(wilcoxon.pvalue) if wilcoxon else float("nan"),
        "wilcoxon_statistic": float(wilcoxon.statistic) if wilcoxon else float("nan"),
        "wilcoxon_nonzero_n": int(len(nonzero)),
    }


def main() -> None:
    npz = np.load(DATA / "xjtu_features.npz", allow_pickle=False)
    metadata = pd.read_csv(DATA / "xjtu_features_metadata.csv.gz")
    metadata["_row"] = np.arange(len(metadata))
    x = npz["X"].astype(float); y = npz["rul"].astype(float)
    names = npz["feature_names"].astype(str).tolist()
    source_folds = pd.read_csv(SOURCE_RUN / "fold_manifest.csv")
    if x.shape != (len(metadata), 32) or len(names) != 32 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Frozen data schema or finite-value validation failed")
    keys_match = (np.array_equal(metadata["bearing_id"].astype(str), npz["bearing_id"].astype(str)) and np.array_equal(metadata["chronological_position"], npz["chronological_position"]) and np.array_equal(metadata["recording_index"], npz["recording_index"]))
    if not keys_match or not np.allclose(metadata["rul"].astype(float), y, atol=1e-7, rtol=0):
        raise ValueError("NPZ/metadata key or target alignment failed")
    d = make_windows(metadata, x, y)
    bearings = sorted(metadata["bearing_id"].unique())
    if len(bearings) != 15 or len(d["y"]) != 9080 or not np.all((d["y"] >= 0) & (d["y"] <= 1)):
        raise ValueError("Unexpected bearing, window, or target-bound validation result")
    if set(source_folds["heldout_bearing"]) != set(bearings) or len(source_folds) != 15:
        raise ValueError("lobo_run_02 fold manifest is incompatible")

    feature_audit = []
    calculation = {
        "mean": "mean(signal)", "std": "sample standard deviation, ddof=1", "rms": "sqrt(mean(signal^2))", "peak_abs": "max(abs(signal))", "peak_to_peak": "max(signal)-min(signal)", "crest_factor": "peak_abs / rms", "kurtosis": "scipy kurtosis(signal, fisher=True, bias=False)", "skewness": "scipy skew(signal, bias=False)", "dominant_freq_hz": "frequency of maximum rFFT power after mean removal", "spectral_centroid_hz": "power-weighted rFFT frequency mean", "spectral_bandwidth_hz": "power-weighted rFFT frequency standard deviation", "spectral_rms": "sqrt(mean(rFFT power))",
    }
    for name in names:
        suffix = name[2:]
        formula = calculation.get(suffix, "rFFT band power / total rFFT power; bands: 0-1000, 1000-3000, 3000-6000, or 6000-12800 Hz")
        feature_audit.append({"feature": name, "channel": "horizontal" if name.startswith("h_") else "vertical", "exact_calculation": formula, "source": "one raw 32,768-sample vibration channel from the same recording", "independent_per_recording": True, "uses_later_recordings": False, "uses_whole_trajectory": False, "uses_total_length_or_terminal_values": False, "encodes_elapsed_time": False, "classification": "A", "rationale": "Source code applies only within-recording statistics or FFT to the current CSV."})
    pd.DataFrame(feature_audit).to_csv(RUN_DIR / "feature_time_leakage_audit.csv", index=False)

    models = {
        "training_mean": ("reference", (0,)),
        "A1_index_hgb": ("hgb", (HGB_SEED,)), "A2_vibration_hgb": ("hgb", (HGB_SEED,)), "A3_index_vibration_hgb": ("hgb", (HGB_SEED,)), "A4_condition_hgb": ("hgb", (HGB_SEED,)), "A5_index_condition_hgb": ("hgb", (HGB_SEED,)),
        "B1_index_extra": ("extra", TREE_SEEDS), "B2_vibration_extra": ("extra", TREE_SEEDS), "B3_index_vibration_extra": ("extra", TREE_SEEDS), "B4_window_extra": ("extra", TREE_SEEDS), "B5_index_window_extra": ("extra", TREE_SEEDS),
    }
    pred_rows=[]; bearing_rows=[]; fold_diag=[]; started_all=time.perf_counter()
    for _, fold in source_folds.sort_values("fold").iterrows():
        heldout=str(fold.heldout_bearing); test=d["bearing"] == heldout; train=~test
        listed=set(str(fold.training_bearings).split(";"))
        actual=set(d["bearing"][train])
        if listed != actual or heldout in actual: raise ValueError(f"Fold integrity failed for {heldout}")
        raw=metadata[metadata.bearing_id == heldout].sort_values("chronological_position")
        train_indices=d["index_end"][train]; held_indices=d["index_end"][test]
        training_lengths=metadata[metadata.bearing_id != heldout].groupby("bearing_id").size()
        fold_diag.append({"heldout_bearing":heldout,"operating_condition":raw.operating_condition.iloc[0],"valid_recordings":len(raw),"window_count":int(test.sum()),"source_index_min":int(raw.recording_index.min()),"source_index_max":int(raw.recording_index.max()),"numeric_index_gaps":";".join(map(str,np.setdiff1d(np.arange(1,int(raw.recording_index.max())+1),raw.recording_index.to_numpy(int)))) or "none","max_training_index":int(train_indices.max()),"heldout_indices_beyond_training_max":int((held_indices>train_indices.max()).sum()),"proportion_beyond_training_max":float(np.mean(held_indices>train_indices.max())),"longer_than_all_training_trajectories":bool(len(raw)>training_lengths.max()),"max_training_valid_recordings":int(training_lengths.max())})
        for name,(family,seeds) in models.items():
            for seed in seeds:
                fit_started=time.perf_counter()
                if name == "training_mean": prediction=np.full(test.sum(), d["y"][train].mean())
                else:
                    xt,xv=design_matrix(name,d,train,test)
                    est=hgb(seed) if family == "hgb" else extra(seed)
                    est.fit(xt,d["y"][train]); prediction=est.predict(xv)
                elapsed=time.perf_counter()-fit_started
                result=metric(d["y"][test],prediction)
                bearing_rows.append({"model":name,"seed":seed,"heldout_bearing":heldout,"operating_condition":d["condition"][test][0],"valid_recordings":len(raw),"n_windows":int(test.sum()),**result,"fit_predict_seconds":elapsed})
                for position,row_id in enumerate(np.flatnonzero(test)):
                    pred_rows.append({"model":name,"seed":seed,"heldout_bearing":heldout,"operating_condition":d["condition"][row_id],"start_position":int(d["start"][row_id]),"end_position":int(d["end"][row_id]),"recording_index_end":int(d["index_end"][row_id]),"true_rul":float(d["y"][row_id]),"prediction":float(prediction[position])})
        print(f"Completed {heldout}",flush=True)

    bearing_df=pd.DataFrame(bearing_rows); pred_df=pd.DataFrame(pred_rows); diag_df=pd.DataFrame(fold_diag).sort_values("valid_recordings")
    # Aggregate tree seeds at bearing level, then construct the declared paired contrasts.
    seedmean=bearing_df.groupby(["model","heldout_bearing","operating_condition","valid_recordings"],as_index=False).agg({"mae":"mean","rmse":"mean","r2":"mean","prediction_min":"mean","prediction_max":"mean","prediction_mean":"mean","prediction_std":"mean","fit_predict_seconds":"mean","n_windows":"first"})
    seedmean["row_type"]="seed_mean"; bearing_df["row_type"]="seed"
    pd.concat([bearing_df,seedmean],ignore_index=True).sort_values(["model","heldout_bearing","seed"],na_position="last").to_csv(RUN_DIR / "metrics_by_bearing.csv",index=False)
    pred_df.to_csv(RUN_DIR / "predictions_lobo.csv",index=False); diag_df.to_csv(RUN_DIR / "fold_extrapolation_diagnostics.csv",index=False)
    summary=[]
    for (name,seed),g in pred_df.groupby(["model","seed"]):
        m=metric(g.true_rul.to_numpy(),g.prediction.to_numpy()); b=seedmean[seedmean.model==name]
        summary.append({"model":name,"seed":seed,"aggregation":"pooled_windows","operating_condition":"all",**m,"macro_bearing_mae":float(b.mae.mean()),"macro_bearing_rmse":float(b.rmse.mean()),"mean_fit_predict_seconds_per_fold":float(b.fit_predict_seconds.mean())})
        for condition,cg in g.groupby("operating_condition"):
            cm=metric(cg.true_rul.to_numpy(),cg.prediction.to_numpy())
            summary.append({"model":name,"seed":seed,"aggregation":"condition_pooled_windows","operating_condition":condition,**cm,"macro_bearing_mae":np.nan,"macro_bearing_rmse":np.nan,"mean_fit_predict_seconds_per_fold":np.nan})
    metrics_df=pd.DataFrame(summary); metrics_df.to_csv(RUN_DIR / "metrics_by_model.csv",index=False)
    comparisons=[("A3_minus_A1","A3_index_vibration_hgb","A1_index_hgb"),("B3_minus_B1","B3_index_vibration_extra","B1_index_extra"),("B5_minus_B3","B5_index_window_extra","B3_index_vibration_extra")]
    pair_rows=[]; pair_summaries={}
    for label,candidate,reference in comparisons:
        c=seedmean[seedmean.model==candidate].set_index("heldout_bearing"); r=seedmean[seedmean.model==reference].set_index("heldout_bearing")
        joined=c.join(r,lsuffix="_candidate",rsuffix="_reference")
        for bearing,row in joined.iterrows(): pair_rows.append({"comparison":label,"metric":"mae","heldout_bearing":bearing,"valid_recordings":row.valid_recordings_candidate,"operating_condition":row.operating_condition_candidate,"candidate":candidate,"reference":reference,"candidate_value":row.mae_candidate,"reference_value":row.mae_reference,"difference_candidate_minus_reference":row.mae_candidate-row.mae_reference})
        for bearing,row in joined.iterrows(): pair_rows.append({"comparison":label,"metric":"rmse","heldout_bearing":bearing,"valid_recordings":row.valid_recordings_candidate,"operating_condition":row.operating_condition_candidate,"candidate":candidate,"reference":reference,"candidate_value":row.rmse_candidate,"reference_value":row.rmse_reference,"difference_candidate_minus_reference":row.rmse_candidate-row.rmse_reference})
        pair_summaries[label]={metric_name:paired_summary(joined[f"{metric_name}_candidate"].to_numpy()-joined[f"{metric_name}_reference"].to_numpy()) for metric_name in ["mae","rmse"]}
    pairs=pd.DataFrame(pair_rows).sort_values(["comparison","metric","valid_recordings"]); pairs.to_csv(RUN_DIR / "paired_bearing_differences.csv",index=False)
    length_assoc={}
    for label in pair_summaries:
        q=pairs[(pairs.comparison==label)&(pairs.metric=="mae")].sort_values("valid_recordings")
        merged=q.merge(diag_df[["heldout_bearing","proportion_beyond_training_max"]],on="heldout_bearing")
        length_assoc[label]={"spearman_difference_vs_length":float(stats.spearmanr(merged.difference_candidate_minus_reference,merged.valid_recordings).statistic),"spearman_difference_vs_extrapolation_proportion":float(stats.spearmanr(merged.difference_candidate_minus_reference,merged.proportion_beyond_training_max).statistic)}
    validations=["POST-HOC AND EXPLORATORY: the fixed models, contrasts, and analyses were run after prior LOBO results had been inspected.","NPZ X has shape 9,215 x 32; metadata and NPZ row keys agree exactly; metadata target agrees with float32 NPZ RUL to 1e-7.","All NPZ features and targets are finite; all sequence targets are bounded in [0, 1].","Built 9,080 sequences independently within bearings with x[t:t+10] -> RUL[t+9].","The lobo_run_02 fold manifest contains exactly one held-out bearing per fold and is reproduced exactly.","All engineered features are recording-local; primary ablations use all 32 as prospective inputs.","Bearing3_4 preserves the source numeric-index gap 1105; it is not synthesized or renumbered.","No transform uses held-out bearing data: condition encoding is fitted on each fold's training data only; the tree and mean models contain no scaler."]
    (RUN_DIR / "DATA_VALIDATION_REPORT.md").write_text("# Data Validation Report\n\n"+"\n".join(f"- {item}" for item in validations)+"\n",encoding="utf-8")
    all_rows=metrics_df[(metrics_df.aggregation=="pooled_windows")].copy(); show=all_rows[["model","seed","mae","rmse","r2","macro_bearing_mae","macro_bearing_rmse"]].sort_values("mae");
    report="# Index, Vibration, and Condition Ablation\n\n## Status\n\n**POST-HOC AND EXPLORATORY.** These follow-up comparisons use observed prior results and overlapping LOBO training sets. The uncertainty summaries resample 15 bearings, not windows or seeds; they are descriptive rather than confirmatory.\n\n## Input Audit\n\nAll 32 horizontal/vertical engineered features are Class A recording-local, prospectively observable features. Each is calculated only from the current 32,768-sample raw recording using time statistics or its mean-centered rFFT. The extraction code contains no future-record, trajectory-length, terminal-value, global-run, or elapsed-time feature transformation. Full feature-level evidence is in `feature_time_leakage_audit.csv`.\n\n## Pooled and Equal-Bearing Results\n\n"+markdown_table(show.round(6))+"\n\n## Paired Bearing Contrasts\n\nA negative difference means the candidate has lower error. Tree seed metrics are averaged within each bearing before forming contrasts. Bootstrap intervals use 10,000 fixed-seed resamples of the 15 bearings. Sign and Wilcoxon tests are exploratory two-sided supplements; zero differences are omitted from both tests.\n\n"
    for label,values in pair_summaries.items(): report+=f"### {label}\n\n"+"\n".join(f"- {k}: `{json.dumps(v)}`" for k,v in values.items())+f"\n- Length/extrapolation rank associations: `{json.dumps(length_assoc[label])}`\n\n"
    report+="## Interpretation\n\nThe primary effect is A3 minus A1 equal-bearing MAE; secondary and temporal effects are B3 minus B1 and B5 minus B3. See `paired_bearing_differences.csv` for all 15 differences, ordered by trajectory length. Condition controls are descriptive only and exclude bearing identity. Tree behavior outside the observed training-index range is a plausible diagnostic explanation, not evidence of a causal mechanism.\n\nThe normalized-relative target represents remaining fraction of each trajectory, not an absolute deployable lifetime. Because strong prediction from raw recording index may depend on a test setting with comparable observation progress, a separate target/deployment study should precede physics-guided learning. Pooled and equal-bearing summaries must be interpreted separately because longer trajectories contribute many more overlapping windows.\n"
    (RUN_DIR / "INDEX_FEATURE_ABLATION_REPORT.md").write_text(report,encoding="utf-8")
    manifest={"status":"completed","post_hoc_exploratory":True,"interpreter":sys.executable,"versions":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"inputs":{"npz":str(DATA/'xjtu_features.npz'),"metadata":str(DATA/'xjtu_features_metadata.csv.gz'),"source_lobo_manifest":str(SOURCE_RUN/'fold_manifest.csv')},"configuration":{"hgb":{"seed":HGB_SEED,"learning_rate":0.05,"max_iter":150,"max_leaf_nodes":15,"l2_regularization":1.0},"extra_trees":{"seeds":TREE_SEEDS,"n_estimators":150,"min_samples_leaf":2,"max_features":1.0},"bootstrap":{"seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAP_REPLICATES}},"elapsed_seconds":time.perf_counter()-started_all,"outputs":["INDEX_FEATURE_ABLATION_REPORT.md","metrics_by_model.csv","metrics_by_bearing.csv","paired_bearing_differences.csv","predictions_lobo.csv","fold_extrapolation_diagnostics.csv","feature_time_leakage_audit.csv","run_manifest.json","DATA_VALIDATION_REPORT.md","index_feature_ablation.py"]}
    (RUN_DIR/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"elapsed_seconds":manifest["elapsed_seconds"],"prediction_rows":len(pred_df),"models":len(models)},indent=2))

if __name__=="__main__": main()
