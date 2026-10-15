"""Sequence dataset utilities for XJTU-SY feature experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceBundle:
    x: np.ndarray
    y: np.ndarray
    fault_label_id: np.ndarray
    bearing_id: np.ndarray
    start_position: np.ndarray
    end_position: np.ndarray
    recording_index_end: np.ndarray


class RULSequenceDataset(Dataset):
    def __init__(self, bundle: SequenceBundle):
        self.x = torch.as_tensor(bundle.x, dtype=torch.float32)
        self.y = torch.as_tensor(bundle.y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_processed_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.sort_values(["bearing_id", "chronological_position"]).reset_index(drop=True)


def fit_scaler(frame: pd.DataFrame, feature_columns: list[str], train_bearings: Iterable[str]) -> StandardScaler:
    train_bearing_set = set(train_bearings)
    train_rows = frame["bearing_id"].isin(train_bearing_set)
    if not train_rows.any():
        raise ValueError("No rows found for scaler training bearings")
    scaler = StandardScaler()
    scaler.fit(frame.loc[train_rows, feature_columns].to_numpy(dtype=np.float32))
    return scaler


def transform_features(frame: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)


def make_sequences(
    frame: pd.DataFrame,
    transformed_features: np.ndarray,
    bearings: Iterable[str],
    sequence_length: int,
    stride: int,
) -> SequenceBundle:
    bearing_set = set(bearings)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    bearing_ids: list[str] = []
    fault_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    recording_ends: list[int] = []

    for bearing_id in sorted(bearing_set):
        bearing_index = frame.index[frame["bearing_id"] == bearing_id].to_numpy()
        if bearing_index.size == 0:
            raise ValueError(f"No rows found for bearing {bearing_id}")
        bearing_frame = frame.loc[bearing_index].sort_values("chronological_position")
        ordered_index = bearing_frame.index.to_numpy()
        positions = bearing_frame["chronological_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(positions))):
            raise ValueError(f"Chronological positions are not contiguous for {bearing_id}")

        for start in range(0, len(ordered_index) - sequence_length + 1, stride):
            end = start + sequence_length
            window_index = ordered_index[start:end]
            window_bearings = frame.loc[window_index, "bearing_id"].unique()
            if len(window_bearings) != 1 or window_bearings[0] != bearing_id:
                raise ValueError(f"Sequence crosses bearing boundary for {bearing_id}")

            target_row = frame.loc[window_index[-1]]
            xs.append(transformed_features[window_index])
            ys.append(float(target_row["rul"]))
            fault_ids.append(int(target_row["fault_label_id"]))
            bearing_ids.append(bearing_id)
            starts.append(int(frame.loc[window_index[0], "chronological_position"]))
            ends.append(int(target_row["chronological_position"]))
            recording_ends.append(int(target_row["recording_index"]))

    if not xs:
        raise ValueError("No sequences were created")

    return SequenceBundle(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        fault_label_id=np.asarray(fault_ids, dtype=np.int64),
        bearing_id=np.asarray(bearing_ids),
        start_position=np.asarray(starts, dtype=np.int64),
        end_position=np.asarray(ends, dtype=np.int64),
        recording_index_end=np.asarray(recording_ends, dtype=np.int64),
    )
