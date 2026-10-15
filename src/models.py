"""PyTorch models for the isolated XJTU-SY experiments."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMRULRegressor(nn.Module):
    """M1 baseline: BiLSTM encoder with one normalized RUL regression head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        last_step = encoded[:, -1, :]
        return self.rul_head(last_step)


class BiLSTMMultiTaskRegressorClassifier(nn.Module):
    """M2 baseline: shared BiLSTM with RUL and fault-class heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_fault_classes: int = 4,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        representation_size = hidden_size * 2
        self.rul_head = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.fault_head = nn.Sequential(
            nn.Linear(representation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_fault_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(x)
        representation = encoded[:, -1, :]
        return self.rul_head(representation), self.fault_head(representation)
