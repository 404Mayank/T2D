"""Tabular MLP heads: CORN (K-1), CORAL (K-1), CE (K)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int = 64,
        dropout: float = 0.4,
        activation: str = "relu",
    ):
        super().__init__()
        act: nn.Module
        if activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            type(act)(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    arm: str,
    d_in: int,
    n_classes: int = 4,
    hidden: int = 64,
    dropout: float = 0.4,
) -> TabularMLP:
    arm = arm.lower()
    if arm in ("corn", "coral"):
        d_out = n_classes - 1
    elif arm in ("ce", "ce_mlp", "softmax"):
        d_out = n_classes
    else:
        raise ValueError(f"unknown arm={arm!r}")
    return TabularMLP(d_in=d_in, d_out=d_out, hidden=hidden, dropout=dropout)
