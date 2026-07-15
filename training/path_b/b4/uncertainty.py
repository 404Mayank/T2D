"""Kendall uncertainty weighting for two-task MTL (CE + traj)."""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeights(nn.Module):
    """Learnable log-variances: L = exp(-s1)*L1 + s1 + exp(-s2)*L2 + s2."""

    def __init__(self) -> None:
        super().__init__()
        self.log_var_ce = nn.Parameter(torch.zeros(()))
        self.log_var_traj = nn.Parameter(torch.zeros(()))

    def combine(self, loss_ce: torch.Tensor, loss_traj: torch.Tensor) -> torch.Tensor:
        # precision = exp(-log_var)
        p_ce = torch.exp(-self.log_var_ce)
        p_tr = torch.exp(-self.log_var_traj)
        return p_ce * loss_ce + self.log_var_ce + p_tr * loss_traj + self.log_var_traj

    def state_dict_small(self) -> dict[str, float]:
        return {
            "log_var_ce": float(self.log_var_ce.detach()),
            "log_var_traj": float(self.log_var_traj.detach()),
            "w_ce": float(torch.exp(-self.log_var_ce.detach())),
            "w_traj": float(torch.exp(-self.log_var_traj.detach())),
        }
