"""B4-A: 1D CNN patch encoder + traj decoder + class head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNNEncoder(nn.Module):
    """Patchify 5-min grid → conv encoder → per-bin states (nearest upsample)."""

    def __init__(
        self,
        d_in: int,
        *,
        hidden: int = 64,
        patch_size: int = 12,
        patch_stride: int = 12,
        dropout: float = 0.2,
        n_classes: int = 4,
    ):
        super().__init__()
        self.hidden = hidden
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=patch_size, stride=patch_stride),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.attn = nn.Linear(hidden, 1)
        self.class_head = nn.Linear(hidden, n_classes)
        # per-bin CGM from upsampled states
        self.traj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,T,d], valid_mask: [B,T] True = observed (not pad; preferably wear)
        returns h_t [B,T,H], z [B,H], alpha [B,T]
        """
        b, t, d = x.shape
        # Conv1d wants [B,C,T]
        xc = x.transpose(1, 2)
        # zero padded positions before conv
        xc = xc * valid_mask.unsqueeze(1).to(xc.dtype)
        hp = self.input(xc)  # [B,H,Tp]
        # upsample patch states to T
        h_t = F.interpolate(hp, size=t, mode="linear", align_corners=False)
        h_t = h_t.transpose(1, 2)  # [B,T,H]
        h_t = self.dropout(h_t)

        scores = self.attn(h_t).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1e9)
        all_bad = ~valid_mask.any(dim=1)
        if all_bad.any():
            scores = scores.clone()
            scores[all_bad, 0] = 0.0
        alpha = torch.softmax(scores, dim=1)
        z = torch.sum(alpha.unsqueeze(-1) * h_t, dim=1)
        return h_t, z, alpha

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h_t, z, alpha = self.encode(x, valid_mask)
        logits = self.class_head(z)
        cgm_pred = self.traj_head(h_t).squeeze(-1)  # [B,T]
        return {"logits": logits, "cgm_pred": cgm_pred, "z": z, "alpha": alpha, "h_t": h_t}


def masked_traj_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """pred/target [B,T], mask [B,T]."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not mask.any():
        return pred.new_zeros(())
    diff = (pred - target)[mask]
    return diff.pow(2).mean()


def ce_loss(
    logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor | None
) -> torch.Tensor:
    if class_weights is not None:
        return F.cross_entropy(logits, y, weight=class_weights)
    return F.cross_entropy(logits, y)
