"""Relational Knowledge Distillation (Park et al., CVPR 2019).

Distance-wise + angle-wise losses on projected embeddings.
Not pointwise raw-z MSE (frozen B4-B) — relational geometry only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP projection for RKD/CRD (dim 128 default)."""

    def __init__(self, d_in: int, d_out: int = 128, hidden: int | None = None):
        super().__init__()
        h = hidden if hidden is not None else max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _pdist(e: torch.Tensor, squared: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise Euclidean distances [B,B]."""
    e2 = (e * e).sum(dim=1, keepdim=True)
    dist = e2 + e2.t() - 2.0 * (e @ e.t())
    dist = dist.clamp(min=eps)
    if not squared:
        dist = dist.sqrt()
    # zero diagonal
    dist = dist * (1.0 - torch.eye(e.size(0), device=e.device, dtype=e.dtype))
    return dist


def rkd_distance_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean-normalized pairwise distance matching (Huber)."""
    if student.size(0) < 2:
        return student.new_zeros(())
    with torch.no_grad():
        t = _pdist(teacher, squared=False, eps=eps)
        mean_t = t.sum() / (t.numel() - t.size(0) + eps)
        t = t / (mean_t + eps)
    s = _pdist(student, squared=False, eps=eps)
    mean_s = s.sum() / (s.numel() - s.size(0) + eps)
    s = s / (mean_s + eps)
    return F.smooth_l1_loss(s, t, reduction="mean")


def rkd_angle_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Triplet angle matching (mean over valid triplets)."""
    b = student.size(0)
    if b < 3:
        return student.new_zeros(())

    def _angles(e: torch.Tensor) -> torch.Tensor:
        # unit vectors of pairwise diffs: for each i,j: e_i - e_j
        # Park: angle at j between i and k via normalized (e_i-e_j), (e_k-e_j)
        e_exp = e.unsqueeze(0)  # [1,B,D]
        diff = e_exp - e.unsqueeze(1)  # [B,B,D]  diff[j,i] = e_i - e_j
        # normalize
        norm = diff.norm(dim=2, keepdim=True).clamp_min(eps)
        u = diff / norm
        # cos angle at j between i and k: sum_d u[j,i]*u[j,k] → [B,B,B]
        # memory: B^3 — OK for batch ≤ 32–64
        cos = torch.einsum("jid,jkd->jik", u, u).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return cos

    with torch.no_grad():
        ta = _angles(teacher)
    sa = _angles(student)
    # mask self-pairs on i or k == j and i==k
    idx = torch.arange(b, device=student.device)
    mask = torch.ones(b, b, b, dtype=torch.bool, device=student.device)
    mask[idx, :, idx] = False  # i==j
    mask[:, idx, idx] = False  # k==j
    # i==k still has angle 1; include or drop — drop diagonal i==k
    eye = torch.eye(b, dtype=torch.bool, device=student.device)
    mask = mask & (~eye.unsqueeze(0))
    if not mask.any():
        return student.new_zeros(())
    return F.smooth_l1_loss(sa[mask], ta[mask], reduction="mean")


def rkd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    dist_ratio: float = 1.0,
    angle_ratio: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Combined RKD. teacher should be stopgrad before call."""
    if student.size(0) < 2:
        return student.new_zeros(())
    ld = rkd_distance_loss(student, teacher, eps=eps) if dist_ratio != 0 else student.new_zeros(())
    la = rkd_angle_loss(student, teacher, eps=eps) if angle_ratio != 0 else student.new_zeros(())
    return float(dist_ratio) * ld + float(angle_ratio) * la
