"""Contrastive Representation Distillation (Tian et al., ICLR 2020) — compact.

Memory bank of student projections; InfoNCE vs teacher projection.
Bank never stores teacher z / val / test (PLAN_B4_V2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """FIFO queue of student projected features (negatives)."""

    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = int(size)
        self.dim = int(dim)
        self.register_buffer("bank", torch.randn(self.size, self.dim))
        self.bank = F.normalize(self.bank, dim=1)
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor) -> None:
        """z: [B, D] student projections (will normalize)."""
        if z.numel() == 0:
            return
        z = F.normalize(z.detach(), dim=1)
        b = z.size(0)
        ptr = int(self.ptr.item())
        if b >= self.size:
            self.bank.copy_(z[-self.size :])
            self.ptr.fill_(0)
            self.filled.fill_(self.size)
            return
        end = ptr + b
        if end <= self.size:
            self.bank[ptr:end] = z
        else:
            first = self.size - ptr
            self.bank[ptr:] = z[:first]
            self.bank[: end % self.size] = z[first:]
        self.ptr.fill_(end % self.size)
        self.filled.fill_(min(self.size, int(self.filled.item()) + b))

    def n_valid(self) -> int:
        return int(self.filled.item())


def crd_nce_loss(
    student_proj: torch.Tensor,
    teacher_proj: torch.Tensor,
    bank: MemoryBank,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE: positive = matched teacher proj; negatives = memory bank.
    student_proj / teacher_proj: [B, D] (only aux rows for distill).
    """
    if student_proj.size(0) == 0:
        return student_proj.new_zeros(())
    s = F.normalize(student_proj, dim=1)
    t = F.normalize(teacher_proj.detach(), dim=1)
    # positive logits [B,1]
    pos = (s * t).sum(dim=1, keepdim=True) / temperature
    n = bank.n_valid()
    if n < 1:
        # no negatives yet — fall back to in-batch other teachers as weak negs
        if s.size(0) < 2:
            return s.new_zeros(())
        logits_ib = (s @ t.t()) / temperature  # [B,B]
        labels = torch.arange(s.size(0), device=s.device)
        return F.cross_entropy(logits_ib, labels)
    # clone so later FIFO enqueue (inplace on bank buffer) cannot break backward
    neg_bank = bank.bank[:n].detach().clone()
    neg = (s @ neg_bank.t()) / temperature  # [B, n]
    logits = torch.cat([pos, neg], dim=1)  # [B, 1+n]
    labels = torch.zeros(s.size(0), dtype=torch.long, device=s.device)
    return F.cross_entropy(logits, labels)
