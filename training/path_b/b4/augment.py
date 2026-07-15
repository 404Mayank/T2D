"""Mask-aware time-series augmentation for grid tensors (B4-V2).

No CGM in student path. Does not move pad boundary.
"""

from __future__ import annotations

import torch


def augment_wear_batch(
    x: torch.Tensor,
    wear_mask: torch.Tensor,
    *,
    jitter_std: float = 0.05,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
    p: float = 0.5,
) -> torch.Tensor:
    """
    x: [B,T,D], wear_mask: [B,T] True=observed wear.
    Applies independent jitter + global scale per sample with prob p.
    Only touches wear-valid bins; pad/missing stay 0.
    """
    if p <= 0:
        return x
    b, t, d = x.shape
    out = x.clone()
    device = x.device
    do = torch.rand(b, device=device) < p
    if not do.any():
        return out
    # jitter
    noise = torch.randn_like(out) * jitter_std
    # scale per sample
    scales = torch.empty(b, 1, 1, device=device).uniform_(scale_min, scale_max)
    aug = out * scales + noise
    m = wear_mask.unsqueeze(-1).to(out.dtype)
    # only apply where do
    do_m = do.view(b, 1, 1).to(out.dtype)
    out = out * (1 - do_m) + (aug * m) * do_m
    return out
