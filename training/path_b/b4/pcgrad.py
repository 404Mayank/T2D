"""PCGrad: project conflicting task gradients (Yu et al., NeurIPS 2020).

Used on shared encoder params for CE vs traj multi-task (B4-A-V2).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _flatten_grads(params: Sequence[nn.Parameter]) -> torch.Tensor | None:
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None
    return torch.cat(grads)


def _write_grads(params: Sequence[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        g = flat[offset : offset + n].view_as(p)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += n


def grad_cosine(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    n1 = g1.norm()
    n2 = g2.norm()
    if float(n1) < eps or float(n2) < eps:
        return float("nan")
    return float((g1 @ g2) / (n1 * n2 + eps))


def pcgrad_step(
    losses: Sequence[torch.Tensor],
    shared_params: Iterable[nn.Parameter],
    *,
    retain_graph: bool = False,
) -> dict[str, float]:
    """
    For each task loss, compute grad on shared_params, then PCGrad-project
    pairwise when cos < 0. Writes final combined grad into .grad of shared_params.

    Non-shared params should have their grads computed separately by the caller
    (backward each head loss onto head params, or full-model backward with
    create_graph=False after this).

    Returns diagnostics including mean pairwise cos before projection.
    """
    params = [p for p in shared_params if p.requires_grad]
    if not params or not losses:
        return {"cos_mean": float("nan"), "n_conflicts": 0}

    # collect per-task grads
    task_grads: list[torch.Tensor] = []
    cos_list: list[float] = []
    for i, loss in enumerate(losses):
        for p in params:
            if p.grad is not None:
                p.grad = None
        # retain graph for all but last unless caller forces
        rg = retain_graph or (i < len(losses) - 1)
        if loss.requires_grad:
            loss.backward(retain_graph=rg)
        g = _flatten_grads(params)
        assert g is not None
        task_grads.append(g.clone())
        for p in params:
            if p.grad is not None:
                p.grad = None

    # pairwise cos before PCGrad
    for i in range(len(task_grads)):
        for j in range(i + 1, len(task_grads)):
            cos_list.append(grad_cosine(task_grads[i], task_grads[j]))

    # PCGrad projection (in-place on copies)
    g_pc = [g.clone() for g in task_grads]
    n_conflicts = 0
    # random order per Yu et al.
    order = torch.randperm(len(g_pc)).tolist()
    for i in order:
        for j in order:
            if i == j:
                continue
            gi, gj = g_pc[i], task_grads[j]
            dot = torch.dot(gi, gj)
            if float(dot) < 0:
                n_conflicts += 1
                gj_norm2 = torch.dot(gj, gj).clamp_min(1e-12)
                g_pc[i] = gi - (dot / gj_norm2) * gj

    # Mean (not sum) so effective shared LR matches a single combined loss path
    # (H-5: sum doubles scale vs CE+traj backward).
    combined = torch.stack(g_pc, dim=0).mean(dim=0)
    _write_grads(params, combined)
    cos_mean = float(sum(cos_list) / len(cos_list)) if cos_list else float("nan")
    return {"cos_mean": cos_mean, "n_conflicts": n_conflicts, "n_pairs": len(cos_list)}
