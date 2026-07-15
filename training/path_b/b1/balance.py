"""Gradient-balanced multi-task helpers for B1 GS (PCGrad + uncertainty weights).

Locks: PLAN_B1_GS.md §4–5.
Shared params = input / lstm / proj only (attn is CE-exclusive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def shared_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters that both CE and glu paths flow through.

    attn / class_head are CE-only; glu_head is glu-only.
    """
    out: list[nn.Parameter] = []
    for name in ("input", "lstm", "proj"):
        mod = getattr(model, name, None)
        if mod is None or isinstance(mod, nn.Identity):
            continue
        out.extend(p for p in mod.parameters() if p.requires_grad)
    return out


def exclusive_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Heads + attn (not in PCGrad shared set)."""
    out: list[nn.Parameter] = []
    for name in ("attn", "class_head", "glu_head"):
        mod = getattr(model, name, None)
        if mod is None:
            continue
        out.extend(p for p in mod.parameters() if p.requires_grad)
    return out


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
) -> torch.Tensor | None:
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts)


def unflatten_grads(
    flat: torch.Tensor,
    like: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    out: list[torch.Tensor | None] = []
    offset = 0
    for g in like:
        if g is None:
            out.append(None)
            continue
        n = g.numel()
        out.append(flat[offset : offset + n].view_as(g))
        offset += n
    return out


def pcgrad_project(
    g_i: torch.Tensor,
    g_j: torch.Tensor,
) -> torch.Tensor:
    """Project g_i to remove conflicting component of g_j (if cos < 0)."""
    dot = torch.dot(g_i, g_j)
    if dot >= 0:
        return g_i
    denom = torch.dot(g_j, g_j).clamp_min(1e-12)
    return g_i - (dot / denom) * g_j


def pcgrad_combine(
    g_a: torch.Tensor,
    g_b: torch.Tensor,
    *,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PCGrad over two flattened shared grads.

    Convention (locked): each task is projected onto the **original** other
    gradient (not the already-projected vector). For two tasks this is
    commutative, so random order is a no-op for the sum; we still draw the
    coin for API parity with multi-task PCGrad. Matches Yu et al. Alg.1 for
    |T|=2 when g_j is the pre-update task gradient.
    """
    if rng is None:
        first_a = bool(torch.randint(0, 2, (1,)).item())
    else:
        first_a = bool(torch.randint(0, 2, (1,), generator=rng).item())

    # Project each onto the original peer (not sequential mutate).
    a_p = pcgrad_project(g_a, g_b)
    b_p = pcgrad_project(g_b, g_a)
    # first_a only affects reporting order; sum is order-invariant for |T|=2
    combined = (a_p + b_p) if first_a else (b_p + a_p)

    na = torch.norm(g_a).clamp_min(1e-12)
    nb = torch.norm(g_b).clamp_min(1e-12)
    cos = float((torch.dot(g_a, g_b) / (na * nb)).item())
    stats = {
        "cos": cos,
        "conflict": float(cos < 0.0),
        "norm_ce": float(na.item()),
        "norm_glu": float(nb.item()),
        "norm_ratio": float((nb / na).item()),
        "first_a": float(first_a),
    }
    return combined, stats


@dataclass
class ConflictMeter:
    """Accumulate cos / conflict only on glu-active steps."""

    n_train_steps: int = 0
    n_glu_active: int = 0
    n_conflict: int = 0
    cos_sum: float = 0.0
    norm_ce_sum: float = 0.0
    norm_glu_sum: float = 0.0
    # epoch buffers
    epoch_rows: list[dict[str, float]] = field(default_factory=list)

    def step_total(self) -> None:
        self.n_train_steps += 1

    def step_glu_active(self, stats: dict[str, float]) -> None:
        self.n_glu_active += 1
        self.cos_sum += float(stats["cos"])
        self.n_conflict += int(stats["conflict"] > 0.5)
        self.norm_ce_sum += float(stats["norm_ce"])
        self.norm_glu_sum += float(stats["norm_glu"])

    def epoch_summary(self, epoch: int) -> dict[str, float]:
        n = max(self.n_glu_active, 1)
        row = {
            "epoch": float(epoch),
            "n_train_steps": float(self.n_train_steps),
            "n_glu_active_steps": float(self.n_glu_active),
            "conflict_rate": float(self.n_conflict / n) if self.n_glu_active else float("nan"),
            "mean_cos": float(self.cos_sum / n) if self.n_glu_active else float("nan"),
            "mean_norm_ce": float(self.norm_ce_sum / n) if self.n_glu_active else float("nan"),
            "mean_norm_glu": float(self.norm_glu_sum / n) if self.n_glu_active else float("nan"),
            "mean_norm_ratio": (
                float(self.norm_glu_sum / max(self.norm_ce_sum, 1e-12))
                if self.n_glu_active
                else float("nan")
            ),
        }
        self.epoch_rows.append(row)
        # reset per-epoch accumulators (keep epoch_rows)
        self.n_train_steps = 0
        self.n_glu_active = 0
        self.n_conflict = 0
        self.cos_sum = 0.0
        self.norm_ce_sum = 0.0
        self.norm_glu_sum = 0.0
        return row


class UncertaintyWeights(nn.Module):
    """Homoscedastic uncertainty with CE-primary prior (PLAN_B1_GS §4.4).

    L = exp(-s_ce)*L_ce + s_ce + 0.5*exp(-s_glu)*L_glu + 0.5*s_glu
    """

    def __init__(self, clamp: float = 5.0):
        super().__init__()
        self.s_ce = nn.Parameter(torch.zeros(()))
        self.s_glu = nn.Parameter(torch.zeros(()))
        self.clamp = float(clamp)

    def clamped(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.s_ce.clamp(-self.clamp, self.clamp),
            self.s_glu.clamp(-self.clamp, self.clamp),
        )

    def clamp_hits(self) -> dict[str, float]:
        """1.0 if raw |s| hit clamp bound this step (diagnostic)."""
        c = self.clamp
        return {
            "clamp_hit_ce": float(self.s_ce.detach().abs().item() >= c - 1e-8),
            "clamp_hit_glu": float(self.s_glu.detach().abs().item() >= c - 1e-8),
        }

    def effective_weights(self) -> dict[str, float]:
        s_ce, s_glu = self.clamped()
        out = {
            "s_ce": float(s_ce.detach().cpu()),
            "s_glu": float(s_glu.detach().cpu()),
            "w_ce": float(torch.exp(-s_ce).detach().cpu()),
            "w_glu": float((0.5 * torch.exp(-s_glu)).detach().cpu()),
            "s_ce_raw": float(self.s_ce.detach().cpu()),
            "s_glu_raw": float(self.s_glu.detach().cpu()),
        }
        out.update(self.clamp_hits())
        return out

    def combine(
        self, loss_ce: torch.Tensor, loss_glu: torch.Tensor, *, glu_active: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, weighted_ce, weighted_glu (data terms only for surgery)."""
        s_ce, s_glu = self.clamped()
        w_ce = torch.exp(-s_ce)
        w_glu = 0.5 * torch.exp(-s_glu)
        term_ce = w_ce * loss_ce
        term_glu = w_glu * loss_glu if glu_active else loss_ce.new_zeros(())
        total = term_ce + s_ce + term_glu + 0.5 * s_glu
        return total, term_ce, term_glu


def apply_grads_(
    params: Sequence[nn.Parameter],
    grads: Sequence[torch.Tensor | None],
) -> None:
    for p, g in zip(params, grads):
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach().clone()


def pcgrad_backward(
    *,
    model: nn.Module,
    loss_ce: torch.Tensor,
    loss_glu: torch.Tensor,
    glu_active: bool,
    meter: ConflictMeter | None = None,
    rng: torch.Generator | None = None,
    also_params: Iterable[nn.Parameter] | None = None,
    retain_graph: bool = False,
) -> dict[str, float]:
    """Compute PCGrad on shared params; plain grads on exclusive (+ also_params).

    Returns step stats (empty conflict fields if glu inactive).
    retain_graph=True keeps the graph for a subsequent autograd on total (UW s_*).
    """
    shared = shared_parameters(model)
    exclusive = exclusive_parameters(model)
    extra = list(also_params or [])
    # de-dupe exclusive+extra by id
    seen = {id(p) for p in shared}
    head_params: list[nn.Parameter] = []
    for p in list(exclusive) + extra:
        if id(p) not in seen:
            head_params.append(p)
            seen.add(id(p))

    if meter is not None:
        meter.step_total()

    # CE grads (shared + heads that CE touches)
    grads_ce_shared = torch.autograd.grad(
        loss_ce, shared, retain_graph=True, allow_unused=True
    )
    # heads: CE may not touch glu_head
    grads_ce_head = torch.autograd.grad(
        loss_ce, head_params, retain_graph=True, allow_unused=True
    )

    if not glu_active:
        apply_grads_(shared, grads_ce_shared)
        apply_grads_(head_params, grads_ce_head)
        return {"glu_active": 0.0, "conflict": float("nan"), "cos": float("nan")}

    grads_glu_shared = torch.autograd.grad(
        loss_glu, shared, retain_graph=True, allow_unused=True
    )
    grads_glu_head = torch.autograd.grad(
        loss_glu,
        head_params,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    flat_ce = flatten_grads(grads_ce_shared)
    flat_glu = flatten_grads(grads_glu_shared)
    if flat_ce is None or flat_glu is None:
        # degenerate: fall back to CE
        apply_grads_(shared, grads_ce_shared)
        # head: sum where both exist
        head_sum = []
        for gc, gg in zip(grads_ce_head, grads_glu_head):
            if gc is None and gg is None:
                head_sum.append(None)
            elif gc is None:
                head_sum.append(gg)
            elif gg is None:
                head_sum.append(gc)
            else:
                head_sum.append(gc + gg)
        apply_grads_(head_params, head_sum)
        return {"glu_active": 0.0, "conflict": float("nan"), "cos": float("nan")}

    # zero-fill unused shared slots for flatten consistency already handled
    # Replace Nones with zeros matching param shapes for combine/unflatten
    ce_filled = [
        g if g is not None else torch.zeros_like(p)
        for p, g in zip(shared, grads_ce_shared)
    ]
    glu_filled = [
        g if g is not None else torch.zeros_like(p)
        for p, g in zip(shared, grads_glu_shared)
    ]
    flat_ce = flatten_grads(ce_filled)
    flat_glu = flatten_grads(glu_filled)
    assert flat_ce is not None and flat_glu is not None

    if float(flat_glu.norm().item()) < 1e-12:
        apply_grads_(shared, grads_ce_shared)
        apply_grads_(head_params, grads_ce_head)
        return {"glu_active": 0.0, "conflict": float("nan"), "cos": float("nan")}

    combined, stats = pcgrad_combine(flat_ce, flat_glu, rng=rng)
    shared_grads = unflatten_grads(combined, ce_filled)
    apply_grads_(shared, shared_grads)

    head_sum = []
    for gc, gg in zip(grads_ce_head, grads_glu_head):
        if gc is None and gg is None:
            head_sum.append(None)
        elif gc is None:
            head_sum.append(gg)
        elif gg is None:
            head_sum.append(gc)
        else:
            head_sum.append(gc + gg)
    apply_grads_(head_params, head_sum)

    if meter is not None:
        meter.step_glu_active(stats)
    out = {"glu_active": 1.0, **stats}
    return out


def plain_backward(
    *,
    model: nn.Module,
    loss: torch.Tensor,
    also_params: Iterable[nn.Parameter] | None = None,
) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if also_params is not None:
        seen = {id(p) for p in params}
        for p in also_params:
            if id(p) not in seen and p.requires_grad:
                params.append(p)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    apply_grads_(params, grads)
