"""CORN/CORAL probability conversion + weighted CORN loss.

coral-pytorch API (1.4.0):
  from coral_pytorch.losses import corn_loss, coral_loss
  from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from coral_pytorch.dataset import corn_label_from_logits as _lib_corn_label
    from coral_pytorch.losses import corn_loss as _lib_corn_loss
except ImportError:  # pragma: no cover - hand-port fallback
    _lib_corn_label = None
    _lib_corn_loss = None


def corn_conditional_to_proba(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits (n, K-1) → rank-consistent class proba (n, K).

    CORN models s_k = P(Y > k | Y ≥ k). With cumprod:
      u_k = P(Y > k) = ∏_{j≤k} s_j
      P0 = 1 - u0; P1 = u0 - u1; ...; P_{K-1} = u_{K-2}
    """
    if logits.ndim != 2:
        raise ValueError(f"logits ndim {logits.ndim} != 2")
    s = torch.sigmoid(logits)
    u = torch.cumprod(s, dim=1)
    parts = [1.0 - u[:, 0:1]]
    for k in range(u.shape[1] - 1):
        parts.append(u[:, k : k + 1] - u[:, k + 1 : k + 2])
    parts.append(u[:, -1:])
    proba = torch.cat(parts, dim=1)
    proba = torch.clamp(proba, min=0.0)
    denom = proba.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return proba / denom


def corn_conditional_to_proba_np(logits: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    return corn_conditional_to_proba(t).numpy()


def corn_hard_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Hard rank label via count(cumprod(sigmoid) > 0.5) — matches coral-pytorch."""
    if _lib_corn_label is not None:
        return _lib_corn_label(logits)
    # hand-port of coral_pytorch.dataset.corn_label_from_logits
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    return torch.sum(predict_levels, dim=1)


def corn_loss_unweighted(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if _lib_corn_loss is not None:
        return _lib_corn_loss(logits, y, num_classes)
    return _hand_corn_loss(logits, y, num_classes, weights=None)


def _hand_corn_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    """Hand-port of coral-pytorch corn_loss with optional per-example weights."""
    y = y.long().view(-1)
    num_examples = 0.0
    losses = logits.new_zeros(())
    weight_sum = 0.0
    for task_index in range(num_classes - 1):
        label_mask = y > task_index - 1
        if not bool(label_mask.any()):
            continue
        train_labels = (y[label_mask] > task_index).to(torch.float32)
        pred = logits[label_mask, task_index]
        log_sig = F.logsigmoid(pred)
        # BCE-with-logits style: - (y log s + (1-y) log(1-s))
        per = -(log_sig * train_labels + (log_sig - pred) * (1.0 - train_labels))
        if weights is None:
            losses = losses + per.sum()
            num_examples += float(train_labels.numel())
        else:
            w = weights[label_mask]
            losses = losses + (per * w).sum()
            weight_sum += float(w.sum().item())
            num_examples += float(train_labels.numel())
    if weights is None:
        if num_examples <= 0:
            return logits.sum() * 0.0
        return losses / num_examples
    if weight_sum <= 0:
        return logits.sum() * 0.0
    # mean over weighted conditional rows (sum of weights across tasks)
    return losses / weight_sum


def corn_loss_weighted(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-example weights applied inside each CORN conditional task."""
    if weights.shape[0] != y.shape[0]:
        raise ValueError(f"weights {weights.shape} vs y {y.shape}")
    # Always use hand path so weights are honored (lib corn_loss is unweighted).
    return _hand_corn_loss(logits, y, num_classes, weights=weights)


def balanced_person_weights(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """sklearn-style balanced: n / (K * n_k)."""
    y = np.asarray(y, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    n = float(len(y))
    w_class = np.zeros(n_classes, dtype=np.float64)
    for k in range(n_classes):
        w_class[k] = n / (n_classes * counts[k]) if counts[k] > 0 else 0.0
    return w_class[y]


def coral_levels_to_proba(logits: torch.Tensor) -> torch.Tensor:
    """CORAL cumulative logits → class proba via ordered thresholds.

    P(Y > k) ≈ sigmoid(logit_k); differences give class mass (clamped).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits ndim {logits.ndim} != 2")
    u = torch.sigmoid(logits)
    # enforce nonincreasing for numerical stability
    for k in range(1, u.shape[1]):
        u[:, k] = torch.minimum(u[:, k], u[:, k - 1])
    parts = [1.0 - u[:, 0:1]]
    for k in range(u.shape[1] - 1):
        parts.append(u[:, k : k + 1] - u[:, k + 1 : k + 2])
    parts.append(u[:, -1:])
    proba = torch.cat(parts, dim=1)
    proba = torch.clamp(proba, min=0.0)
    denom = proba.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return proba / denom


def self_check() -> dict[str, Any]:
    """Unit checks runnable without data. Raises on failure."""
    # fixture from plan critique counterexample
    logits = torch.tensor([[0.3, 2.0, -0.5], [2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]])
    proba = corn_conditional_to_proba(logits)
    s = torch.sigmoid(logits)
    u = torch.cumprod(s, dim=1)
    assert torch.allclose(proba.sum(dim=1), torch.ones(3), atol=1e-5), proba.sum(1)
    assert bool((proba >= -1e-7).all())
    # nonincreasing survival
    assert bool((u[:, 1:] <= u[:, :-1] + 1e-6).all())

    hard = corn_hard_label_from_logits(logits)
    # replicate count(u > 0.5)
    hard_ref = torch.sum(u > 0.5, dim=1)
    assert torch.equal(hard, hard_ref), (hard, hard_ref)

    # counterexample: argmax may disagree with hard
    argmax = proba.argmax(dim=1)
    # first row: hard=2, argmax=0 (documented)
    assert int(hard[0].item()) == 2
    assert int(argmax[0].item()) == 0

    # weighted loss runs and is finite
    y = torch.tensor([0, 2, 3])
    w = torch.tensor([1.0, 2.0, 1.5])
    loss_w = corn_loss_weighted(logits, y, num_classes=4, weights=w)
    loss_u = corn_loss_unweighted(logits, y, num_classes=4)
    assert torch.isfinite(loss_w) and torch.isfinite(loss_u)

    # proba column 0 is P0
    assert proba.shape == (3, 4)

    return {
        "ok": True,
        "hard_labels": hard.tolist(),
        "argmax_labels": argmax.tolist(),
        "argmax_ne_hard_row0": True,
        "loss_weighted": float(loss_w.item()),
        "loss_unweighted": float(loss_u.item()),
        "lib_corn_available": _lib_corn_loss is not None,
    }


if __name__ == "__main__":
    print(self_check())
