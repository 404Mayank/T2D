"""Evaluation metrics for Path A multiclass / binary / ordinal views."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)


def _as_int(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).astype(np.int64).ravel()


def _as_proba(p: np.ndarray, n_classes: int = 4) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"proba shape {p.shape} != (*, {n_classes})")
    return p


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray, labels: list[int] | None = None) -> float:
    y = _as_int(y_true)
    p = _as_proba(proba)
    labels = labels or list(range(p.shape[1]))
    return float(
        roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels)
    )


def per_class_ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> dict[int, float]:
    y = _as_int(y_true)
    p = _as_proba(proba)
    out: dict[int, float] = {}
    for k in range(p.shape[1]):
        y_bin = (y == k).astype(int)
        if y_bin.min() == y_bin.max():
            out[k] = float("nan")
        else:
            out[k] = float(roc_auc_score(y_bin, p[:, k]))
    return out


def macro_auprc(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = _as_int(y_true)
    p = _as_proba(proba)
    scores = []
    for k in range(p.shape[1]):
        y_bin = (y == k).astype(int)
        if y_bin.sum() == 0:
            continue
        scores.append(average_precision_score(y_bin, p[:, k]))
    return float(np.mean(scores)) if scores else float("nan")


def per_class_auprc(y_true: np.ndarray, proba: np.ndarray) -> dict[int, float]:
    y = _as_int(y_true)
    p = _as_proba(proba)
    out: dict[int, float] = {}
    for k in range(p.shape[1]):
        y_bin = (y == k).astype(int)
        if y_bin.sum() == 0:
            out[k] = float("nan")
        else:
            out[k] = float(average_precision_score(y_bin, p[:, k]))
    return out


def binary_from_multiclass(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Healthy-vs-not: score = 1 - P(class 0)."""
    y = _as_int(y_true)
    p = _as_proba(proba)
    y_bin = (y > 0).astype(int)
    score = 1.0 - p[:, 0]
    if y_bin.min() == y_bin.max():
        return {"binary_auc": float("nan"), "binary_auprc": float("nan")}
    return {
        "binary_auc": float(roc_auc_score(y_bin, score)),
        "binary_auprc": float(average_precision_score(y_bin, score)),
    }


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error over one-hot vs proba (multiclass Brier)."""
    y = _as_int(y_true)
    p = _as_proba(proba)
    n, k = p.shape
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def binary_brier_from_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = _as_int(y_true)
    p = _as_proba(proba)
    y_bin = (y > 0).astype(int)
    score = 1.0 - p[:, 0]
    return float(brier_score_loss(y_bin, score))


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = _as_int(y_true)
    pred = _as_int(y_pred)
    return {
        "mae": float(np.mean(np.abs(y.astype(float) - pred.astype(float)))),
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")),
    }


def multiclass_logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = _as_int(y_true)
    p = _as_proba(proba)
    # clip for numerical safety
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(y, p, labels=list(range(p.shape[1]))))


def full_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    y_pred: np.ndarray | None = None,
    tag: str = "",
) -> dict[str, Any]:
    y = _as_int(y_true)
    p = _as_proba(proba)
    if y_pred is None:
        y_pred = p.argmax(axis=1)
    rep: dict[str, Any] = {
        "tag": tag,
        "n": int(len(y)),
        "macro_ovr_auc": macro_ovr_auc(y, p),
        "macro_auprc": macro_auprc(y, p),
        "per_class_ovr_auc": per_class_ovr_auc(y, p),
        "per_class_auprc": per_class_auprc(y, p),
        "multiclass_brier": multiclass_brier(y, p),
        "multiclass_logloss": multiclass_logloss(y, p),
        "ordinal": ordinal_metrics(y, y_pred),
    }
    rep.update(binary_from_multiclass(y, p))
    rep["binary_brier"] = binary_brier_from_multiclass(y, p)
    return rep


def binary_report(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    tag: str = "",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary metrics from labels (0/1 or multiclass>0) and P(y=1) scores."""
    y_raw = _as_int(y_true)
    # accept multiclass labels by collapsing, or already-binary
    if y_raw.min() < 0:
        raise ValueError("negative labels")
    if y_raw.max() > 1:
        y = (y_raw > 0).astype(np.int64)
    else:
        y = y_raw
    s = np.asarray(score, dtype=float).ravel()
    if len(s) != len(y):
        raise ValueError(f"score length {len(s)} != y length {len(y)}")
    base_rate = float(y.mean()) if len(y) else float("nan")
    if y.min() == y.max():
        auc = float("nan")
        auprc = float("nan")
    else:
        auc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    pred = (s >= threshold).astype(np.int64)
    return {
        "tag": tag,
        "n": int(len(y)),
        "base_rate": base_rate,
        "binary_auc": auc,
        "binary_auprc": auprc,
        "binary_brier": float(brier_score_loss(y, s)),
        "threshold": threshold,
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
    }


def select_better(
    current: dict[str, float] | None,
    candidate: dict[str, float],
    *,
    eps: float = 0.005,
) -> bool:
    """Pairwise helper — prefer ``select_best`` for multi-candidate pools.

    Kept for 2-way checks. Global rule is implemented in ``select_best``:
    max AUC, then among AUC >= best_auc - eps pick max AUPRC.
    """
    if current is None:
        return True
    return select_best([current, candidate], eps=eps) is candidate


def select_best(
    candidates: list[dict[str, float]],
    *,
    eps: float = 0.005,
    auc_key: str = "macro_ovr_auc",
    auprc_key: str = "macro_auprc",
) -> dict[str, float] | None:
    """Global selection: max AUC, then max AUPRC among AUC >= best_auc - eps.

    Stable tie-break: higher AUC, then higher AUPRC, then lower index.
    """
    if not candidates:
        return None
    scored: list[tuple[float, float, int, dict[str, float]]] = []
    for i, c in enumerate(candidates):
        scored.append((float(c[auc_key]), float(c[auprc_key]), i, c))
    best_auc = max(a for a, _, _, _ in scored)
    pool = [t for t in scored if t[0] >= best_auc - eps]
    # sort: AUPRC desc, AUC desc, index asc
    pool.sort(key=lambda t: (-t[1], -t[0], t[2]))
    return pool[0][3]
