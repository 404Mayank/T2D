"""B1 metrics: re-export Path A helpers + bootstrap + glu MSE."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_watch.metrics import (  # noqa: F401
    binary_from_multiclass,
    full_report,
    macro_ovr_auc,
    multiclass_brier,
    per_class_ovr_auc,
)


def glu_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """pred/target [N,T,C]; mask [N,T]."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"glu_mse": float("nan"), "glu_mae": float("nan"), "n_glu": 0}
    if pred.ndim == 3:
        # advanced index: [n_valid, C]
        sel_p = pred[mask]
        sel_t = target[mask]
    else:
        sel_p = pred[mask]
        sel_t = target[mask]
    return {
        "glu_mse": float(np.mean((sel_p - sel_t) ** 2)),
        "glu_mae": float(np.mean(np.abs(sel_p - sel_t))),
        "n_glu": int(mask.sum()),
    }


def paired_bootstrap_delta_auc(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired person bootstrap of macro-OVR AUC(b) - AUC(a)."""
    y = np.asarray(y).astype(int).ravel()
    pa = np.asarray(proba_a, dtype=float)
    pb = np.asarray(proba_b, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    base_a = macro_ovr_auc(y, pa)
    base_b = macro_ovr_auc(y, pb)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            da = macro_ovr_auc(y[idx], pa[idx])
            db = macro_ovr_auc(y[idx], pb[idx])
            deltas[i] = db - da
        except ValueError:
            deltas[i] = np.nan
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "auc_a": float(base_a),
        "auc_b": float(base_b),
        "delta": float(base_b - base_a),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot_ok": int(len(deltas)),
        "ci_lo_gt_0": bool(lo > 0) if np.isfinite(lo) else False,
    }
