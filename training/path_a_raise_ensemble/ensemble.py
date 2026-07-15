"""Cross-family probability blends (arith / geom)."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_watch.metrics import macro_ovr_auc


def arith_mean(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    out = 0.5 * (a + b)
    # numerical hygiene (should already sum ~1)
    s = out.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return out / s


def geom_mean(p_a: np.ndarray, p_b: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    a = np.clip(np.asarray(p_a, dtype=float), eps, None)
    b = np.clip(np.asarray(p_b, dtype=float), eps, None)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    g = np.sqrt(a * b)
    s = g.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return g / s


def blend_reports(
    y_val: np.ndarray,
    y_test: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    *,
    tag: str,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "val_macro_ovr_auc": float(macro_ovr_auc(y_val, p_val)),
        "test_macro_ovr_auc": float(macro_ovr_auc(y_test, p_test)),
    }
