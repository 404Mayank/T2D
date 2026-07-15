"""Metrics, decision bar, paired Δ vs C1."""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.metrics import roc_auc_score

from training.path_a_blocks.diagnostics import paired_delta_bootstrap
from training.path_a_watch.metrics import full_report, macro_ovr_auc


def assert_proba_contract(proba: np.ndarray, n_classes: int = 4) -> None:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise AssertionError(f"proba shape {p.shape} != (*, {n_classes})")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise AssertionError("proba rows must sum to 1")
    if np.any(p < -1e-6):
        raise AssertionError("negative proba")


def decision_bar(
    *,
    point_delta_auc: float,
    boot: dict[str, Any],
    perm_stable: bool,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    thr = float(cfg["decision_bar"]["delta_macro_ovr_auc_gt"])
    c1 = bool(point_delta_auc > thr)
    d = boot["delta_macro_ovr_auc"]
    c2 = bool(d.get("ci_lower_gt_zero"))
    c3 = bool(perm_stable)
    return {
        "criterion1_point_delta_gt": c1,
        "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
        "criterion3_perm_stable": c3,
        "decision_bar_pass": bool(c1 and c2 and c3),
        "point_delta_macro_ovr_auc": float(point_delta_auc),
        "threshold": thr,
    }


def soft_class2(
    proba_new: np.ndarray,
    y: np.ndarray,
    parent_class2: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    from training.path_a_watch.metrics import per_class_ovr_auc

    pc = per_class_ovr_auc(y, proba_new)
    c2 = float(pc[2])
    delta = c2 - float(parent_class2)
    thr = float(cfg["decision_bar"]["soft_class2_delta"])
    return {
        "class2_ovr_auc": c2,
        "parent_class2_ovr_auc": float(parent_class2),
        "delta": delta,
        "soft_win": bool(delta >= thr),
        "threshold": thr,
        "per_class_ovr_auc": {str(k): float(v) for k, v in pc.items()},
    }


def compare_to_parent(
    y_test: np.ndarray,
    proba_new: np.ndarray,
    proba_parent: np.ndarray,
    *,
    parent_class2: float,
    cfg: dict[str, Any],
    perm_stable: bool,
    n_boot: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    assert_proba_contract(proba_new)
    assert_proba_contract(proba_parent)
    y = np.asarray(y_test).astype(int)
    raw = full_report(y, proba_new, tag="test_raw")
    parent_auc = float(macro_ovr_auc(y, proba_parent))
    point_d = float(raw["macro_ovr_auc"] - parent_auc)
    n_boot = int(n_boot if n_boot is not None else cfg["run"]["bootstrap_n"])
    alpha = 1.0 - float(cfg["run"]["bootstrap_ci"])
    boot = paired_delta_bootstrap(
        y,
        proba_new,
        proba_parent,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    bar = decision_bar(
        point_delta_auc=point_d,
        boot=boot,
        perm_stable=perm_stable,
        cfg=cfg,
    )
    soft = soft_class2(proba_new, y, parent_class2, cfg)
    return {
        "selected_raw": raw,
        "parent_test_macro_ovr_auc_recomputed": parent_auc,
        "delta_vs_c1": {
            "point_delta_macro_ovr_auc": point_d,
            "point_delta_binary_auc": float(raw["binary_auc"])
            - float(
                roc_auc_score((y > 0).astype(int), 1.0 - proba_parent[:, 0])
            ),
            "bootstrap_paired_delta": boot,
            **bar,
        },
        "soft_class2": soft,
    }


def arm_vs_arm_delta(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired Δ AUC: A − B (e.g. CORN − CE)."""
    boot = paired_delta_bootstrap(
        y, proba_a, proba_b, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return {
        "point_delta_macro_ovr_auc": float(
            macro_ovr_auc(y, proba_a) - macro_ovr_auc(y, proba_b)
        ),
        "bootstrap_paired_delta": boot,
    }
