"""Decision bar + paired Δ vs C1 for ensemble raise arms."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_blocks.diagnostics import bootstrap_ci, paired_delta_bootstrap
from training.path_a_watch.metrics import full_report, macro_ovr_auc


def arm_report(y: np.ndarray, proba: np.ndarray, *, tag: str) -> dict[str, Any]:
    return full_report(y, proba, tag=tag)


def evaluate_arm_vs_parent(
    y_test: np.ndarray,
    proba_arm: np.ndarray,
    proba_parent: np.ndarray,
    parent_auc: float,
    parent_bin: float,
    *,
    tag: str,
    bootstrap_n: int,
    bootstrap_seed: int,
    paired_bootstrap_seed: int,
    min_n_boot_ok: int,
    delta_auc_bar: float,
) -> dict[str, Any]:
    tr = full_report(y_test, proba_arm, tag=tag)
    d_auc = float(tr["macro_ovr_auc"] - parent_auc)
    d_bin = float(tr["binary_auc"] - parent_bin)

    boot = bootstrap_ci(
        y_test,
        proba_arm,
        n_boot=bootstrap_n,
        alpha=0.05,
        seed=bootstrap_seed,
    )
    boot_d = paired_delta_bootstrap(
        y_test,
        proba_arm,
        proba_parent,
        n_boot=bootstrap_n,
        alpha=0.05,
        seed=paired_bootstrap_seed,
    )

    n_ok_abs = int(boot.get("macro_ovr_auc", {}).get("n_boot_ok", boot.get("n_boot_ok", 0)) or 0)
    # bootstrap_ci structure: check both shapes
    if "macro_ovr_auc" in boot and isinstance(boot["macro_ovr_auc"], dict):
        n_ok_abs = int(boot["macro_ovr_auc"].get("n_boot_ok", 0))
    n_ok_d = int(boot_d["delta_macro_ovr_auc"].get("n_boot_ok", 0))
    if n_ok_d < min_n_boot_ok:
        raise AssertionError(
            f"{tag}: paired bootstrap n_boot_ok={n_ok_d} < min {min_n_boot_ok}"
        )

    c1 = bool(d_auc > delta_auc_bar)
    c2 = bool(boot_d["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))

    return {
        "tag": tag,
        "selected_raw": tr,
        "delta_vs_c1": {
            "delta_macro_ovr_auc": d_auc,
            "delta_binary_auc": d_bin,
            "criterion1_point_delta_gt_0p01": c1,
            "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
            "bootstrap_paired_delta": boot_d,
        },
        "bootstrap_test": boot,
        "n_boot_ok_paired": n_ok_d,
        "n_boot_ok_abs": n_ok_abs,
    }


def c3_bag_cat(
    seed_val_aucs: list[float],
    s_cat_val_auc: float,
    bag_val_auc: float,
    *,
    c1_freeze_val_auc: float = 0.7438628978278522,
    n_seeds_ge: int = 4,
    val_floor_slack: float = 0.01,
    bag_vs_single_slack: float = 0.002,
) -> dict[str, Any]:
    """Primary A c3: multi-seed consistency on val."""
    n_ok = sum(1 for a in seed_val_aucs if a >= c1_freeze_val_auc - val_floor_slack)
    mean_seed = float(np.mean(seed_val_aucs)) if seed_val_aucs else float("nan")
    cond_seeds = n_ok >= n_seeds_ge
    cond_bag = mean_seed >= s_cat_val_auc - bag_vs_single_slack
    return {
        "n_seeds": len(seed_val_aucs),
        "n_seeds_ge_floor": int(n_ok),
        "required_n_seeds_ge": n_seeds_ge,
        "c1_freeze_val_auc": c1_freeze_val_auc,
        "val_floor": c1_freeze_val_auc - val_floor_slack,
        "mean_seed_val_auc": mean_seed,
        "s_cat_val_auc": s_cat_val_auc,
        "bag_val_auc": bag_val_auc,
        "cond_seeds": cond_seeds,
        "cond_bag_vs_single": cond_bag,
        "criterion3_pass": bool(cond_seeds and cond_bag),
    }


def c3_e_arith(
    y_test: np.ndarray,
    p_arith: np.ndarray,
    p_best_bag: np.ndarray,
    *,
    best_bag_name: str,
    bootstrap_n: int,
    paired_bootstrap_seed: int,
    min_n_boot_ok: int,
    point_margin: float = 0.005,
) -> dict[str, Any]:
    """Primary B c3: blend superiority vs best single-family bag on test."""
    auc_e = float(macro_ovr_auc(y_test, p_arith))
    auc_b = float(macro_ovr_auc(y_test, p_best_bag))
    point_ok = bool(auc_e >= auc_b + point_margin)

    boot_d = paired_delta_bootstrap(
        y_test,
        p_arith,
        p_best_bag,
        n_boot=bootstrap_n,
        alpha=0.05,
        seed=paired_bootstrap_seed + 17,
    )
    n_ok = int(boot_d["delta_macro_ovr_auc"].get("n_boot_ok", 0))
    if n_ok < min_n_boot_ok:
        raise AssertionError(f"E_arith c3 bootstrap n_boot_ok={n_ok} < {min_n_boot_ok}")
    ci_ok = bool(boot_d["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))

    return {
        "best_bag": best_bag_name,
        "test_auc_e_arith": auc_e,
        "test_auc_best_bag": auc_b,
        "point_margin": point_margin,
        "point_superiority": point_ok,
        "paired_bootstrap_vs_best_bag": boot_d,
        "ci_lower_gt_zero": ci_ok,
        "criterion3_pass": bool(ci_ok or point_ok),
    }


def near_bar_trigger(d_auc: float, c1: bool, c2: bool) -> bool:
    """Pre-committed S=10 follow-up trigger."""
    if c1 and not c2:
        return True
    if (not c1) and (0.0 < d_auc <= 0.01):
        return True
    return False
