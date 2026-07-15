"""B2 metrics, decision bars, paired bootstrap vs matched baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_blocks.diagnostics import paired_delta_bootstrap
from training.path_a_watch.metrics import full_report, macro_ovr_auc


def summarize_arm(result: dict[str, Any]) -> dict[str, Any]:
    raw = result["metrics"]["test_raw"]
    cal = result["metrics"]["test_cal"]
    return {
        "arm": result.get("arm"),
        "pool": result["pool"],
        "family": result["family"],
        "n_train": result["n_train"],
        "n_val": result["n_val"],
        "n_test": result["n_test"],
        "n_features": len(result["feature_cols"]),
        "val_macro_ovr_auc": result["val_macro_ovr_auc"],
        "val_macro_auprc": result["val_macro_auprc"],
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_binary_auprc": float(raw["binary_auprc"]),
        "test_multiclass_brier": float(raw["multiclass_brier"]),
        "test_binary_brier": float(raw["binary_brier"]),
        "test_cal_macro_ovr_auc": float(cal["macro_ovr_auc"]),
        "test_cal_multiclass_brier": float(cal["multiclass_brier"]),
        "per_class_ovr_auc": raw["per_class_ovr_auc"],
        "deployable": result.get("deployable"),
        "oracle": result.get("oracle"),
    }


def compare_arms(
    new: dict[str, Any],
    base: dict[str, Any],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap on shared test pids (inner join by person_id)."""
    pid_n = np.asarray(new["pid_test"])
    pid_b = np.asarray(base["pid_test"])
    # align
    map_b = {int(p): i for i, p in enumerate(pid_b)}
    idx_n = []
    idx_b = []
    for i, p in enumerate(pid_n):
        j = map_b.get(int(p))
        if j is not None:
            idx_n.append(i)
            idx_b.append(j)
    if not idx_n:
        raise AssertionError("no shared test pids for paired compare")
    idx_n = np.asarray(idx_n)
    idx_b = np.asarray(idx_b)
    y = np.asarray(new["y_test"])[idx_n]
    # verify labels match
    yb = np.asarray(base["y_test"])[idx_b]
    if not np.array_equal(y, yb):
        raise AssertionError("label mismatch on shared test pids")
    p_n = np.asarray(new["proba_test"])[idx_n]
    p_b = np.asarray(base["proba_test"])[idx_b]
    boot = paired_delta_bootstrap(
        y, p_n, p_b, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return {
        "new_arm": new.get("arm"),
        "base_arm": base.get("arm"),
        "n_shared_test": int(len(y)),
        "point_new_auc": float(macro_ovr_auc(y, p_n)),
        "point_base_auc": float(macro_ovr_auc(y, p_b)),
        **boot,
    }


def apply_decision_bars(
    arm_summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    stage1_val: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    bars = cfg["decision_bars"]
    freeze = cfg["frozen_c1_reference"]
    out: dict[str, Any] = {"stage1_gate_pass": bool(stage1_val.get("gate_pass", False))}

    def _ablation(key: str, name: str) -> dict[str, Any]:
        c = comparisons.get(key)
        if c is None:
            return {"status": "missing"}
        d = c["delta_macro_ovr_auc"]
        return {
            "status": "pass" if d.get("ci_lower_gt_zero") else "fail",
            "point_delta": d.get("point"),
            "ci": [d.get("lo"), d.get("hi")],
            "ci_lower_gt_zero": d.get("ci_lower_gt_zero"),
            "soft_note": bool(
                d.get("point") is not None
                and d["point"] > 0.005
                and not d.get("ci_lower_gt_zero")
            ),
            "name": name,
        }

    out["T1_vs_D1"] = _ablation("T1_vs_D1", "predicted CGM helps on C1 (full core)")
    out["T0_vs_D0"] = _ablation("T0_vs_D0", "predicted CGM helps on W0 (full core)")

    # oracle headroom O1 vs D1a
    c = comparisons.get("O1_vs_D1a")
    if c is not None:
        pt = float(c["delta_macro_ovr_auc"]["point"])
        out["oracle_headroom_O1_vs_D1a"] = {
            "point_delta": pt,
            "headroom_pass": pt >= float(bars["oracle_headroom_pass"]),
            "kill_pivot": pt < float(bars["oracle_kill"]),
            "ci": [
                c["delta_macro_ovr_auc"].get("lo"),
                c["delta_macro_ovr_auc"].get("hi"),
            ],
        }

    # user ambition bar
    d1 = arm_summaries.get("D1")
    t1 = arm_summaries.get("T1")
    if d1 and t1:
        d1_auc = float(d1["test_macro_ovr_auc"])
        t1_auc = float(t1["test_macro_ovr_auc"])
        t1_bin = float(t1["test_binary_auc"])
        freeze_auc = float(freeze["test_macro_ovr_auc"])
        freeze_bin = float(freeze["test_binary_auc"])
        drift = d1_auc - freeze_auc
        use_internal = d1_auc < freeze_auc - float(bars["d1_freeze_drift_tol"])
        out["user_ambition"] = {
            "T1_test_auc": t1_auc,
            "T1_test_binary": t1_bin,
            "D1_test_auc": d1_auc,
            "frozen_c1_auc": freeze_auc,
            "frozen_c1_binary": freeze_bin,
            "D1_minus_freeze_auc": drift,
            "primary_compare": "T1_vs_D1" if use_internal else "T1_vs_D1_and_freeze_anchor",
            "fallback_triggered": use_internal,
            "beats_D1": t1_auc > d1_auc,
            "beats_frozen_c1_auc": t1_auc > freeze_auc,
            "beats_frozen_c1_binary": t1_bin > freeze_bin,
            "ambition_pass_strict": bool(
                t1_auc > freeze_auc and t1_bin > freeze_bin and not use_internal
            ),
            "note": (
                "D1 drifted below freeze by > tol; fair bar is T1 vs D1"
                if use_internal
                else "D1 within freeze tol; report T1 vs D1 and vs frozen C1"
            ),
        }
    return out
