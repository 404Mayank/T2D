"""B2-V2 metrics, decision bars, paired bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_blocks.diagnostics import paired_delta_bootstrap
from training.path_a_watch.metrics import macro_ovr_auc


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
    pid_n = np.asarray(new["pid_test"])
    pid_b = np.asarray(base["pid_test"])
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
    freeze_w0 = cfg.get("frozen_w0_reference", {})
    primary_pack = str(stage1_val.get("primary_pack", "var"))
    # coverage demotion: P_point becomes primary ablation (PLAN §3.10)
    primary_key = "T1v_vs_D1" if primary_pack == "var" else "T1p_vs_D1"
    primary_arm = "T1v" if primary_pack == "var" else "T1p"

    out: dict[str, Any] = {
        "stage1_gate_pass": bool(stage1_val.get("gate_pass", False)),
        "stage1_coverage_pass": bool(stage1_val.get("coverage_pass", False)),
        "stage1_early_kill": bool(stage1_val.get("early_kill", False)),
        "stage1_mean_r2": stage1_val.get("mean_r2"),
        "primary_pack": primary_pack,
        "primary_ablation_key": primary_key,
        "primary_arm": primary_arm,
    }

    def _ablation(key: str, name: str) -> dict[str, Any]:
        c = comparisons.get(key)
        if c is None:
            return {"status": "missing", "name": name}
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
            "is_primary": key == primary_key,
        }

    out["T1v_vs_D1"] = _ablation("T1v_vs_D1", "variance pack on C1")
    out["T0v_vs_D0"] = _ablation("T0v_vs_D0", "variance pack on W0")
    out["T1v_vs_T1p"] = _ablation("T1v_vs_T1p", "variance vs point (same daily S1)")
    out["T1p_vs_D1"] = _ablation("T1p_vs_D1", "daily point alone on C1")
    out["primary_ablation"] = out.get(primary_key, {"status": "missing"})

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

    d1 = arm_summaries.get("D1")
    t1v = arm_summaries.get("T1v")
    d0 = arm_summaries.get("D0")
    if d1:
        d1_auc = float(d1["test_macro_ovr_auc"])
        freeze_auc = float(freeze["test_macro_ovr_auc"])
        drift = d1_auc - freeze_auc
        out["parity_D1"] = {
            "D1_test_auc": d1_auc,
            "frozen_c1_auc": freeze_auc,
            "delta": drift,
            "within_1e5": abs(drift) < 1e-5,
            "within_tol": abs(drift) <= float(bars["d1_freeze_drift_tol"]),
        }
    if d0 and freeze_w0:
        d0_auc = float(d0["test_macro_ovr_auc"])
        w0_auc = float(freeze_w0["test_macro_ovr_auc"])
        out["parity_D0"] = {
            "D0_test_auc": d0_auc,
            "frozen_w0_auc": w0_auc,
            "delta": d0_auc - w0_auc,
            "within_1e5": abs(d0_auc - w0_auc) < 1e-5,
        }

    t_primary = arm_summaries.get(primary_arm)
    if d1 and t_primary:
        d1_auc = float(d1["test_macro_ovr_auc"])
        t1_auc = float(t_primary["test_macro_ovr_auc"])
        t1_bin = float(t_primary["test_binary_auc"])
        freeze_auc = float(freeze["test_macro_ovr_auc"])
        freeze_bin = float(freeze["test_binary_auc"])
        drift = d1_auc - freeze_auc
        use_internal = d1_auc < freeze_auc - float(bars["d1_freeze_drift_tol"])
        out["user_ambition"] = {
            "primary_arm": primary_arm,
            "primary_pack": primary_pack,
            "primary_test_auc": t1_auc,
            "primary_test_binary": t1_bin,
            "T1v_test_auc": float(t1v["test_macro_ovr_auc"]) if t1v else None,
            "T1v_test_binary": float(t1v["test_binary_auc"]) if t1v else None,
            "D1_test_auc": d1_auc,
            "frozen_c1_auc": freeze_auc,
            "frozen_c1_binary": freeze_bin,
            "D1_minus_freeze_auc": drift,
            "primary_compare": primary_key,
            "fallback_triggered": use_internal,
            "beats_D1": t1_auc > d1_auc,
            "beats_frozen_c1_auc": t1_auc > freeze_auc,
            "beats_frozen_c1_binary": t1_bin > freeze_bin,
            "ambition_pass_strict": bool(
                t1_auc > freeze_auc and t1_bin > freeze_bin and not use_internal
            ),
            "note": (
                "D1 drifted below freeze by > tol; fair bar is primary vs D1"
                if use_internal
                else f"D1 within freeze tol; primary_pack={primary_pack}"
            ),
        }
    return out
