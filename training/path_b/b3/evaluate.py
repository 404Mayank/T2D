"""B3 metrics summaries, paired bootstrap, decision bars."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_blocks.diagnostics import paired_delta_bootstrap
from training.path_a_watch.metrics import macro_ovr_auc
from training.path_b.b2.evaluate import compare_arms, summarize_arm


def apply_decision_bars(
    arm_summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    teacher_pack: dict[str, Any],
    cfg: dict[str, Any],
    *,
    g0_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bars = cfg["decision_bars"]
    freeze = cfg["frozen_c1_reference"]
    out: dict[str, Any] = {}

    oof = teacher_pack.get("oof", {})
    out["oof_teacher"] = {
        "mean_val_auc": oof.get("mean_val_auc"),
        "std_val_auc": oof.get("std_val_auc"),
        "d1a_val_auc": oof.get("d1a_val_auc"),
        "gate_pass": oof.get("gate_pass"),
        "gate_margin": oof.get("gate_margin"),
    }

    # teacher headroom Tch vs D1a
    c = comparisons.get("Tch_vs_D1a")
    if c is not None:
        pt = float(c["delta_macro_ovr_auc"]["point"])
        out["teacher_headroom_Tch_vs_D1a"] = {
            "point_delta": pt,
            "headroom_pass": pt >= float(bars["teacher_headroom_pass"]),
            "kill_pivot": pt < float(bars["teacher_kill"]),
            "ci": [
                c["delta_macro_ovr_auc"].get("lo"),
                c["delta_macro_ovr_auc"].get("hi"),
            ],
            "ci_lower_gt_zero": c["delta_macro_ovr_auc"].get("ci_lower_gt_zero"),
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
        }

    out["G_a0.3_vs_D1"] = _ablation(
        "G_a0.3_vs_D1", "user ambition: soft-label LGBM KD vs D1"
    )
    out["N_a0.3_vs_N0"] = _ablation(
        "N_a0.3_vs_N0", "KD science: Hinton MLP vs hard MLP"
    )

    d1 = arm_summaries.get("D1")
    g = arm_summaries.get("G_a=0.3") or arm_summaries.get("G_a0.3")
    # normalize key lookup
    for k, v in arm_summaries.items():
        if k.replace(" ", "") in ("G_a=0.3", "G_a0.3", "G_α=0.3") or (
            str(v.get("alpha")) == "0.3" and str(v.get("family")) == "lightgbm" and k.startswith("G")
        ):
            g = v
            break

    if d1 and g:
        d1_auc = float(d1["test_macro_ovr_auc"])
        g_auc = float(g["test_macro_ovr_auc"])
        g_bin = float(g["test_binary_auc"])
        freeze_auc = float(freeze["test_macro_ovr_auc"])
        freeze_bin = float(freeze["test_binary_auc"])
        drift = d1_auc - freeze_auc
        use_internal = d1_auc < freeze_auc - float(bars["d1_freeze_drift_tol"])
        boot = comparisons.get("G_a0.3_vs_D1", {})
        dboot = boot.get("delta_macro_ovr_auc", {}) if boot else {}
        out["user_ambition"] = {
            "decision_arm": "G_a=0.3",
            "G_test_auc": g_auc,
            "G_test_binary": g_bin,
            "D1_test_auc": d1_auc,
            "frozen_c1_auc": freeze_auc,
            "frozen_c1_binary": freeze_bin,
            "D1_minus_freeze_auc": drift,
            "fallback_triggered": use_internal,
            "beats_D1_point": g_auc > d1_auc,
            "beats_frozen_c1_auc": g_auc > freeze_auc,
            "beats_frozen_c1_binary": g_bin > freeze_bin,
            "ci_lower_gt_zero": dboot.get("ci_lower_gt_zero"),
            "ambition_pass": bool(
                dboot.get("ci_lower_gt_zero") and g_auc > d1_auc
            ),
            "note": (
                "D1 drifted below freeze by > tol; fair bar is G vs D1"
                if use_internal
                else "decision arm is pre-registered G_a=0.3 only (not max-over-α)"
            ),
        }

    if g0_protocol is not None:
        out["g0_protocol"] = g0_protocol

    out["limitations"] = (
        "Gα vs G0 does not separate teacher dark knowledge from generic "
        "soft-label regularisation; shuffled-soft control only if headline raise."
    )
    return out


__all__ = [
    "summarize_arm",
    "compare_arms",
    "apply_decision_bars",
    "macro_ovr_auc",
    "paired_delta_bootstrap",
]
