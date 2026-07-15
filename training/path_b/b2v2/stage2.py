"""Stage-2 multiclass T2D: Path A family HPO + val-select (B2-V2 arms)."""

from __future__ import annotations

from typing import Any

import numpy as np

from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.hpo import pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report
from training.path_a_watch.models import predict_proba, resolve_lgbm_device


def train_stage2(
    splits: dict[str, Any],
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> dict[str, Any]:
    if n_trials is not None:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = int(n_trials)

    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"stage2 HPO n_trials={cfg['run']['n_trials']} lgbm_device={device}")

    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xva, yva = splits["X_val"], splits["y_val"]
    Xte, yte = splits["X_test"], splits["y_test"]

    lgbm_pack = tune_lightgbm(Xtr, ytr, Xva, yva, cfg, device=device)
    log(
        f"  LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f} "
        f"auprc={lgbm_pack['val_macro_auprc']:.4f}"
    )
    cat_pack = tune_catboost(Xtr, ytr, Xva, yva, cfg)
    log(
        f"  Cat  val_auc={cat_pack['val_macro_ovr_auc']:.4f} "
        f"auprc={cat_pack['val_macro_auprc']:.4f}"
    )

    selected = pick_family([lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"  SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    model = selected["model"]
    proba_val = predict_proba(model, Xva)
    proba_test = predict_proba(model, Xte)
    proba_train = predict_proba(model, Xtr)

    cal = fit_calibrators(
        proba_val,
        yva,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test_cal = cal["primary"].transform(proba_test)
    proba_val_cal = cal["primary"].transform(proba_val)

    raw_val = full_report(yva, proba_val, tag="val_raw")
    raw_test = full_report(yte, proba_test, tag="test_raw")
    cal_test = full_report(yte, proba_test_cal, tag="test_cal_primary")
    cal_val = full_report(yva, proba_val_cal, tag="val_cal_primary")

    return {
        "family": selected["family"],
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "source": selected.get("source"),
        "val_macro_ovr_auc": float(selected["val_macro_ovr_auc"]),
        "val_macro_auprc": float(selected["val_macro_auprc"]),
        "model": model,
        "lgbm_pack_meta": {
            k: v for k, v in lgbm_pack.items() if k not in ("model", "val_report")
        },
        "cat_pack_meta": {
            k: v for k, v in cat_pack.items() if k not in ("model", "val_report")
        },
        "proba_train": proba_train,
        "proba_val": proba_val,
        "proba_test": proba_test,
        "proba_test_cal": proba_test_cal,
        "calibrator": cal,
        "metrics": {
            "val_raw": raw_val,
            "test_raw": raw_test,
            "val_cal": cal_val,
            "test_cal": cal_test,
        },
        "feature_cols": list(splits["feature_cols"]),
        "pool": splits["pool"],
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_test": splits["n_test"],
        "pid_test": splits["pid_test"],
        "y_test": yte,
    }


def arm_feature_cols(
    arm: str,
    groups: dict[str, list[str]],
) -> list[str]:
    """Resolve feature list for B2-V2 pre-registered arm id."""
    w0 = groups["w0"]
    c1 = groups["c1"]
    true_cols = groups["true_cols"]
    point = groups["point_cols"]
    var = groups["var_cols"]

    if arm in ("D0", "D0a"):
        return list(w0)
    if arm in ("D1", "D1a"):
        return list(c1)
    if arm == "T0p":
        return list(w0) + list(point)
    if arm == "T1p":
        return list(c1) + list(point)
    if arm == "T0v":
        return list(w0) + list(var)
    if arm == "T1v":
        return list(c1) + list(var)
    if arm == "O0":
        return list(w0) + list(true_cols)
    if arm == "O1":
        return list(c1) + list(true_cols)
    raise ValueError(f"unknown arm {arm}")


def arm_pool(arm: str) -> str:
    if arm in ("D0", "D1", "T0p", "T1p", "T0v", "T1v"):
        return "core"
    if arm in ("D0a", "D1a", "O0", "O1"):
        return "aux"
    raise ValueError(arm)


def is_oracle_arm(arm: str) -> bool:
    return arm in ("O0", "O1")


def is_deployable_arm(arm: str) -> bool:
    return arm in ("D0", "D1", "T0p", "T1p", "T0v", "T1v", "D0a", "D1a")
