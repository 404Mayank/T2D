"""GBM student: hard D1 + LightGBM soft-row expansion (G0 / Gα)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler

from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.hpo import _lgbm_space, pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report, macro_auprc, macro_ovr_auc
from training.path_a_watch.models import (
    best_iteration,
    make_lgbm,
    predict_proba,
    resolve_lgbm_device,
)
from training.path_b.b3.data import (
    assert_student_features,
    expand_soft_rows,
    hard_class_weights,
    split_xy,
)
from training.path_b.b3.teacher import train_hard_gbm

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_d1(
    df: pd.DataFrame,
    c1_cols: list[str],
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> dict[str, Any]:
    assert_student_features(c1_cols, cfg)
    splits = split_xy(df, c1_cols, pool="core")
    log(f"D1 n_train={splits['n_train']} n_val={splits['n_val']} n_test={splits['n_test']}")
    pack = train_hard_gbm(splits, cfg, n_trials=n_trials, log=log, families="both")
    pack["arm"] = "D1"
    pack["deployable"] = True
    pack["oracle"] = False
    pack["alpha"] = 0.0
    # also store pure LGBM pack for G0 pin (even if CatBoost wins D1)
    return pack


def _fit_lgbm_weighted(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    n_estimators = int(cfg["run"]["n_estimators_max"])
    es_rounds = int(cfg["run"]["es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    seed = int(cfg["run"]["seed"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4

    # class_weight=None — imbalance handled via sample_weight
    model = make_lgbm(
        params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        n_estimators=n_estimators,
        class_weight=None,
    )
    callbacks = [
        lgb.early_stopping(es_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )
    proba_val = predict_proba(model, X_val)
    return {
        "family": "lightgbm",
        "params": dict(params),
        "best_iteration": best_iteration(model),
        "val_macro_ovr_auc": float(macro_ovr_auc(y_val, proba_val)),
        "val_macro_auprc": float(macro_auprc(y_val, proba_val)),
        "model": model,
        "device": device,
    }


def tune_lgbm_soft(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> dict[str, Any]:
    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    n_trials = n_trials if n_trials is not None else int(cfg["run"]["n_trials"])
    seed = int(cfg["run"]["seed"])
    eps = float(cfg["run"]["auc_tie_eps"])

    packs: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = _lgbm_space(trial, cfg)
        try:
            pack = _fit_lgbm_weighted(
                X_train, y_train, w_train, X_val, y_val, params, cfg, device=device
            )
            packs.append(pack)
            return float(pack["val_macro_ovr_auc"])
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="lgbm_soft",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if not packs:
        raise RuntimeError("lgbm soft HPO: no successful trials")
    # pick by auc then auprc
    score_rows = [
        {
            "macro_ovr_auc": float(p["val_macro_ovr_auc"]),
            "macro_auprc": float(p["val_macro_auprc"]),
            "_idx": i,
        }
        for i, p in enumerate(packs)
    ]
    from training.path_a_watch.metrics import select_best

    chosen = select_best(score_rows, eps=eps)
    assert chosen is not None
    best = packs[int(chosen["_idx"])]
    log(
        f"  soft-LGBM val_auc={best['val_macro_ovr_auc']:.4f} "
        f"auprc={best['val_macro_auprc']:.4f} n_trials={n_trials}"
    )
    return best


def train_g_alpha(
    df: pd.DataFrame,
    c1_cols: list[str],
    soft_by_pid: dict[int, np.ndarray],
    cfg: dict[str, Any],
    *,
    alpha: float,
    temperature: float | None = None,
    pinned_params: dict[str, Any] | None = None,
    n_trials: int | None = None,
    log=lambda msg: None,
    arm_name: str | None = None,
) -> dict[str, Any]:
    """Train G0 (α=0, pinned) or Gα (soft expansion + HPO unless pinned)."""
    assert_student_features(c1_cols, cfg)
    temp = float(temperature if temperature is not None else cfg["run"]["temperature"])
    splits = split_xy(df, c1_cols, pool="core")

    aux_train = df[
        (df["recommended_split"] == "train") & df["aux_eligible"].astype(bool)
    ]
    aux_pids = set(int(p) for p in aux_train["person_id"].tolist())

    ytr = splits["y_train"]
    cw = hard_class_weights(ytr, n_classes=4)

    X_exp, y_exp, w_exp, exp_diag = expand_soft_rows(
        splits["X_train"],
        ytr,
        splits["pid_train"],
        soft_by_pid=soft_by_pid,
        aux_pids=aux_pids,
        alpha=float(alpha),
        temperature=temp,
        class_weights=cw,
        eps=float(cfg["data"]["soft_eps"]),
    )
    log(
        f"Gα={alpha} expand: persons={exp_diag['n_persons']} "
        f"rows={exp_diag['n_expanded_rows']} soft_pids={exp_diag['n_soft_persons']} "
        f"alpha0_ok={exp_diag.get('alpha0_n_rows_eq_n_persons')}"
    )

    if float(alpha) == 0.0:
        if not exp_diag["alpha0_n_rows_eq_n_persons"]:
            raise AssertionError("G0 expansion must emit exactly n_train rows")
        if exp_diag["max_abs_mass_minus_1"] > 1e-6:
            raise AssertionError("G0 per-person mass != 1")
        if pinned_params is None:
            raise AssertionError("G0 requires pinned_params from D1 LightGBM")

    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    if pinned_params is not None:
        pack = _fit_lgbm_weighted(
            X_exp,
            y_exp,
            w_exp,
            splits["X_val"],
            splits["y_val"],
            pinned_params,
            cfg,
            device=device,
        )
        log(f"  pinned LGBM val_auc={pack['val_macro_ovr_auc']:.4f}")
    else:
        pack = tune_lgbm_soft(
            X_exp,
            y_exp,
            w_exp,
            splits["X_val"],
            splits["y_val"],
            cfg,
            n_trials=n_trials,
            log=log,
        )

    model = pack["model"]
    Xva, yva = splits["X_val"], splits["y_val"]
    Xte, yte = splits["X_test"], splits["y_test"]
    proba_val = predict_proba(model, Xva)
    proba_test = predict_proba(model, Xte)

    cal = fit_calibrators(
        proba_val,
        yva,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test_cal = cal["primary"].transform(proba_test)

    name = arm_name or (f"G_a={alpha}" if alpha != 0 else "G0")
    return {
        "arm": name,
        "alpha": float(alpha),
        "temperature": temp,
        "family": "lightgbm",
        "params": pack["params"],
        "best_iteration": pack.get("best_iteration"),
        "val_macro_ovr_auc": float(macro_ovr_auc(yva, proba_val)),
        "val_macro_auprc": float(macro_auprc(yva, proba_val)),
        "model": model,
        "proba_val": proba_val,
        "proba_test": proba_test,
        "proba_test_cal": proba_test_cal,
        "calibrator": cal,
        "metrics": {
            "val_raw": full_report(yva, proba_val, tag="val_raw"),
            "test_raw": full_report(yte, proba_test, tag="test_raw"),
            "test_cal": full_report(yte, proba_test_cal, tag="test_cal_primary"),
        },
        "feature_cols": list(c1_cols),
        "pool": "core",
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_test": splits["n_test"],
        "pid_test": splits["pid_test"],
        "y_test": yte,
        "deployable": True,
        "oracle": False,
        "expansion_diag": exp_diag,
        "pinned": pinned_params is not None,
    }
