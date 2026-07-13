"""Optuna HPO for LightGBM and CatBoost (val-ranked, global selection)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .metrics import (
    binary_report,
    full_report,
    macro_auprc,
    macro_ovr_auc,
    select_best,
)
from .models import (
    best_iteration,
    fit_lgbm,
    fit_lgbm_binary,
    make_lgbm,
    make_lgbm_binary,
    predict_proba,
    predict_proba_positive,
    try_catboost_binary,
    try_catboost_boosting,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _lgbm_space(trial: optuna.Trial, cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["hpo"]["lightgbm"]
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", s["learning_rate"][0], s["learning_rate"][1], log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", s["num_leaves"][0], s["num_leaves"][1]),
        "max_depth": trial.suggest_int("max_depth", s["max_depth"][0], s["max_depth"][1]),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", s["min_child_samples"][0], s["min_child_samples"][1]
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain", s["min_split_gain"][0], s["min_split_gain"][1]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", s["colsample_bytree"][0], s["colsample_bytree"][1]
        ),
        "subsample": trial.suggest_float("subsample", s["subsample"][0], s["subsample"][1]),
        "subsample_freq": trial.suggest_int(
            "subsample_freq", s["subsample_freq"][0], s["subsample_freq"][1]
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", s["reg_alpha"][0], s["reg_alpha"][1], log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", s["reg_lambda"][0], s["reg_lambda"][1], log=True
        ),
    }


def _cat_space(trial: optuna.Trial, cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["hpo"]["catboost"]
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", s["learning_rate"][0], s["learning_rate"][1], log=True
        ),
        "depth": trial.suggest_int("depth", s["depth"][0], s["depth"][1]),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", s["l2_leaf_reg"][0], s["l2_leaf_reg"][1], log=True
        ),
        "random_strength": trial.suggest_float(
            "random_strength", s["random_strength"][0], s["random_strength"][1]
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", s["bagging_temperature"][0], s["bagging_temperature"][1]
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", s["min_data_in_leaf"][0], s["min_data_in_leaf"][1]
        ),
        "border_count": trial.suggest_int(
            "border_count", s["border_count"][0], s["border_count"][1]
        ),
    }


def default_lgbm_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "min_split_gain": 4.0,
        "colsample_bytree": 0.35,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }


def default_cat_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "bagging_temperature": 0.5,
        "min_data_in_leaf": 20,
        "border_count": 64,
    }


def _finalize_pack(
    packs: list[dict[str, Any]],
    *,
    eps: float,
    y_val: np.ndarray,
    X_val: pd.DataFrame,
    tag: str,
) -> dict[str, Any]:
    if not packs:
        raise RuntimeError(f"{tag}: no successful trials/baselines")
    score_rows = [
        {
            "macro_ovr_auc": float(p["val_macro_ovr_auc"]),
            "macro_auprc": float(p["val_macro_auprc"]),
            "_idx": i,
        }
        for i, p in enumerate(packs)
    ]
    chosen = select_best(score_rows, eps=eps)
    assert chosen is not None
    best = packs[int(chosen["_idx"])]
    best = dict(best)
    best["n_candidates"] = len(packs)
    best["selection"] = "global_max_auc_then_auprc_within_eps"
    best["val_report"] = full_report(
        y_val, predict_proba(best["model"], X_val), tag=f"{tag}_val"
    )
    # Consistent with selected pack (not pure Optuna best_value)
    best["selected_val_macro_ovr_auc"] = best["val_macro_ovr_auc"]
    best["selected_val_macro_auprc"] = best["val_macro_auprc"]
    return best


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int | None = None,
    study_name: str = "lgbm",
) -> dict[str, Any]:
    run = cfg["run"]
    n_trials = n_trials if n_trials is not None else int(run["n_trials"])
    seed = int(run["seed"])
    es_rounds = int(run["es_rounds"])
    n_estimators = int(run["n_estimators_max"])
    n_jobs = int(run["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4  # avoid oversubscribe with OpenCL
    eps = float(run["auc_tie_eps"])
    cw = cfg["class_weights"]["lightgbm"]

    trial_packs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _fit_one(params: dict[str, Any], dev: str) -> dict[str, Any]:
        model = make_lgbm(
            params,
            seed=seed,
            n_jobs=n_jobs,
            device=dev,
            n_estimators=n_estimators,
            class_weight=cw,
        )
        fit_lgbm(model, X_train, y_train, X_val, y_val, es_rounds=es_rounds)
        proba = predict_proba(model, X_val)
        return {
            "family": "lightgbm",
            "params": params,
            "best_iteration": best_iteration(model),
            "val_macro_ovr_auc": macro_ovr_auc(y_val, proba),
            "val_macro_auprc": macro_auprc(y_val, proba),
            "model": model,
            "device": dev,
        }

    def objective(trial: optuna.Trial) -> float:
        params = _lgbm_space(trial, cfg)
        dev = device
        try:
            pack = _fit_one(params, dev)
        except Exception as e1:
            if device == "gpu":
                try:
                    pack = _fit_one(params, "cpu")
                    pack["device_fallback"] = f"gpu_failed:{type(e1).__name__}"
                    dev = "cpu"
                except Exception as e2:
                    failures.append(
                        {"trial": trial.number, "error": f"{e1} | cpu: {e2}"}
                    )
                    trial.set_user_attr("failed", True)
                    return 0.0
            else:
                failures.append({"trial": trial.number, "error": str(e1)})
                trial.set_user_attr("failed", True)
                return 0.0

        pack["source"] = "optuna"
        pack["trial_number"] = trial.number
        trial.set_user_attr("macro_auprc", pack["val_macro_auprc"])
        trial.set_user_attr("best_iteration", pack["best_iteration"])
        trial.set_user_attr("device", pack["device"])
        trial_packs.append(pack)
        return float(pack["val_macro_ovr_auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # baseline always competes
    try:
        base = _fit_one(default_lgbm_params(), device)
        base["source"] = "baseline"
        base["trial_number"] = -1
        trial_packs.append(base)
    except Exception as e:
        if device == "gpu":
            base = _fit_one(default_lgbm_params(), "cpu")
            base["source"] = "baseline"
            base["trial_number"] = -1
            base["device_fallback"] = str(e)
            trial_packs.append(base)
        else:
            failures.append({"trial": -1, "error": f"baseline: {e}"})

    best = _finalize_pack(trial_packs, eps=eps, y_val=y_val, X_val=X_val, tag="lgbm")
    best["n_trials"] = n_trials
    best["n_failures"] = len(failures)
    best["failures"] = failures[:20]
    best["optuna_best_auc_only"] = (
        float(study.best_value) if study.best_trial is not None else None
    )
    return best


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    study_name: str = "catboost",
) -> dict[str, Any]:
    run = cfg["run"]
    n_trials = n_trials if n_trials is not None else int(run["n_trials"])
    seed = int(run["seed"])
    es_rounds = int(run["es_rounds"])
    n_estimators = int(run["n_estimators_max"])
    eps = float(run["auc_tie_eps"])
    preferred = run.get("catboost_boosting_type", "Ordered")
    fallback = run.get("catboost_boosting_fallback", "Plain")
    task_type = run.get("catboost_task_type", "CPU")
    acw = cfg["class_weights"]["catboost"]

    trial_packs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _fit_one(params: dict[str, Any]) -> dict[str, Any]:
        model, bt = try_catboost_boosting(
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed,
            n_estimators=n_estimators,
            es_rounds=es_rounds,
            preferred=preferred,
            fallback=fallback,
            auto_class_weights=acw,
            task_type=task_type,
        )
        proba = predict_proba(model, X_val)
        return {
            "family": "catboost",
            "params": params,
            "best_iteration": best_iteration(model),
            "val_macro_ovr_auc": macro_ovr_auc(y_val, proba),
            "val_macro_auprc": macro_auprc(y_val, proba),
            "model": model,
            "boosting_type": bt,
            "task_type": task_type,
        }

    def objective(trial: optuna.Trial) -> float:
        params = _cat_space(trial, cfg)
        try:
            pack = _fit_one(params)
        except Exception as e:
            failures.append({"trial": trial.number, "error": str(e)})
            trial.set_user_attr("failed", True)
            return 0.0
        pack["source"] = "optuna"
        pack["trial_number"] = trial.number
        trial.set_user_attr("macro_auprc", pack["val_macro_auprc"])
        trial.set_user_attr("best_iteration", pack["best_iteration"])
        trial.set_user_attr("boosting_type", pack["boosting_type"])
        trial_packs.append(pack)
        return float(pack["val_macro_ovr_auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        base = _fit_one(default_cat_params())
        base["source"] = "baseline"
        base["trial_number"] = -1
        trial_packs.append(base)
    except Exception as e:
        failures.append({"trial": -1, "error": f"baseline: {e}"})

    best = _finalize_pack(trial_packs, eps=eps, y_val=y_val, X_val=X_val, tag="catboost")
    best["n_trials"] = n_trials
    best["n_failures"] = len(failures)
    best["failures"] = failures[:20]
    best["optuna_best_auc_only"] = (
        float(study.best_value) if study.best_trial is not None else None
    )
    return best


def pick_family(
    packs: list[dict[str, Any]],
    *,
    eps: float = 0.005,
) -> dict[str, Any]:
    """Val-select among family packs under global rule."""
    rows = [
        {
            "macro_ovr_auc": float(p["val_macro_ovr_auc"]),
            "macro_auprc": float(p["val_macro_auprc"]),
            "_idx": i,
        }
        for i, p in enumerate(packs)
    ]
    chosen = select_best(rows, eps=eps)
    if chosen is None:
        raise RuntimeError("no packs to select")
    return packs[int(chosen["_idx"])]


def _finalize_pack_binary(
    packs: list[dict[str, Any]],
    *,
    eps: float,
    y_val: np.ndarray,
    X_val: pd.DataFrame,
    tag: str,
) -> dict[str, Any]:
    if not packs:
        raise RuntimeError(f"{tag}: no successful trials/baselines")
    score_rows = [
        {
            "binary_auc": float(p["val_binary_auc"]),
            "binary_auprc": float(p["val_binary_auprc"]),
            "_idx": i,
        }
        for i, p in enumerate(packs)
    ]
    chosen = select_best(
        score_rows, eps=eps, auc_key="binary_auc", auprc_key="binary_auprc"
    )
    assert chosen is not None
    best = packs[int(chosen["_idx"])]
    best = dict(best)
    best["n_candidates"] = len(packs)
    best["selection"] = "global_max_binary_auc_then_auprc_within_eps"
    score = predict_proba_positive(best["model"], X_val)
    best["val_report"] = binary_report(y_val, score, tag=f"{tag}_val")
    best["selected_val_binary_auc"] = best["val_binary_auc"]
    best["selected_val_binary_auprc"] = best["val_binary_auprc"]
    return best


def tune_lightgbm_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int | None = None,
    study_name: str = "lgbm_binary",
) -> dict[str, Any]:
    """Binary LGBM HPO ranked on val binary AUC then AUPRC.

    Uses class_weight='balanced' literal — never multiclass weight dicts.
    """
    run = cfg["run"]
    n_trials = n_trials if n_trials is not None else int(run["n_trials"])
    seed = int(run["seed"])
    es_rounds = int(run["es_rounds"])
    n_estimators = int(run["n_estimators_max"])
    n_jobs = int(run["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4
    eps = float(run["auc_tie_eps"])
    # O4: do NOT read cfg["class_weights"]["lightgbm"] (may be 4-class dict)
    cw: str = "balanced"

    y_tr = np.asarray(y_train).astype(int).ravel()
    y_va = np.asarray(y_val).astype(int).ravel()
    if y_tr.max() > 1:
        y_tr = (y_tr > 0).astype(int)
    if y_va.max() > 1:
        y_va = (y_va > 0).astype(int)

    trial_packs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _fit_one(params: dict[str, Any], dev: str) -> dict[str, Any]:
        model = make_lgbm_binary(
            params,
            seed=seed,
            n_jobs=n_jobs,
            device=dev,
            n_estimators=n_estimators,
            class_weight=cw,
        )
        fit_lgbm_binary(model, X_train, y_tr, X_val, y_va, es_rounds=es_rounds)
        score = predict_proba_positive(model, X_val)
        rep = binary_report(y_va, score, tag="lgbm_bin_val")
        return {
            "family": "lightgbm",
            "params": params,
            "best_iteration": best_iteration(model),
            "val_binary_auc": rep["binary_auc"],
            "val_binary_auprc": rep["binary_auprc"],
            "model": model,
            "device": dev,
            "task": "binary",
        }

    def objective(trial: optuna.Trial) -> float:
        params = _lgbm_space(trial, cfg)
        try:
            pack = _fit_one(params, device)
        except Exception as e1:
            if device == "gpu":
                try:
                    pack = _fit_one(params, "cpu")
                    pack["device_fallback"] = f"gpu_failed:{type(e1).__name__}"
                except Exception as e2:
                    failures.append({"trial": trial.number, "error": f"{e1} | cpu: {e2}"})
                    trial.set_user_attr("failed", True)
                    return 0.0
            else:
                failures.append({"trial": trial.number, "error": str(e1)})
                trial.set_user_attr("failed", True)
                return 0.0
        pack["source"] = "optuna"
        pack["trial_number"] = trial.number
        trial.set_user_attr("binary_auprc", pack["val_binary_auprc"])
        trial.set_user_attr("best_iteration", pack["best_iteration"])
        trial_packs.append(pack)
        return float(pack["val_binary_auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        base = _fit_one(default_lgbm_params(), device)
        base["source"] = "baseline"
        base["trial_number"] = -1
        trial_packs.append(base)
    except Exception as e:
        if device == "gpu":
            base = _fit_one(default_lgbm_params(), "cpu")
            base["source"] = "baseline"
            base["trial_number"] = -1
            base["device_fallback"] = str(e)
            trial_packs.append(base)
        else:
            failures.append({"trial": -1, "error": f"baseline: {e}"})

    best = _finalize_pack_binary(
        trial_packs, eps=eps, y_val=y_va, X_val=X_val, tag="lgbm_binary"
    )
    best["n_trials"] = n_trials
    best["n_failures"] = len(failures)
    best["failures"] = failures[:20]
    best["optuna_best_auc_only"] = (
        float(study.best_value) if study.best_trial is not None else None
    )
    return best


def tune_catboost_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    study_name: str = "catboost_binary",
) -> dict[str, Any]:
    run = cfg["run"]
    n_trials = n_trials if n_trials is not None else int(run["n_trials"])
    seed = int(run["seed"])
    es_rounds = int(run["es_rounds"])
    n_estimators = int(run["n_estimators_max"])
    eps = float(run["auc_tie_eps"])
    preferred = run.get("catboost_boosting_type", "Ordered")
    fallback = run.get("catboost_boosting_fallback", "Plain")
    task_type = run.get("catboost_task_type", "CPU")
    # O4: Balanced literal for binary
    acw = "Balanced"

    y_tr = np.asarray(y_train).astype(int).ravel()
    y_va = np.asarray(y_val).astype(int).ravel()
    if y_tr.max() > 1:
        y_tr = (y_tr > 0).astype(int)
    if y_va.max() > 1:
        y_va = (y_va > 0).astype(int)

    trial_packs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _fit_one(params: dict[str, Any]) -> dict[str, Any]:
        model, bt = try_catboost_binary(
            params,
            X_train,
            y_tr,
            X_val,
            y_va,
            seed=seed,
            n_estimators=n_estimators,
            es_rounds=es_rounds,
            preferred=preferred,
            fallback=fallback,
            auto_class_weights=acw,
            task_type=task_type,
        )
        score = predict_proba_positive(model, X_val)
        rep = binary_report(y_va, score, tag="cat_bin_val")
        return {
            "family": "catboost",
            "params": params,
            "best_iteration": best_iteration(model),
            "val_binary_auc": rep["binary_auc"],
            "val_binary_auprc": rep["binary_auprc"],
            "model": model,
            "boosting_type": bt,
            "task_type": task_type,
            "task": "binary",
        }

    def objective(trial: optuna.Trial) -> float:
        params = _cat_space(trial, cfg)
        try:
            pack = _fit_one(params)
        except Exception as e:
            failures.append({"trial": trial.number, "error": str(e)})
            trial.set_user_attr("failed", True)
            return 0.0
        pack["source"] = "optuna"
        pack["trial_number"] = trial.number
        trial.set_user_attr("binary_auprc", pack["val_binary_auprc"])
        trial.set_user_attr("best_iteration", pack["best_iteration"])
        trial_packs.append(pack)
        return float(pack["val_binary_auc"])

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    try:
        base = _fit_one(default_cat_params())
        base["source"] = "baseline"
        base["trial_number"] = -1
        trial_packs.append(base)
    except Exception as e:
        failures.append({"trial": -1, "error": f"baseline: {e}"})

    best = _finalize_pack_binary(
        trial_packs, eps=eps, y_val=y_va, X_val=X_val, tag="catboost_binary"
    )
    best["n_trials"] = n_trials
    best["n_failures"] = len(failures)
    best["failures"] = failures[:20]
    best["optuna_best_auc_only"] = (
        float(study.best_value) if study.best_trial is not None else None
    )
    return best


def pick_family_binary(
    packs: list[dict[str, Any]],
    *,
    eps: float = 0.005,
) -> dict[str, Any]:
    rows = [
        {
            "binary_auc": float(p["val_binary_auc"]),
            "binary_auprc": float(p["val_binary_auprc"]),
            "_idx": i,
        }
        for i, p in enumerate(packs)
    ]
    chosen = select_best(rows, eps=eps, auc_key="binary_auc", auprc_key="binary_auprc")
    if chosen is None:
        raise RuntimeError("no binary packs to select")
    return packs[int(chosen["_idx"])]
