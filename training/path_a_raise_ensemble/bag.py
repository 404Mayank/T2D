"""Multi-seed bagging for frozen C1 LGBM + CatBoost params."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from training.path_a_watch.data import SplitData
from training.path_a_watch.metrics import macro_ovr_auc
from training.path_a_watch.models import (
    best_iteration,
    fit_lgbm,
    make_lgbm,
    predict_proba,
    try_catboost_boosting,
)


@dataclass
class SeedFit:
    family: str
    seed: int
    model: Any
    best_iteration: int | None
    boosting_type: str | None
    device: str | None
    proba_train: np.ndarray
    proba_val: np.ndarray
    proba_test: np.ndarray
    val_macro_ovr_auc: float
    test_macro_ovr_auc: float


@dataclass
class FamilyBag:
    family: str
    seeds: list[int]
    fits: list[SeedFit] = field(default_factory=list)
    proba_train: np.ndarray | None = None
    proba_val: np.ndarray | None = None
    proba_test: np.ndarray | None = None
    val_macro_ovr_auc: float | None = None
    test_macro_ovr_auc: float | None = None

    def finalize(self, y_val: np.ndarray, y_test: np.ndarray) -> None:
        self.proba_train = np.mean([f.proba_train for f in self.fits], axis=0)
        self.proba_val = np.mean([f.proba_val for f in self.fits], axis=0)
        self.proba_test = np.mean([f.proba_test for f in self.fits], axis=0)
        self.val_macro_ovr_auc = float(macro_ovr_auc(y_val, self.proba_val))
        self.test_macro_ovr_auc = float(macro_ovr_auc(y_test, self.proba_test))


def fit_lgbm_seed(
    splits: SplitData,
    *,
    params: dict[str, Any],
    seed: int,
    device: str,
    n_jobs: int,
    n_estimators: int,
    es_rounds: int,
    class_weight: str,
) -> SeedFit:
    model = make_lgbm(
        params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        n_estimators=n_estimators,
        class_weight=class_weight,
    )
    fit_lgbm(
        model,
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        es_rounds=es_rounds,
    )
    p_tr = predict_proba(model, splits.X_train)
    p_va = predict_proba(model, splits.X_val)
    p_te = predict_proba(model, splits.X_test)
    return SeedFit(
        family="lightgbm",
        seed=seed,
        model=model,
        best_iteration=best_iteration(model),
        boosting_type=None,
        device=device,
        proba_train=p_tr,
        proba_val=p_va,
        proba_test=p_te,
        val_macro_ovr_auc=float(macro_ovr_auc(splits.y_val, p_va)),
        test_macro_ovr_auc=float(macro_ovr_auc(splits.y_test, p_te)),
    )


def fit_cat_seed(
    splits: SplitData,
    *,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    es_rounds: int,
    preferred: str,
    fallback: str,
    auto_class_weights: str,
    task_type: str,
) -> SeedFit:
    model, bt = try_catboost_boosting(
        params,
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        seed=seed,
        n_estimators=n_estimators,
        es_rounds=es_rounds,
        preferred=preferred,
        fallback=fallback,
        auto_class_weights=auto_class_weights,
        task_type=task_type,
    )
    p_tr = predict_proba(model, splits.X_train)
    p_va = predict_proba(model, splits.X_val)
    p_te = predict_proba(model, splits.X_test)
    return SeedFit(
        family="catboost",
        seed=seed,
        model=model,
        best_iteration=best_iteration(model),
        boosting_type=bt,
        device=None,
        proba_train=p_tr,
        proba_val=p_va,
        proba_test=p_te,
        val_macro_ovr_auc=float(macro_ovr_auc(splits.y_val, p_va)),
        test_macro_ovr_auc=float(macro_ovr_auc(splits.y_test, p_te)),
    )


def fit_family_bag(
    splits: SplitData,
    *,
    family: str,
    seeds: list[int],
    params: dict[str, Any],
    cfg_run: dict[str, Any],
    class_weights: dict[str, Any],
    lgbm_device: str,
    log: Callable[[str], None] | None = None,
) -> FamilyBag:
    """Fit multi-seed bag for one family on the outer train/val/test splits."""
    _log = log or (lambda _m: None)
    bag = FamilyBag(family=family, seeds=list(seeds))
    n_est = int(cfg_run["n_estimators_max"])
    es = int(cfg_run["es_rounds"])
    n_jobs = int(cfg_run.get("n_jobs", -1))
    if family == "lightgbm" and lgbm_device == "gpu" and n_jobs == -1:
        n_jobs = 4

    for seed in seeds:
        _log(f"fit {family} seed={seed}")
        if family == "lightgbm":
            fit = fit_lgbm_seed(
                splits,
                params=params,
                seed=int(seed),
                device=lgbm_device,
                n_jobs=n_jobs,
                n_estimators=n_est,
                es_rounds=es,
                class_weight=str(class_weights["lightgbm"]),
            )
        elif family == "catboost":
            fit = fit_cat_seed(
                splits,
                params=params,
                seed=int(seed),
                n_estimators=n_est,
                es_rounds=es,
                preferred=str(cfg_run.get("catboost_boosting_type", "Ordered")),
                fallback=str(cfg_run.get("catboost_boosting_fallback", "Plain")),
                auto_class_weights=str(class_weights["catboost"]),
                task_type=str(cfg_run.get("catboost_task_type", "CPU")),
            )
        else:
            raise ValueError(family)
        bag.fits.append(fit)
        _log(
            f"  {family} seed={seed} val_auc={fit.val_macro_ovr_auc:.4f} "
            f"best_iter={fit.best_iteration} bt={fit.boosting_type}"
        )

    bag.finalize(splits.y_val, splits.y_test)
    _log(
        f"bag {family} val_auc={bag.val_macro_ovr_auc:.4f} "
        f"test_auc={bag.test_macro_ovr_auc:.4f}"
    )
    return bag


def save_bag_models(bag: FamilyBag, models_dir: Path) -> list[str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fit in bag.fits:
        p = models_dir / f"{bag.family}_seed{fit.seed}.joblib"
        joblib.dump(fit.model, p)
        paths.append(str(p))
    return paths


def bag_meta(bag: FamilyBag) -> dict[str, Any]:
    return {
        "family": bag.family,
        "seeds": list(bag.seeds),
        "val_macro_ovr_auc": bag.val_macro_ovr_auc,
        "test_macro_ovr_auc": bag.test_macro_ovr_auc,
        "per_seed": [
            {
                "seed": f.seed,
                "best_iteration": f.best_iteration,
                "boosting_type": f.boosting_type,
                "device": f.device,
                "val_macro_ovr_auc": f.val_macro_ovr_auc,
                "test_macro_ovr_auc": f.test_macro_ovr_auc,
            }
            for f in bag.fits
        ],
    }


def fit_lgbm_on_arrays(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_pred: pd.DataFrame,
    *,
    params: dict[str, Any],
    seed: int,
    device: str,
    n_jobs: int,
    n_estimators: int,
    es_rounds: int,
    class_weight: str,
) -> np.ndarray:
    """Fit LGBM on arbitrary train/val; predict X_pred (used by stacker OOF)."""
    model = make_lgbm(
        params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        n_estimators=n_estimators,
        class_weight=class_weight,
    )
    fit_lgbm(model, X_tr, y_tr, X_va, y_va, es_rounds=es_rounds)
    return predict_proba(model, X_pred)


def fit_cat_on_arrays(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_pred: pd.DataFrame,
    *,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    es_rounds: int,
    preferred: str,
    fallback: str,
    auto_class_weights: str,
    task_type: str,
) -> np.ndarray:
    model, _bt = try_catboost_boosting(
        params,
        X_tr,
        y_tr,
        X_va,
        y_va,
        seed=seed,
        n_estimators=n_estimators,
        es_rounds=es_rounds,
        preferred=preferred,
        fallback=fallback,
        auto_class_weights=auto_class_weights,
        task_type=task_type,
    )
    return predict_proba(model, X_pred)
