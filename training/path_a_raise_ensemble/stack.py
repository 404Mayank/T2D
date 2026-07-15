"""σ-stacking: multinomial LR on bag-mean family probabilities.

Train OOF mirrors val/test: per fold, multi-seed bag with ES on fold holdout,
then mean proba → stacker features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from training.path_a_raise_ensemble.bag import fit_cat_on_arrays, fit_lgbm_on_arrays
from training.path_a_watch.data import SplitData
from training.path_a_watch.metrics import macro_ovr_auc


def _family_block_l2(coef: np.ndarray, n_classes: int = 4) -> tuple[float, float, float]:
    """coef shape (n_classes, n_features=8) for multinomial; blocks [0:4]=lgbm, [4:8]=cat."""
    # sklearn multinomial coef_: (n_classes, n_features)
    c = np.asarray(coef, dtype=float)
    if c.ndim != 2 or c.shape[1] < 8:
        return float("nan"), float("nan"), float("nan")
    lgbm = float(np.linalg.norm(c[:, 0:4]))
    cat = float(np.linalg.norm(c[:, 4:8]))
    mx = max(lgbm, cat, 1e-12)
    mn = min(lgbm, cat)
    return lgbm, cat, mn / mx


def stack_features(p_lgbm: np.ndarray, p_cat: np.ndarray) -> np.ndarray:
    a = np.asarray(p_lgbm, dtype=float)
    b = np.asarray(p_cat, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return np.hstack([a, b])


@dataclass
class StackResult:
    model: LogisticRegression
    C: float
    oof_train: np.ndarray  # (n_train, 8)
    proba_val: np.ndarray
    proba_test: np.ndarray
    proba_train: np.ndarray
    val_macro_ovr_auc: float
    test_macro_ovr_auc: float
    oof_val_auc_proxy: float  # OOF train labels scored with stacker after fit
    grid: list[dict[str, Any]]
    degeneracy: dict[str, Any]
    claim_eligible: bool
    fold_label_counts: list[dict[str, Any]]


def _oof_bag_mean_family(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    family: str,
    seeds: list[int],
    params: dict[str, Any],
    cfg_run: dict[str, Any],
    class_weights: dict[str, Any],
    lgbm_device: str,
) -> np.ndarray:
    """Bag-mean proba on hold_idx; each seed ES on hold as nested val."""
    X_tr = X.iloc[train_idx]
    y_tr = y[train_idx]
    X_ho = X.iloc[hold_idx]
    y_ho = y[hold_idx]
    n_est = int(cfg_run["n_estimators_max"])
    es = int(cfg_run["es_rounds"])
    n_jobs = int(cfg_run.get("n_jobs", -1))
    if family == "lightgbm" and lgbm_device == "gpu" and n_jobs == -1:
        n_jobs = 4

    acc: list[np.ndarray] = []
    for seed in seeds:
        if family == "lightgbm":
            p = fit_lgbm_on_arrays(
                X_tr,
                y_tr,
                X_ho,
                y_ho,
                X_ho,
                params=params,
                seed=int(seed),
                device=lgbm_device,
                n_jobs=n_jobs,
                n_estimators=n_est,
                es_rounds=es,
                class_weight=str(class_weights["lightgbm"]),
            )
        else:
            p = fit_cat_on_arrays(
                X_tr,
                y_tr,
                X_ho,
                y_ho,
                X_ho,
                params=params,
                seed=int(seed),
                n_estimators=n_est,
                es_rounds=es,
                preferred=str(cfg_run.get("catboost_boosting_type", "Ordered")),
                fallback=str(cfg_run.get("catboost_boosting_fallback", "Plain")),
                auto_class_weights=str(class_weights["catboost"]),
                task_type=str(cfg_run.get("catboost_task_type", "CPU")),
            )
        acc.append(p)
    return np.mean(acc, axis=0)


def build_train_oof(
    splits: SplitData,
    *,
    seeds: list[int],
    lgbm_params: dict[str, Any],
    cat_params: dict[str, Any],
    cfg_run: dict[str, Any],
    class_weights: dict[str, Any],
    lgbm_device: str,
    k_folds: int,
    fold_seed: int = 42,
    log: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return OOF stack features (n_train, 8) and fold label-count diagnostics."""
    _log = log or (lambda _m: None)
    X = splits.X_train
    y = np.asarray(splits.y_train).astype(int)
    n = len(y)
    oof_lgbm = np.zeros((n, 4), dtype=float)
    oof_cat = np.zeros((n, 4), dtype=float)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
    fold_counts: list[dict[str, Any]] = []

    for fi, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(n), y)):
        y_ho = y[ho_idx]
        y_tr = y[tr_idx]
        counts_ho = {int(c): int((y_ho == c).sum()) for c in range(4)}
        counts_tr = {int(c): int((y_tr == c).sum()) for c in range(4)}
        fold_counts.append(
            {"fold": fi, "n_hold": int(len(ho_idx)), "hold": counts_ho, "train": counts_tr}
        )
        for c in range(4):
            if counts_tr[c] < 1 or counts_ho[c] < 1:
                raise AssertionError(
                    f"stack OOF fold {fi} missing class {c}: train={counts_tr} hold={counts_ho}"
                )
        _log(f"stack OOF fold {fi}/{k_folds} hold_n={len(ho_idx)}")
        oof_lgbm[ho_idx] = _oof_bag_mean_family(
            X,
            y,
            tr_idx,
            ho_idx,
            family="lightgbm",
            seeds=seeds,
            params=lgbm_params,
            cfg_run=cfg_run,
            class_weights=class_weights,
            lgbm_device=lgbm_device,
        )
        oof_cat[ho_idx] = _oof_bag_mean_family(
            X,
            y,
            tr_idx,
            ho_idx,
            family="catboost",
            seeds=seeds,
            params=cat_params,
            cfg_run=cfg_run,
            class_weights=class_weights,
            lgbm_device=lgbm_device,
        )
        _log(
            f"  fold {fi} oof lgbm_auc={macro_ovr_auc(y_ho, oof_lgbm[ho_idx]):.4f} "
            f"cat_auc={macro_ovr_auc(y_ho, oof_cat[ho_idx]):.4f}"
        )

    feats = stack_features(oof_lgbm, oof_cat)
    if not np.isfinite(feats).all():
        raise AssertionError("OOF stack features contain non-finite values")
    return feats, fold_counts


def fit_stacker(
    splits: SplitData,
    *,
    oof_train_feats: np.ndarray,
    bag_lgbm_val: np.ndarray,
    bag_cat_val: np.ndarray,
    bag_lgbm_test: np.ndarray,
    bag_cat_test: np.ndarray,
    bag_lgbm_train: np.ndarray,
    bag_cat_train: np.ndarray,
    C_grid: list[float],
    max_iter: int,
    degeneracy_l2_ratio_min: float,
    collapse_val_margin: float,
    bag_cat_val_auc: float,
    bag_lgbm_val_auc: float,
    log: Callable[[str], None] | None = None,
) -> StackResult:
    _log = log or (lambda _m: None)
    y_tr = np.asarray(splits.y_train).astype(int)
    y_va = np.asarray(splits.y_val).astype(int)
    y_te = np.asarray(splits.y_test).astype(int)

    X_va = stack_features(bag_lgbm_val, bag_cat_val)
    X_te = stack_features(bag_lgbm_test, bag_cat_test)
    X_tr_bag = stack_features(bag_lgbm_train, bag_cat_train)

    grid_rows: list[dict[str, Any]] = []
    best_C: float | None = None
    best_auc = -1.0
    best_model: LogisticRegression | None = None

    for C in C_grid:
        lr = LogisticRegression(
            # sklearn>=1.8: multinomial is the multiclass default; multi_class kw removed
            solver="lbfgs",
            max_iter=int(max_iter),
            class_weight="balanced",
            C=float(C),
        )
        lr.fit(oof_train_feats, y_tr)
        p_va = lr.predict_proba(X_va)
        auc = float(macro_ovr_auc(y_va, p_va))
        grid_rows.append({"C": float(C), "val_macro_ovr_auc": auc})
        _log(f"stacker C={C} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_C = float(C)
            best_model = lr

    if best_model is None or best_C is None:
        raise RuntimeError("stacker grid empty")

    p_va = best_model.predict_proba(X_va)
    p_te = best_model.predict_proba(X_te)
    p_tr = best_model.predict_proba(X_tr_bag)
    # OOF train score with chosen stacker (diagnostic)
    p_oof = best_model.predict_proba(oof_train_feats)
    oof_auc = float(macro_ovr_auc(y_tr, p_oof))

    l2_l, l2_c, ratio = _family_block_l2(best_model.coef_)
    best_bag_val = max(float(bag_cat_val_auc), float(bag_lgbm_val_auc))
    degenerate = bool(ratio <= float(degeneracy_l2_ratio_min))
    collapse = bool(best_auc < best_bag_val - float(collapse_val_margin))
    claim_ok = not (degenerate or collapse)

    deg = {
        "l2_lgbm_block": l2_l,
        "l2_cat_block": l2_c,
        "l2_ratio_min_over_max": ratio,
        "degeneracy_l2_ratio_min": float(degeneracy_l2_ratio_min),
        "degenerate": degenerate,
        "val_stack_auc": best_auc,
        "best_bag_val_auc": best_bag_val,
        "collapse_val_margin": float(collapse_val_margin),
        "collapse": collapse,
        "claim_eligible": claim_ok,
    }
    _log(
        f"stacker chosen C={best_C} val_auc={best_auc:.4f} "
        f"degenerate={degenerate} collapse={collapse}"
    )

    return StackResult(
        model=best_model,
        C=best_C,
        oof_train=oof_train_feats,
        proba_val=p_va,
        proba_test=p_te,
        proba_train=p_tr,
        val_macro_ovr_auc=best_auc,
        test_macro_ovr_auc=float(macro_ovr_auc(y_te, p_te)),
        oof_val_auc_proxy=oof_auc,
        grid=grid_rows,
        degeneracy=deg,
        claim_eligible=claim_ok,
        fold_label_counts=[],  # filled by caller
    )
