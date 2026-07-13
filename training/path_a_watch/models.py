"""Model builders and fit helpers for LightGBM / CatBoost / ordinal logistic."""

from __future__ import annotations

import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel


def resolve_lgbm_device(requested: str = "auto") -> str:
    if requested in ("cpu", "gpu"):
        if requested == "gpu":
            ok = _probe_lgbm_gpu()
            return "gpu" if ok else "cpu"
        return "cpu"
    if requested == "auto":
        return "gpu" if _probe_lgbm_gpu() else "cpu"
    raise ValueError(requested)


def _probe_lgbm_gpu() -> bool:
    try:
        X = np.random.randn(64, 4)
        y = np.random.randint(0, 3, size=64)
        m = LGBMClassifier(
            n_estimators=5,
            num_leaves=8,
            verbosity=-1,
            device="gpu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        return True
    except Exception:
        return False


def make_lgbm(
    params: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
    device: str,
    n_estimators: int,
    class_weight: str | dict | None = "balanced",
) -> LGBMClassifier:
    p = dict(params)
    return LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight=class_weight,
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        device=device,
        verbosity=-1,
        **p,
    )


def fit_lgbm(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    es_rounds: int,
) -> LGBMClassifier:
    callbacks = [
        lgb.early_stopping(es_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )
    return model


def make_catboost(
    params: dict[str, Any],
    *,
    seed: int,
    n_estimators: int,
    es_rounds: int,
    boosting_type: str = "Ordered",
    auto_class_weights: str = "Balanced",
    task_type: str = "CPU",
) -> CatBoostClassifier:
    p = dict(params)
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=n_estimators,
        random_seed=seed,
        auto_class_weights=auto_class_weights,
        boosting_type=boosting_type,
        task_type=task_type,
        early_stopping_rounds=es_rounds,
        verbose=False,
        allow_writing_files=False,
        **p,
    )


def fit_catboost(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> CatBoostClassifier:
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model


def try_catboost_boosting(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    es_rounds: int,
    preferred: str = "Ordered",
    fallback: str = "Plain",
    auto_class_weights: str = "Balanced",
    task_type: str = "CPU",
) -> tuple[CatBoostClassifier, str]:
    last_err: Exception | None = None
    tried: list[str] = []
    for bt in (preferred, fallback):
        if bt in tried:
            continue
        tried.append(bt)
        try:
            model = make_catboost(
                params,
                seed=seed,
                n_estimators=n_estimators,
                es_rounds=es_rounds,
                boosting_type=bt,
                auto_class_weights=auto_class_weights,
                task_type=task_type,
            )
            fit_catboost(model, X_train, y_train, X_val, y_val)
            return model, bt
        except Exception as e:  # Ordered can fail on some builds
            last_err = e
            continue
    raise RuntimeError(f"CatBoost fit failed for {tried}: {last_err}")


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"unexpected proba shape {p.shape}")
    return p


def best_iteration(model: Any) -> int | None:
    if isinstance(model, LGBMClassifier):
        bi = getattr(model, "best_iteration_", None)
        if bi is None or bi <= 0:
            return int(model.n_estimators_)
        return int(bi)
    if isinstance(model, CatBoostClassifier):
        bi = model.get_best_iteration()
        if bi is None:
            return int(model.tree_count_)
        return int(bi)
    return None


def fit_ordinal_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    *,
    maxiter: int = 200,
    method: str = "bfgs",
) -> dict[str, Any]:
    """Thin OrderedModel baseline; returns pack with predict_proba fn."""
    del X_val  # unused; kept for call-site compatibility
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train.to_numpy(dtype=float))
    # OrderedModel construction can emit non-critical warnings; keep ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        om = OrderedModel(y_train, Xtr, distr="logit")
        res = om.fit(method=method, maxiter=maxiter, disp=False)

    mle = getattr(res, "mle_retvals", None)
    if not isinstance(mle, dict):
        mle = {}
    converged = bool(mle.get("converged", False))

    def _proba(X: pd.DataFrame) -> np.ndarray:
        Xs = scaler.transform(X.to_numpy(dtype=float))
        return np.asarray(
            res.model.predict(res.params, exog=Xs, which="prob"),
            dtype=float,
        )

    return {
        "result": res,
        "scaler": scaler,
        "predict_proba": _proba,
        "params_summary": str(res.summary()) if hasattr(res, "summary") else "",
        "llf": float(getattr(res, "llf", float("nan"))),
        "n_features": int(X_train.shape[1]),
        "converged": converged,
        "mle_retvals": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                        for k, v in (mle.items() if isinstance(mle, dict) else [])
                        if k in ("converged", "fopt", "iterations", "warnflag")},
    }
