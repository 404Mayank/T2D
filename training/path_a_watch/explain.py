"""SHAP + permutation importance for tree models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .models import predict_proba


def shap_summary(
    model: Any,
    X: pd.DataFrame,
    out_dir: Path,
    *,
    max_samples: int = 400,
    seed: int = 42,
    prefix: str = "model",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        Xs = X.iloc[idx]
    else:
        Xs = X

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    # multiclass: list of arrays or 3D
    if isinstance(sv, list):
        # mean |shap| across classes
        abs_mean = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            # (n, features, classes) or (classes, n, features)
            if arr.shape[0] == len(Xs):
                abs_mean = np.abs(arr).mean(axis=(0, 2))
            else:
                abs_mean = np.abs(arr).mean(axis=(0, 1))
        else:
            abs_mean = np.abs(arr).mean(axis=0)

    importance = (
        pd.Series(abs_mean, index=list(X.columns))
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )
    csv_path = out_dir / f"{prefix}_shap_importance.csv"
    importance.to_csv(csv_path, header=True)

    plt.figure(figsize=(8, 6))
    importance.head(20).iloc[::-1].plot(kind="barh")
    plt.title(f"{prefix} mean |SHAP| (top 20)")
    plt.tight_layout()
    png_path = out_dir / f"{prefix}_shap_bar.png"
    plt.savefig(png_path, dpi=120)
    plt.close()

    coverage = [c for c in ("hr_n", "stress_n", "sleep_n_nights", "activity_n_days") if c in importance.index]
    cov_ranks = {c: int(importance.index.get_loc(c) + 1) for c in coverage}

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "top10": importance.head(10).to_dict(),
        "coverage_ranks": cov_ranks,
        "n_samples_explained": int(len(Xs)),
    }


def permutation_on_val(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    out_dir: Path,
    *,
    seed: int = 42,
    n_repeats: int = 10,
    prefix: str = "model",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _auc_scorer(est, X, y):
        from .metrics import macro_ovr_auc

        p = predict_proba(est, pd.DataFrame(X, columns=X_val.columns))
        return macro_ovr_auc(y, p)

    r = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=_auc_scorer,
        n_jobs=1,
    )
    s = (
        pd.Series(r.importances_mean, index=list(X_val.columns))
        .sort_values(ascending=False)
        .rename("perm_auc_drop")
    )
    csv_path = out_dir / f"{prefix}_perm_importance.csv"
    s.to_csv(csv_path, header=True)

    plt.figure(figsize=(8, 6))
    s.head(20).iloc[::-1].plot(kind="barh")
    plt.title(f"{prefix} permutation ΔAUC (top 20)")
    plt.tight_layout()
    png_path = out_dir / f"{prefix}_perm_bar.png"
    plt.savefig(png_path, dpi=120)
    plt.close()

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "top10": s.head(10).to_dict(),
    }


def permutation_on_val_binary(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    out_dir: Path,
    *,
    seed: int = 42,
    n_repeats: int = 10,
    prefix: str = "model",
) -> dict[str, Any]:
    """Permutation importance scored by binary AUC on P(y=1)."""
    from sklearn.metrics import roc_auc_score

    from .models import predict_proba_positive

    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_val).astype(int).ravel()
    if y.max() > 1:
        y = (y > 0).astype(int)

    def _auc_scorer(est, X, y_in):
        p = predict_proba_positive(est, pd.DataFrame(X, columns=X_val.columns))
        return float(roc_auc_score(y_in, p))

    r = permutation_importance(
        model,
        X_val,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=_auc_scorer,
        n_jobs=1,
    )
    s = (
        pd.Series(r.importances_mean, index=list(X_val.columns))
        .sort_values(ascending=False)
        .rename("perm_binary_auc_drop")
    )
    csv_path = out_dir / f"{prefix}_perm_importance.csv"
    s.to_csv(csv_path, header=True)

    plt.figure(figsize=(8, 6))
    s.head(20).iloc[::-1].plot(kind="barh")
    plt.title(f"{prefix} permutation Δ binary AUC (top 20)")
    plt.tight_layout()
    png_path = out_dir / f"{prefix}_perm_bar.png"
    plt.savefig(png_path, dpi=120)
    plt.close()

    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "top10": s.head(10).to_dict(),
    }
