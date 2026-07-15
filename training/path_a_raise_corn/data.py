"""Load exact C1 matrix, hash-assert, train-median impute + z-score."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import (
    load_watch_onboarding_mood,
    resolve_mood_cols,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]


def load_blocks_cfg(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    p = repo / cfg["paths"]["path_a_blocks_config"]
    with p.open() as f:
        return yaml.safe_load(f)


@dataclass
class C1Bundle:
    df: pd.DataFrame
    feature_cols: list[str]
    watch_cols: list[str]
    onboard_cols: list[str]
    mood_cols: list[str]
    feature_hash: str
    nulls_before: dict[str, Any]


def load_c1_matrix(repo: Path, cfg: dict[str, Any]) -> C1Bundle:
    blocks = load_blocks_cfg(repo, cfg)
    mood_cols = resolve_mood_cols(blocks["data"], "scores")
    df, watch, onboard, mood, feat = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=list(blocks["data"]["onboarding_keep"]),
        mood_cols=mood_cols,
        expected_n=int(cfg["data"]["expected_n"]),
        id_col=str(cfg["data"]["id_col"]),
    )
    # Canonical order from frozen C1 features.json
    parent_art = (
        repo
        / cfg["paths"]["parent_c1_artifacts"]
        / cfg["paths"]["parent_c1_run_id"]
        / "features.json"
    )
    parent_feat = json.loads(parent_art.read_text())
    canonical = list(parent_feat["feature_cols"])
    if set(canonical) != set(feat):
        raise AssertionError(
            f"C1 feature set drift: extra={set(feat)-set(canonical)} "
            f"missing={set(canonical)-set(feat)}"
        )
    feat = canonical
    h = feature_hash(feat)
    expected_h = str(cfg["data"]["feature_hash"])
    if h != expected_h:
        raise AssertionError(f"feature_hash {h} != config {expected_h}")
    if h != parent_feat.get("feature_hash"):
        raise AssertionError(
            f"feature_hash {h} != parent artifact {parent_feat.get('feature_hash')}"
        )
    if len(df) != int(cfg["data"]["expected_n"]):
        raise AssertionError(f"n={len(df)} != expected {cfg['data']['expected_n']}")
    if len(feat) != int(cfg["data"]["n_features"]):
        raise AssertionError(f"n_features={len(feat)} != {cfg['data']['n_features']}")

    null_frac = df[feat].isna().mean()
    nulls = {
        "n": int(len(df)),
        "rows_any_null": int(df[feat].isna().any(axis=1).sum()),
        "null_frac_by_col": {
            k: float(v) for k, v in null_frac.items() if float(v) > 0
        },
    }
    return C1Bundle(
        df=df,
        feature_cols=feat,
        watch_cols=list(watch),
        onboard_cols=list(onboard),
        mood_cols=list(mood),
        feature_hash=h,
        nulls_before=nulls,
    )


def split_frames(
    bundle: C1Bundle,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    df = bundle.df
    id_col = cfg["data"]["id_col"]
    y_col = cfg["data"]["label_col"]
    s_col = cfg["data"]["split_col"]
    cols = bundle.feature_cols

    def _part(name: str) -> dict[str, Any]:
        m = df[s_col] == name
        sub = df.loc[m]
        return {
            "X": sub[cols].copy(),
            "y": sub[y_col].to_numpy(dtype=np.int64),
            "pid": sub[id_col].to_numpy(),
            "n": int(m.sum()),
        }

    out = {
        "train": _part("train"),
        "val": _part("val"),
        "test": _part("test"),
        "feature_cols": cols,
        "feature_hash": bundle.feature_hash,
    }
    n_sum = out["train"]["n"] + out["val"]["n"] + out["test"]["n"]
    if n_sum != len(df):
        raise AssertionError(f"split sum {n_sum} != n {len(df)}")
    return out


def fit_impute_scale(X_train: pd.DataFrame) -> dict[str, Any]:
    med = X_train.median(axis=0)
    Xf = X_train.fillna(med)
    mu = Xf.mean(axis=0).to_numpy(dtype=np.float64)
    sd = Xf.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return {
        "median": {k: float(v) for k, v in med.items()},
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "columns": list(X_train.columns),
    }


def apply_impute_scale(X: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    cols = state["columns"]
    med = pd.Series(state["median"])
    Xf = X[cols].fillna(med)
    mu = np.asarray(state["mean"], dtype=np.float64)
    sd = np.asarray(state["std"], dtype=np.float64)
    return (Xf.to_numpy(dtype=np.float64) - mu) / sd


def fh_post_impute_rates(
    X_train: pd.DataFrame,
    state: dict[str, Any],
    fh_cols: tuple[str, ...] = ("fh_dm2pt", "fh_dm2sb"),
) -> dict[str, Any]:
    """Diagnostics: positive rates before/after train-median fill."""
    out: dict[str, Any] = {}
    med = pd.Series(state["median"])
    for c in fh_cols:
        if c not in X_train.columns:
            continue
        s = X_train[c]
        out[c] = {
            "null_frac": float(s.isna().mean()),
            "pos_rate_observed": float((s.dropna() == 1).mean()) if s.notna().any() else float("nan"),
            "median_fill": float(med[c]),
            "pos_rate_after_impute": float((s.fillna(med[c]) == 1).mean()),
        }
    return out


def assert_parent_c1(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["parent_c1_run_id"]
    art = repo / cfg["paths"]["parent_c1_artifacts"] / run_id
    path = art / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["parent_c1_reference"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"parent_c1 {k}: config {ref[k]} != artifact {v}")
    freeze = json.loads((art / "selected_model.json").read_text())
    model_path = art / "models" / "selected.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        **checks,
        "family": m.get("selected_family"),
        "feature_cols": list(freeze["feature_cols"]),
        "model_path": str(model_path),
        "run_id": run_id,
        "class2_ovr_auc": float(raw["per_class_ovr_auc"]["2"]),
        "per_class_ovr_auc": raw["per_class_ovr_auc"],
    }


def load_parent_c1_proba(
    repo: Path,
    cfg: dict[str, Any],
    parent: dict[str, Any],
    test_pids: np.ndarray,
    *,
    tol: float = 1e-6,
) -> np.ndarray:
    import joblib

    from training.path_a_watch.metrics import macro_ovr_auc
    from training.path_a_watch.models import predict_proba

    bundle = load_c1_matrix(repo, cfg)
    feat = list(parent["feature_cols"])
    df = bundle.df.set_index(cfg["data"]["id_col"])
    X = df[feat].loc[list(test_pids)].reindex(columns=feat)
    if list(X.columns) != feat:
        raise AssertionError("C1 col order mismatch for parent proba")
    model = joblib.load(parent["model_path"])
    proba = predict_proba(model, X)
    y = df.loc[list(test_pids), cfg["data"]["label_col"]].to_numpy(dtype=np.int64)
    recomp = macro_ovr_auc(y, proba)
    if abs(recomp - parent["test_macro_ovr_auc"]) > tol:
        raise AssertionError(f"C1 recompute {recomp} != {parent['test_macro_ovr_auc']}")
    return np.asarray(proba, dtype=float)
