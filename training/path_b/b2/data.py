"""Load C1/W0 blocks + CGM person targets; build Stage-1 OOF Ŷ under PLAN_B2 locks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from training.path_a_blocks.data_blocks import load_watch_onboarding_mood
from training.path_a_watch.data import feature_columns as watch_feature_columns

DENY = {
    "label",
    "recommended_split",
    "clinical_site",
    "person_id",
    "study_group",
    "wearable_core",
    "wearable_core_strict",
    "aux_eligible",
    "age_discrepancy",
}


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def load_path_a_blocks_cfg(repo: Path, path: str) -> dict[str, Any]:
    with _resolve(repo, path).open() as f:
        return yaml.safe_load(f)


def load_b2_frame(
    repo: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Full wearable_core frame with W0, C1, true glu cols, pool flags.

    Returns df + col groups dict.
    """
    pa = load_path_a_blocks_cfg(repo, cfg["paths"]["path_a_blocks_config"])
    onboard_keep = list(pa["data"]["onboarding_keep"])
    mood_scores = list(pa["data"]["mood_scores"])  # cestl, paidscore

    df, watch_cols, onboard_cols, mood_cols, c1_cols = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=onboard_keep,
        mood_cols=mood_scores,
        expected_n=int(cfg["data"]["expected_n_core"]),
    )
    # Ensure aux_eligible is present (load_watch_onboarding_mood uses load_merged)
    if "aux_eligible" not in df.columns:
        raise AssertionError("aux_eligible missing after merge")

    w0 = list(watch_cols)
    if len(w0) != 30:
        raise AssertionError(f"expected 30 GREEN cols, got {len(w0)}")
    expected_c1_n = int(cfg["data"]["expected_c1_n_feat"])
    if len(c1_cols) != expected_c1_n:
        raise AssertionError(
            f"C1 n_feat={len(c1_cols)} != expected {expected_c1_n}: {c1_cols}"
        )

    cgm = pd.read_parquet(_resolve(repo, cfg["paths"]["cgm_person"]))
    glu_targets = list(cfg["data"]["glu_targets"])
    forbid = set(cfg["data"]["glu_forbid"])
    bad = [c for c in glu_targets if c in forbid]
    if bad:
        raise AssertionError(f"forbidden glu targets: {bad}")
    miss = [c for c in glu_targets if c not in cgm.columns]
    if miss:
        raise ValueError(f"cgm_person missing {miss}")

    true_prefix = cfg["data"]["true_prefix"]
    keep_cgm = ["person_id"] + glu_targets
    cgm = cgm[keep_cgm].copy()
    rename = {c: f"{true_prefix}{c}" for c in glu_targets}
    cgm = cgm.rename(columns=rename)
    true_cols = [rename[c] for c in glu_targets]

    n_before = len(df)
    df = df.merge(cgm, on="person_id", how="left")
    if len(df) != n_before:
        raise AssertionError("cgm_person merge changed row count")

    # aux must have true CGM
    aux = df["aux_eligible"].astype(bool)
    if df.loc[aux, true_cols].isna().any().any():
        n_bad = int(df.loc[aux, true_cols].isna().any(axis=1).sum())
        raise AssertionError(f"aux pids with null true CGM: {n_bad}")

    n_aux = int(aux.sum())
    exp_aux = int(cfg["data"]["expected_n_aux"])
    if n_aux != exp_aux:
        raise AssertionError(f"aux_eligible n={n_aux} != {exp_aux}")

    groups = {
        "w0": w0,
        "onboarding": list(onboard_cols),
        "mood": list(mood_cols),
        "c1": list(c1_cols),
        "glu_targets": glu_targets,
        "true_cols": true_cols,
    }
    return df, groups


def pred_col_names(cfg: dict[str, Any], glu_targets: list[str]) -> list[str]:
    pfx = cfg["data"]["pred_prefix"]
    return [f"{pfx}{c}" for c in glu_targets]


def assert_no_leakage(feature_cols: list[str], cfg: dict[str, Any]) -> None:
    """Deployable-arm guard: no meta, coverage counts, true CGM, or raw glu names."""
    forbid = set(DENY) | set(cfg["data"]["glu_forbid"])
    # raw true glu names without prefix also forbidden if present
    forbid |= set(cfg["data"]["glu_targets"])
    bad = [c for c in feature_cols if c in forbid or c.startswith("ytrue_")]
    if bad:
        raise AssertionError(f"leakage/deny in features: {bad}")


def assert_oracle_features(feature_cols: list[str], true_cols: list[str]) -> None:
    for c in true_cols:
        if c not in feature_cols:
            raise AssertionError(f"oracle missing {c}")
    if any(c.startswith("yhat_") for c in feature_cols):
        raise AssertionError("oracle arm must not include yhat_ predicted CGM")


@dataclass
class Stage1Predictions:
    """Person-level Ŷ aligned to full core df index order by person_id map."""

    yhat_train: pd.DataFrame  # person_id + pred cols (train rows only, OOF)
    yhat_val: pd.DataFrame
    yhat_test: pd.DataFrame
    fold_label_counts: list[dict[str, Any]]
    stage1_val_metrics: dict[str, Any]
    stage1_test_metrics: dict[str, Any]
    best_params_per_target: dict[str, dict[str, Any]]


def merge_yhat_into_df(
    df: pd.DataFrame,
    preds: Stage1Predictions,
    pred_cols: list[str],
) -> pd.DataFrame:
    """Attach Ŷ for all splits onto a copy of df."""
    parts = [preds.yhat_train, preds.yhat_val, preds.yhat_test]
    yhat = pd.concat(parts, axis=0, ignore_index=True)
    if yhat["person_id"].duplicated().any():
        raise AssertionError("duplicate person_id in concatenated Ŷ")
    out = df.merge(yhat, on="person_id", how="left")
    if out[pred_cols].isna().any().any():
        n = int(out[pred_cols].isna().any(axis=1).sum())
        raise AssertionError(f"missing Ŷ after merge for {n} pids")
    return out


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    pool: str,
    id_col: str = "person_id",
    label_col: str = "label",
    split_col: str = "recommended_split",
) -> dict[str, Any]:
    """pool: 'core' | 'aux'."""
    d = df
    if pool == "aux":
        d = df[df["aux_eligible"].astype(bool)].copy()
    elif pool != "core":
        raise ValueError(pool)

    def _part(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        m = d[split_col] == name
        if not m.any():
            raise AssertionError(f"no rows for {pool}/{name}")
        X = d.loc[m, feature_cols].copy()
        y = d.loc[m, label_col].to_numpy(dtype=np.int64)
        pid = d.loc[m, id_col].to_numpy()
        return X, y, pid

    Xtr, ytr, pid_tr = _part("train")
    Xva, yva, pid_va = _part("val")
    Xte, yte, pid_te = _part("test")
    return {
        "pool": pool,
        "feature_cols": list(feature_cols),
        "X_train": Xtr,
        "y_train": ytr,
        "pid_train": pid_tr,
        "X_val": Xva,
        "y_val": yva,
        "pid_val": pid_va,
        "X_test": Xte,
        "y_test": yte,
        "pid_test": pid_te,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
    }


def yhat_drift_table(
    df: pd.DataFrame,
    pred_cols: list[str],
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Percentiles of Ŷ on split, by aux vs non-aux."""
    d = df[df["recommended_split"] == split]
    out: dict[str, Any] = {"split": split, "dims": {}}
    for c in pred_cols:
        rows = {}
        for name, mask in (
            ("aux", d["aux_eligible"].astype(bool)),
            ("non_aux", ~d["aux_eligible"].astype(bool)),
        ):
            s = d.loc[mask, c]
            if len(s) == 0:
                rows[name] = {"n": 0}
            else:
                rows[name] = {
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "p10": float(s.quantile(0.10)),
                    "p50": float(s.quantile(0.50)),
                    "p90": float(s.quantile(0.90)),
                }
        out["dims"][c] = rows
    return out


def subsample_train_for_smoke(
    df: pd.DataFrame,
    *,
    n_train: int,
    seed: int,
) -> pd.DataFrame:
    """Keep all val/test; subsample train stratified by label (core)."""
    tr = df[df["recommended_split"] == "train"]
    other = df[df["recommended_split"] != "train"]
    if len(tr) <= n_train:
        return df.copy()
    parts = []
    rng = np.random.default_rng(seed)
    # proportional per label
    for lab, g in tr.groupby("label"):
        frac = len(g) / len(tr)
        k = max(1, int(round(n_train * frac)))
        k = min(k, len(g))
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        parts.append(df.loc[idx])
    tr_s = pd.concat(parts, axis=0)
    # fix size if rounding drifted
    if len(tr_s) > n_train:
        tr_s = tr_s.sample(n=n_train, random_state=seed)
    out = pd.concat([tr_s, other], axis=0).sort_index()
    return out.reset_index(drop=True)
