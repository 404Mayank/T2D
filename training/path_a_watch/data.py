"""Load watch_green ⋈ pool_masks under the PROCESSED consumer contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COVERAGE_COLS_DEFAULT = ("hr_n", "stress_n", "sleep_n_nights", "activity_n_days")
DENY_COLS_DEFAULT = (
    "label",
    "recommended_split",
    "clinical_site",
    "study_group",
    "wearable_core",
    "wearable_core_strict",
    "aux_eligible",
    "age",
    "age_discrepancy",
)


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_cols: list[str]
    feature_set: str
    person_id_train: np.ndarray
    person_id_val: np.ndarray
    person_id_test: np.ndarray
    meta_table: pd.DataFrame  # one row per pid used (diagnostics only)


def _resolve(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_merged(
    repo_root: Path,
    watch_green: str | Path,
    pool_masks: str | Path,
    *,
    expected_n: int = 1824,
    id_col: str = "person_id",
) -> pd.DataFrame:
    feats = pd.read_parquet(_resolve(repo_root, watch_green))
    meta = pd.read_parquet(_resolve(repo_root, pool_masks))
    if id_col not in feats.columns:
        raise ValueError(f"{watch_green} missing {id_col}")
    need = [
        id_col,
        "label",
        "recommended_split",
        "clinical_site",
        "wearable_core",
        "wearable_core_strict",
        "aux_eligible",
    ]
    missing = [c for c in need if c not in meta.columns]
    if missing:
        raise ValueError(f"pool_masks missing columns: {missing}")

    df = feats.merge(meta[need], on=id_col, how="inner")
    if len(df) != expected_n:
        raise AssertionError(
            f"merged n={len(df)} != expected_n={expected_n} (PROCESSED wearable_core)"
        )
    if not bool(df["wearable_core"].all()):
        raise AssertionError("not all merged rows are wearable_core")
    if df[id_col].duplicated().any():
        raise AssertionError("duplicate person_id after merge")
    if df["label"].isna().any() or df["recommended_split"].isna().any():
        raise AssertionError("null label or split after merge")
    return df


def feature_columns(
    df: pd.DataFrame,
    *,
    feature_set: str = "full_green",
    id_col: str = "person_id",
    deny_cols: tuple[str, ...] | list[str] = DENY_COLS_DEFAULT,
    coverage_cols: tuple[str, ...] | list[str] = COVERAGE_COLS_DEFAULT,
) -> list[str]:
    """Return ordered feature columns with hard leakage denies."""
    metaish = set(deny_cols) | {id_col}
    # Anything that came from pool_masks join besides id is forbidden in X
    # even if not listed (defense in depth).
    joined_meta = {
        "label",
        "recommended_split",
        "clinical_site",
        "wearable_core",
        "wearable_core_strict",
        "aux_eligible",
    }
    cols = [c for c in df.columns if c not in metaish and c not in joined_meta]

    # Must only be original watch_green numeric/feature columns
    bad = [c for c in cols if c in deny_cols or c in joined_meta]
    if bad:
        raise AssertionError(f"deny-list columns leaked into feature list: {bad}")

    if feature_set == "full_green":
        out = cols
    elif feature_set == "physio_only":
        drop = set(coverage_cols)
        out = [c for c in cols if c not in drop]
        missing = [c for c in coverage_cols if c not in cols]
        if missing:
            raise AssertionError(f"coverage cols not in matrix: {missing}")
    else:
        raise ValueError(f"unknown feature_set={feature_set!r}")

    if not out:
        raise AssertionError("empty feature list")
    return out


def make_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    feature_set: str,
    id_col: str = "person_id",
    label_col: str = "label",
    split_col: str = "recommended_split",
) -> SplitData:
    for req in (id_col, label_col, split_col):
        if req not in df.columns:
            raise ValueError(f"missing {req}")

    def _part(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        m = df[split_col] == name
        if not m.any():
            raise AssertionError(f"no rows for split={name}")
        X = df.loc[m, feature_cols].copy()
        y = df.loc[m, label_col].to_numpy(dtype=np.int64)
        pid = df.loc[m, id_col].to_numpy()
        return X, y, pid

    X_tr, y_tr, p_tr = _part("train")
    X_va, y_va, p_va = _part("val")
    X_te, y_te, p_te = _part("test")

    meta_table = df[
        [id_col, label_col, split_col, "clinical_site", "wearable_core"]
    ].copy()

    return SplitData(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        feature_cols=list(feature_cols),
        feature_set=feature_set,
        person_id_train=p_tr,
        person_id_val=p_va,
        person_id_test=p_te,
        meta_table=meta_table,
    )


def split_base_rates(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, g in df.groupby("recommended_split"):
        vc = g["label"].value_counts().sort_index()
        out[str(split)] = {
            "n": int(len(g)),
            "by_label": {int(k): int(v) for k, v in vc.items()},
        }
    return out


def site_by_label(df: pd.DataFrame, split: str | None = "train") -> dict[str, Any]:
    d = df if split is None else df[df["recommended_split"] == split]
    ct = pd.crosstab(d["clinical_site"], d["label"])
    return {
        "split": split,
        "counts": {str(s): {int(k): int(v) for k, v in row.items()} for s, row in ct.iterrows()},
        "row_frac": {
            str(s): {int(k): float(v) for k, v in row.items()}
            for s, row in ct.div(ct.sum(axis=1), axis=0).iterrows()
        },
    }


def describe_cohort(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    return {
        "n": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "nulls_in_X": int(df[feature_cols].isna().sum().sum()),
        "base_rates": split_base_rates(df),
        "site_by_label_train": site_by_label(df, "train"),
        "label_counts_all": {
            int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
        },
    }
