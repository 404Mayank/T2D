"""Load watch GREEN ± onboarding under PROCESSED rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from training.path_a_watch.data import (
    DENY_COLS_DEFAULT,
    SplitData,
    feature_columns as watch_feature_columns,
    load_merged,
    make_splits,
)


def _resolve(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_watch_onboarding(
    repo_root: Path,
    *,
    watch_green: str,
    onboarding: str,
    pool_masks: str,
    onboarding_keep: list[str],
    expected_n: int = 1824,
    id_col: str = "person_id",
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Return merged df, watch_cols, onboard_cols, all feature cols."""
    watch_df = load_merged(
        repo_root,
        watch_green,
        pool_masks,
        expected_n=expected_n,
        id_col=id_col,
    )
    watch_cols = watch_feature_columns(
        watch_df, feature_set="full_green", id_col=id_col
    )

    onboard = pd.read_parquet(_resolve(repo_root, onboarding))
    missing = [c for c in onboarding_keep if c not in onboard.columns]
    if missing:
        raise ValueError(f"onboarding missing keep cols: {missing}")

    keep = [id_col] + list(onboarding_keep)
    onboard = onboard[keep].copy()
    # dedupe person_id if any
    if onboard[id_col].duplicated().any():
        onboard = onboard.drop_duplicates(id_col, keep="first")

    n_before = len(watch_df)
    df = watch_df.merge(onboard, on=id_col, how="left")
    if len(df) != n_before:
        raise AssertionError("onboarding merge changed row count")
    if not bool(df["wearable_core"].all()):
        raise AssertionError("wearable_core violated after onboard merge")

    onboard_cols = list(onboarding_keep)
    # age is a valid onboarding feature (Training.md hard onboarding).
    # Deny only true leakage / pool meta — not anthropometrics.
    forbid = {
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
    bad = [c for c in onboard_cols if c in forbid]
    if bad:
        raise AssertionError(f"forbidden onboard cols: {bad}")
    # watch cols must not include pool meta either
    bad_w = [c for c in watch_cols if c in forbid or c in DENY_COLS_DEFAULT]
    if bad_w:
        raise AssertionError(f"forbidden watch cols: {bad_w}")

    feature_cols = list(watch_cols) + onboard_cols
    for c in feature_cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature {c}")

    return df, list(watch_cols), onboard_cols, feature_cols


def block_tags(watch_cols: list[str], onboard_cols: list[str]) -> dict[str, str]:
    tags = {c: "watch_green" for c in watch_cols}
    tags.update({c: "onboarding" for c in onboard_cols})
    return tags


def null_report(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    s = df[cols].isna().mean().sort_values(ascending=False)
    return {
        "null_frac_by_col": {k: float(v) for k, v in s.items() if v > 0},
        "rows_any_null": int(df[cols].isna().any(axis=1).sum()),
        "n": int(len(df)),
    }


def make_block_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    feature_set: str = "watch_onboarding",
) -> SplitData:
    return make_splits(df, feature_cols, feature_set=feature_set)
