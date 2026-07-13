"""Load watch GREEN ± onboarding under PROCESSED rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


COMORB_FEATURE_SETS = (
    "core",
    "no_hbp",
    "plus_complications",
    "plus_obs",
    "ge5pct",
)


def resolve_comorbidity_binaries(cfg_data: dict[str, Any], feature_set: str) -> list[str]:
    """Return ordered unique comorbidity binary column names for a 1B feature set."""
    if feature_set not in COMORB_FEATURE_SETS:
        raise ValueError(f"unknown comorbidity feature_set={feature_set!r}")
    core = list(cfg_data["comorbidity_core"])
    if feature_set == "core":
        bins = core
    elif feature_set == "no_hbp":
        bins = [c for c in core if c != "mhoccur_hbp"]
    elif feature_set == "plus_complications":
        bins = core + list(cfg_data["comorbidity_complications"])
    elif feature_set == "plus_obs":
        bins = core + list(cfg_data["comorbidity_obs"])
    else:  # ge5pct
        bins = core + list(cfg_data.get("comorbidity_ge5pct_extra", []))
    # dedupe preserve order; strip always-exclude
    ban = set(cfg_data.get("comorbidity_exclude_always", []))
    out: list[str] = []
    for c in bins:
        if c in ban or c in out:
            continue
        out.append(c)
    if not out:
        raise AssertionError("empty comorbidity binary list")
    return out


def load_watch_onboarding_comorbidity(
    repo_root: Path,
    *,
    watch_green: str,
    onboarding: str,
    comorbidity: str,
    pool_masks: str,
    onboarding_keep: list[str],
    comorbidity_binaries: list[str],
    expected_n: int = 1824,
    id_col: str = "person_id",
    count_col: str = "comorb_count_core",
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """df, watch_cols, onboard_cols, comorb_feature_cols, all feature_cols.

    comorb_feature_cols = binaries + [count_col].
    count: sum of binary == 1 (null treated as 0 for counting only).
    """
    df, watch_cols, onboard_cols, _ = load_watch_onboarding(
        repo_root,
        watch_green=watch_green,
        onboarding=onboarding,
        pool_masks=pool_masks,
        onboarding_keep=onboarding_keep,
        expected_n=expected_n,
        id_col=id_col,
    )
    comb = pd.read_parquet(_resolve(repo_root, comorbidity))
    missing = [c for c in comorbidity_binaries if c not in comb.columns]
    if missing:
        raise ValueError(f"comorbidity missing cols: {missing}")
    keep = [id_col] + list(comorbidity_binaries)
    comb = comb[keep].copy()
    if comb[id_col].duplicated().any():
        comb = comb.drop_duplicates(id_col, keep="first")

    n_before = len(df)
    df = df.merge(comb, on=id_col, how="left")
    if len(df) != n_before:
        raise AssertionError("comorbidity merge changed row count")
    if df[id_col].isin(comb[id_col]).sum() != n_before:
        # left merge always keeps rows; check coverage
        pass
    # every core pid should appear in comorbidity table
    n_matched = int(df[id_col].isin(set(comb[id_col])).sum())
    if n_matched != n_before:
        raise AssertionError(
            f"comorbidity pid coverage {n_matched}/{n_before} (expected full match)"
        )
    covered = df[comorbidity_binaries].notna().any(axis=1).sum()
    if covered < expected_n * 0.99:
        raise AssertionError(f"comorbidity non-null coverage too low: {covered}/{expected_n}")

    # engineered count: number of yeses (null → not yes)
    mat = df[comorbidity_binaries]
    df[count_col] = (mat == 1).sum(axis=1).astype(np.int16)

    comorb_cols = list(comorbidity_binaries) + [count_col]
    feature_cols = list(watch_cols) + list(onboard_cols) + comorb_cols
    for c in feature_cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature {c}")
    return df, list(watch_cols), list(onboard_cols), comorb_cols, feature_cols


def block_tags_1b(
    watch_cols: list[str],
    onboard_cols: list[str],
    comorb_cols: list[str],
) -> dict[str, str]:
    tags = {c: "watch_green" for c in watch_cols}
    tags.update({c: "onboarding" for c in onboard_cols})
    tags.update({c: "comorbidity" for c in comorb_cols})
    return tags


MOOD_FEATURE_SETS = ("scores", "scores_via", "paid_items", "full")


def resolve_mood_cols(cfg_data: dict[str, Any], feature_set: str) -> list[str]:
    if feature_set not in MOOD_FEATURE_SETS:
        raise ValueError(f"unknown mood feature_set={feature_set!r}")
    if feature_set == "scores":
        cols = list(cfg_data["mood_scores"])
    elif feature_set == "scores_via":
        cols = list(cfg_data["mood_scores"]) + list(cfg_data["mood_via"])
    elif feature_set == "paid_items":
        # items only (no paidscore) for clean item-level SHAP — plan lock
        cols = list(cfg_data["mood_paid_items"])
    else:
        cols = (
            list(cfg_data["mood_scores"])
            + list(cfg_data["mood_ces_items"])
            + list(cfg_data["mood_paid_items"])
            + list(cfg_data["mood_via"])
        )
    out: list[str] = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def load_watch_onboarding_mood(
    repo_root: Path,
    *,
    watch_green: str,
    onboarding: str,
    mood: str,
    pool_masks: str,
    onboarding_keep: list[str],
    mood_cols: list[str],
    expected_n: int = 1824,
    id_col: str = "person_id",
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """df, watch, onboard, mood_feature_cols, all features."""
    df, watch_cols, onboard_cols, _ = load_watch_onboarding(
        repo_root,
        watch_green=watch_green,
        onboarding=onboarding,
        pool_masks=pool_masks,
        onboarding_keep=onboarding_keep,
        expected_n=expected_n,
        id_col=id_col,
    )
    md = pd.read_parquet(_resolve(repo_root, mood))
    missing = [c for c in mood_cols if c not in md.columns]
    if missing:
        raise ValueError(f"mood missing cols: {missing}")
    keep = [id_col] + list(mood_cols)
    md = md[keep].copy()
    if md[id_col].duplicated().any():
        md = md.drop_duplicates(id_col, keep="first")
    n_before = len(df)
    df = df.merge(md, on=id_col, how="left")
    if len(df) != n_before:
        raise AssertionError("mood merge changed row count")
    n_matched = int(df[id_col].isin(set(md[id_col])).sum())
    if n_matched != n_before:
        raise AssertionError(f"mood pid coverage {n_matched}/{n_before}")

    feature_cols = list(watch_cols) + list(onboard_cols) + list(mood_cols)
    for c in feature_cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature {c}")
    return df, list(watch_cols), list(onboard_cols), list(mood_cols), feature_cols


def block_tags_1c(
    watch_cols: list[str],
    onboard_cols: list[str],
    mood_cols: list[str],
) -> dict[str, str]:
    tags = {c: "watch_green" for c in watch_cols}
    tags.update({c: "onboarding" for c in onboard_cols})
    tags.update({c: "mood" for c in mood_cols})
    return tags


def load_c1_plus_comorb(
    repo_root: Path,
    *,
    watch_green: str,
    onboarding: str,
    mood: str,
    comorbidity: str,
    pool_masks: str,
    onboarding_keep: list[str],
    mood_cols: list[str],
    comorbidity_binaries: list[str],
    expected_n: int = 1824,
    id_col: str = "person_id",
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    """C1 matrix + selected comorbidity binaries (NO engineered count).

    Returns df, watch, onboard, mood, comorb_bins, all feature_cols.
    """
    df, watch_cols, onboard_cols, mood_feat, _ = load_watch_onboarding_mood(
        repo_root,
        watch_green=watch_green,
        onboarding=onboarding,
        mood=mood,
        pool_masks=pool_masks,
        onboarding_keep=onboarding_keep,
        mood_cols=mood_cols,
        expected_n=expected_n,
        id_col=id_col,
    )
    if not comorbidity_binaries:
        raise AssertionError("empty comorbidity binary list")
    comb = pd.read_parquet(_resolve(repo_root, comorbidity))
    missing = [c for c in comorbidity_binaries if c not in comb.columns]
    if missing:
        raise ValueError(f"comorbidity missing cols: {missing}")
    keep = [id_col] + list(comorbidity_binaries)
    comb = comb[keep].copy()
    if comb[id_col].duplicated().any():
        comb = comb.drop_duplicates(id_col, keep="first")

    n_before = len(df)
    df = df.merge(comb, on=id_col, how="left")
    if len(df) != n_before:
        raise AssertionError("comorbidity merge changed row count")
    n_matched = int(df[id_col].isin(set(comb[id_col])).sum())
    if n_matched != n_before:
        raise AssertionError(
            f"comorbidity pid coverage {n_matched}/{n_before} (expected full match)"
        )

    comorb_cols = list(comorbidity_binaries)
    # plan O7: no engineered count column
    if any(c.startswith("comorb_count") for c in df.columns):
        # do not auto-add; ignore if present from elsewhere
        pass
    feature_cols = (
        list(watch_cols) + list(onboard_cols) + list(mood_feat) + list(comorb_cols)
    )
    for c in feature_cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature {c}")
    return (
        df,
        list(watch_cols),
        list(onboard_cols),
        list(mood_feat),
        comorb_cols,
        feature_cols,
    )


def block_tags_wrap(
    watch_cols: list[str],
    onboard_cols: list[str] | None = None,
    mood_cols: list[str] | None = None,
    comorb_cols: list[str] | None = None,
) -> dict[str, str]:
    tags = {c: "watch_green" for c in watch_cols}
    if onboard_cols:
        tags.update({c: "onboarding" for c in onboard_cols})
    if mood_cols:
        tags.update({c: "mood" for c in mood_cols})
    if comorb_cols:
        tags.update({c: "comorbidity" for c in comorb_cols})
    return tags
