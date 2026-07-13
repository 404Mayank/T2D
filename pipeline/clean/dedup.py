"""Deduplicate series rows."""

from __future__ import annotations

import pandas as pd


def dedup_instant(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    value_cols: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and timestamp collisions for point-sampled series."""
    stats = {"n_in": len(df), "exact_dups": 0, "ts_dups": 0}
    if df.empty:
        stats["n_out"] = 0
        return df, stats

    before = len(df)
    df = df.drop_duplicates()
    stats["exact_dups"] = before - len(df)

    if ts_col not in df.columns:
        stats["n_out"] = len(df)
        return df, stats

    # timestamp-level dups
    subset = ["person_id", ts_col] if "person_id" in df.columns else [ts_col]
    before = len(df)
    if keep == "last":
        df = df.drop_duplicates(subset=subset, keep="last")
    elif keep == "mean" and value_cols:
        # group mean for numeric value cols, first for others
        num = [c for c in value_cols if c in df.columns]
        others = [c for c in df.columns if c not in num and c not in subset]
        agg = {c: "mean" for c in num}
        for c in others:
            agg[c] = "first"
        df = df.groupby(subset, as_index=False).agg(agg)
    else:
        df = df.drop_duplicates(subset=subset, keep="first")
    stats["ts_dups"] = before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def dedup_interval(
    df: pd.DataFrame,
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> tuple[pd.DataFrame, dict]:
    stats = {"n_in": len(df), "exact_dups": 0, "ts_dups": 0}
    if df.empty:
        stats["n_out"] = 0
        return df, stats
    before = len(df)
    df = df.drop_duplicates()
    stats["exact_dups"] = before - len(df)
    subset = ["person_id", start_col, end_col]
    subset = [c for c in subset if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    stats["ts_dups"] = before - len(df)
    stats["n_out"] = len(df)
    return df, stats
