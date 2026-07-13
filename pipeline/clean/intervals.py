"""Interval modality cleaning (sleep, physical_activity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clean.dedup import dedup_interval
from pipeline.constants import activity_intensity


def clean_sleep(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    df, dstats = dedup_interval(df)
    stats["dedup"] = dstats
    if df.empty:
        stats["n_out"] = 0
        return df, stats

    icfg = cfg["intervals"]["sleep"]
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    dur_s = (df["end_time"] - df["start_time"]).dt.total_seconds()
    min_s = float(icfg["min_duration_seconds"])
    max_h = float(icfg["max_duration_hours"])
    ok = (dur_s >= min_s) & (dur_s <= max_h * 3600) & (df["end_time"] >= df["start_time"])
    stats["bad_interval_dropped"] = int((~ok).sum())
    df = df.loc[ok].copy()

    if icfg.get("drop_unknown_stage", True) and "sleep_stage_state" in df.columns:
        unk = df["sleep_stage_state"].astype(str).str.lower() == "unknown"
        stats["unknown_stage_dropped"] = int(unk.sum())
        df = df.loc[~unk].copy()
    else:
        stats["unknown_stage_dropped"] = 0

    stats["n_out"] = len(df)
    return df, stats


def clean_activity(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    df, dstats = dedup_interval(df)
    stats["dedup"] = dstats
    if df.empty:
        stats["n_out"] = 0
        return df, stats

    icfg = cfg["intervals"]["activity"]
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    dur_s = (df["end_time"] - df["start_time"]).dt.total_seconds()
    min_s = float(icfg["min_duration_seconds"])
    max_h = float(icfg["max_duration_hours"])
    ok = (dur_s >= min_s) & (dur_s <= max_h * 3600) & (df["end_time"] >= df["start_time"])
    stats["bad_interval_dropped"] = int((~ok).sum())
    df = df.loc[ok].copy()

    blank_as = icfg.get("blank_name_as", "unknown")
    name = df["activity_name"].astype(str).replace({"nan": "", "None": ""})
    name = name.str.strip()
    name = name.mask(name == "", blank_as)
    df["activity_name"] = name
    # float64 always — avoids pyarrow null-type schema lock if first pid is all-unknown
    df["intensity_tier"] = pd.Series(
        [activity_intensity(n, cfg) for n in df["activity_name"]],
        index=df.index,
        dtype="float64",
    )
    df["duration_minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0

    stats["n_out"] = len(df)
    return df, stats


INTERVAL_CLEANERS = {
    "sleep": clean_sleep,
    "physical_activity": clean_activity,
}
