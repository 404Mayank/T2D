"""UTC→local conversion and wear-window selection.

Windows are computed with timezone-aware Timestamps only (no int64 unit
assumptions). Garmin/dexcom may be datetime64[ms, tz]; casting to int64 yields
milliseconds — never pass those ints to pd.Timestamp() without unit=.
"""

from __future__ import annotations

import pandas as pd


def localize_series(ts: pd.Series, zone: str) -> pd.Series:
    """Convert timestamps to timezone-aware local time."""
    t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(zone)


def pick_window_bounds(
    ts_local: pd.Series,
    policy: str,
    days: float,
    long_wear_days: float,
    force_all_to_window: bool = False,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, dict]:
    """Return [start, end) local bounds for the analysis window.

    Policies:
      - all: full span (ignore days/long_wear)
      - first: first `days` from first sample (if truncating)
      - best_coverage: contiguous `days` window maximizing sample count

    Truncation triggers when span > long_wear_days, OR when force_all_to_window
    is True (everyone sliced to `days` for cross-pid consistency).
    """
    info: dict = {
        "policy_applied": policy,
        "truncated": False,
        "span_days": 0.0,
        "force_all_to_window": bool(force_all_to_window),
    }
    if ts_local is None or len(ts_local) == 0:
        return None, None, info

    t = pd.to_datetime(ts_local)
    # Drop NaT
    t = t[t.notna()]
    if len(t) == 0:
        return None, None, info

    t0 = t.min()
    t1 = t.max()
    span_days = (t1 - t0).total_seconds() / 86400.0
    info["span_days"] = float(span_days)

    if policy == "all":
        return t0, t1 + pd.Timedelta(microseconds=1), info

    should_truncate = bool(force_all_to_window) or (span_days > float(long_wear_days))
    if not should_truncate:
        return t0, t1 + pd.Timedelta(microseconds=1), info

    info["truncated"] = True
    win = pd.Timedelta(days=float(days))

    if policy == "first":
        return t0, t0 + win, info

    # best_coverage: two-pointer on sorted timestamps (Timestamp arithmetic only)
    t_sorted = t.sort_values().reset_index(drop=True)
    n = len(t_sorted)
    best_i, best_cnt = 0, 1
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < n and (t_sorted.iloc[j + 1] - t_sorted.iloc[i]) < win:
            j += 1
        cnt = j - i + 1
        if cnt > best_cnt:
            best_cnt = cnt
            best_i = i
    start = t_sorted.iloc[best_i]
    end = start + win
    info["best_count"] = int(best_cnt)
    return start, end, info


def apply_window(
    df: pd.DataFrame,
    ts_col: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    if df.empty or start is None or end is None:
        return df
    t = pd.to_datetime(df[ts_col])
    # Align tz if needed
    if getattr(start, "tz", None) is not None and t.dt.tz is None:
        t = t.dt.tz_localize(start.tz)
    elif getattr(start, "tz", None) is not None and str(t.dt.tz) != str(start.tz):
        try:
            t = t.dt.tz_convert(start.tz)
        except Exception:
            pass
    return df.loc[(t >= start) & (t < end)].copy()


def add_local_time(df: pd.DataFrame, ts_cols: list[str], zone: str) -> pd.DataFrame:
    df = df.copy()
    for c in ts_cols:
        if c not in df.columns:
            continue
        if c == "timestamp":
            local_col = "timestamp_local"
        elif c == "start_time":
            local_col = "start_time_local"
        elif c == "end_time":
            local_col = "end_time_local"
        else:
            local_col = f"{c}_local"
        df[local_col] = localize_series(df[c], zone)
    return df
