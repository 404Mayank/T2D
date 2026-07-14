"""Site-correct local wall-clock helpers for Path B daily FE.

Cleaned `*_local` columns are stored as a single parquet timezone
(`America/Los_Angeles`) for all pids. UTC instants are correct; UAB wall
clock must be re-derived via `zone` from `meta/shared_windows.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_zone_map(cfg: dict) -> dict[int, str]:
    path = Path(cfg["_paths"]["meta_dir"]) / "shared_windows.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run pipeline.run_clean first")
    sw = pd.read_parquet(path, columns=["person_id", "zone"])
    return {
        int(pid): str(z) if pd.notna(z) else "America/Los_Angeles"
        for pid, z in zip(sw["person_id"].astype(int), sw["zone"])
    }


def to_site_local(ts: pd.Series, zone: str) -> pd.Series:
    """Convert a UTC (or tz-aware) timestamp series to site-local wall clock."""
    t = pd.to_datetime(ts, utc=True)
    return t.dt.tz_convert(zone)


def day_str(ts_local: pd.Series) -> pd.Series:
    """Civil day as YYYY-MM-DD (avoids floor('D') DST traps)."""
    return ts_local.dt.strftime("%Y-%m-%d")


def hour_float(ts_local: pd.Series) -> pd.Series:
    return ts_local.dt.hour + ts_local.dt.minute / 60.0 + ts_local.dt.second / 3600.0
