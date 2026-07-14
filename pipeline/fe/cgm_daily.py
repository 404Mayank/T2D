"""Daily + person-level CGM summary features (Path B / B1 targets).

Feature matrices contain person_id (+ day_local) + features only.
Labels/splits/sites/pool flags join at train time from meta/pool_masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.fe.local_time import day_str, load_zone_map, to_site_local
from pipeline.io import clean_path


def _read_pid(path, pid: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        t = pq.read_table(path, filters=[("person_id", "=", int(pid))])
    except Exception:
        return pd.DataFrame()
    return t.to_pandas()


def _daily_stats(g: pd.Series) -> dict:
    x = g.astype(float).to_numpy()
    n = int(len(x))
    if n == 0:
        return {
            "cgm_mean": np.nan,
            "cgm_sd": np.nan,
            "cgm_cv": np.nan,
            "cgm_min": np.nan,
            "cgm_max": np.nan,
            "cgm_tir_70_180": np.nan,
            "cgm_tbr_70": np.nan,
            "cgm_tar_180": np.nan,
            "cgm_n": 0,
        }
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    cv = float(sd / mean) if mean else 0.0
    return {
        "cgm_mean": mean,
        "cgm_sd": sd,
        "cgm_cv": cv,
        "cgm_min": float(np.min(x)),
        "cgm_max": float(np.max(x)),
        "cgm_tir_70_180": float(np.mean((x >= 70.0) & (x <= 180.0))),
        "cgm_tbr_70": float(np.mean(x < 70.0)),
        "cgm_tar_180": float(np.mean(x > 180.0)),
        "cgm_n": n,
    }


def compute_cgm_daily_for_pid(
    cfg: dict, pid: int, zone: str, min_readings: int
) -> pd.DataFrame:
    df = _read_pid(clean_path(cfg, "cgm"), pid)
    if df.empty or "blood_glucose" not in df.columns:
        return pd.DataFrame()

    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_local"
    local = to_site_local(df[ts_col], zone)
    day = day_str(local)
    bg = pd.to_numeric(df["blood_glucose"], errors="coerce")
    tmp = pd.DataFrame({"day_local": day, "blood_glucose": bg}).dropna(
        subset=["blood_glucose"]
    )
    if tmp.empty:
        return pd.DataFrame()

    rows = []
    for d, g in tmp.groupby("day_local", sort=True):
        stats = _daily_stats(g["blood_glucose"])
        stats["person_id"] = int(pid)
        stats["day_local"] = str(d)
        stats["cgm_day_valid"] = bool(stats["cgm_n"] >= min_readings)
        rows.append(stats)
    return pd.DataFrame(rows)


def aggregate_cgm_person(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per person from valid daily rows (+ overall EGV-level fallbacks)."""
    if daily.empty:
        return pd.DataFrame(columns=["person_id"])

    eight = [
        "cgm_mean",
        "cgm_sd",
        "cgm_cv",
        "cgm_min",
        "cgm_max",
        "cgm_tir_70_180",
        "cgm_tbr_70",
        "cgm_tar_180",
    ]
    rows = []
    for pid, g in daily.groupby("person_id", sort=True):
        valid = g.loc[g["cgm_day_valid"].astype(bool)]
        row = {"person_id": int(pid), "n_valid_days": int(len(valid))}
        src = valid if len(valid) else g
        for c in eight:
            row[f"{c}_daymean"] = float(src[c].mean()) if c in src and len(src) else np.nan
        # overall-ish: mean of daily means already; also expose total readings
        row["cgm_n_total"] = int(g["cgm_n"].sum()) if "cgm_n" in g else 0
        row["n_days"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def build_cgm_daily(
    cfg: dict, pool_masks: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcfg = (cfg.get("features") or {}).get("cgm_daily") or {}
    min_readings = int(fcfg.get("min_readings_per_day", 72))
    require = fcfg.get("require_pool")
    # Default: all pids that have any cleaned CGM presence flag if available
    if require and require in pool_masks.columns:
        pids = (
            pool_masks.loc[pool_masks[require].astype(bool), "person_id"]
            .astype(int)
            .tolist()
        )
    elif "has_cgm_valid" in pool_masks.columns:
        pids = (
            pool_masks.loc[pool_masks["has_cgm_valid"].astype(bool), "person_id"]
            .astype(int)
            .tolist()
        )
    else:
        pids = pool_masks["person_id"].astype(int).tolist()

    zones = load_zone_map(cfg)
    progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))
    parts: list[pd.DataFrame] = []
    for i, pid in enumerate(pids, 1):
        zone = zones.get(int(pid), "America/Los_Angeles")
        part = compute_cgm_daily_for_pid(cfg, int(pid), zone, min_readings)
        if not part.empty:
            parts.append(part)
        if progress_every and i % progress_every == 0:
            print(f"  [cgm_daily] {i}/{len(pids)}…", flush=True)

    daily = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=[
                "person_id",
                "day_local",
                "cgm_mean",
                "cgm_sd",
                "cgm_cv",
                "cgm_min",
                "cgm_max",
                "cgm_tir_70_180",
                "cgm_tbr_70",
                "cgm_tar_180",
                "cgm_n",
                "cgm_day_valid",
            ]
        )
    )
    # Column order
    if not daily.empty:
        cols = [
            "person_id",
            "day_local",
            "cgm_mean",
            "cgm_sd",
            "cgm_cv",
            "cgm_min",
            "cgm_max",
            "cgm_tir_70_180",
            "cgm_tbr_70",
            "cgm_tar_180",
            "cgm_n",
            "cgm_day_valid",
        ]
        daily = daily[cols]
    person = aggregate_cgm_person(daily)
    return daily, person
