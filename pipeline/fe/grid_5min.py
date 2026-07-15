"""5-min multi-modal aligned grid for Path B4 (View B).

person_id × bin_start_utc + channels/masks only.
Site-local ToD from UTC + zone (not parquet *_local wall clock).
Subwindow selection for fixed-length tensors lives in training/path_b/b4
(CGM-free); this module only builds the dense grid + person quality stats.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.fe.local_time import hour_float, load_zone_map, to_site_local
from pipeline.io import clean_path, write_parquet

BIN = "5min"
CHANNEL_COLS = [
    "hr",
    "stress",
    "rr",
    "steps",
    "intensity",
    "asleep",
    "cgm",
    "tod_sin",
    "tod_cos",
]
MASK_COLS = [
    "hr_bin_valid",
    "cgm_bin_valid",
    "wear_bin_valid",
    "stress_bin_valid",
    "rr_bin_valid",
]
META_BIN_COLS = ["person_id", "bin_start_utc", "n_hr", "n_cgm"]


def _read_pid(path: Path, pid: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        t = pq.read_table(path, filters=[("person_id", "=", int(pid))])
    except Exception:
        return pd.DataFrame()
    return t.to_pandas()


def _floor_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True)
    return t.dt.floor(BIN)


def _point_mean_bins(
    df: pd.DataFrame,
    value_col: str,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=["bin_start_utc", value_col, f"n_{value_col}"])
    col = ts_col if ts_col in df.columns else "timestamp_local"
    v = pd.to_numeric(df[value_col], errors="coerce")
    b = _floor_utc(df[col])
    tmp = pd.DataFrame({"bin_start_utc": b, "v": v}).dropna(subset=["v"])
    if tmp.empty:
        return pd.DataFrame(columns=["bin_start_utc", value_col, f"n_{value_col}"])
    g = tmp.groupby("bin_start_utc", sort=True)["v"]
    out = g.mean().rename(value_col).to_frame()
    out[f"n_{value_col}"] = g.size().astype(np.int32)
    return out.reset_index()


def _activity_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Sum steps and max intensity over intervals expanded to 5-min bins."""
    if df.empty or "start_time" not in df.columns:
        return pd.DataFrame(columns=["bin_start_utc", "steps", "intensity"])
    start = pd.to_datetime(df["start_time"], utc=True)
    end = pd.to_datetime(df["end_time"], utc=True)
    steps = pd.to_numeric(df.get("steps"), errors="coerce").fillna(0.0).to_numpy()
    inten = pd.to_numeric(df.get("intensity_tier"), errors="coerce").to_numpy()
    # drop invalid intervals
    ok = end > start
    start, end, steps, inten = start[ok], end[ok], steps[ok], inten[ok]
    if len(start) == 0:
        return pd.DataFrame(columns=["bin_start_utc", "steps", "intensity"])

    # Expand each interval to overlapping 5-min bins (vectorized-ish loop; n intervals moderate)
    recs: list[tuple[pd.Timestamp, float, float]] = []
    for s, e, st, it in zip(start, end, steps, inten):
        b0 = s.floor(BIN)
        # last bin that overlaps [s,e)
        b1 = (e - pd.Timedelta(microseconds=1)).floor(BIN)
        if b1 < b0:
            b1 = b0
        # distribute steps uniformly across overlapping bins
        n_bins = int((b1 - b0) / pd.Timedelta(minutes=5)) + 1
        n_bins = max(n_bins, 1)
        st_each = float(st) / n_bins
        it_val = float(it) if np.isfinite(it) else np.nan
        for k in range(n_bins):
            bk = b0 + pd.Timedelta(minutes=5 * k)
            recs.append((bk, st_each, it_val))
    if not recs:
        return pd.DataFrame(columns=["bin_start_utc", "steps", "intensity"])
    tmp = pd.DataFrame(recs, columns=["bin_start_utc", "steps", "intensity"])
    g = tmp.groupby("bin_start_utc", sort=True)
    out = pd.DataFrame(
        {
            "steps": g["steps"].sum(),
            "intensity": g["intensity"].max(),
        }
    ).reset_index()
    return out


def _sleep_asleep_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each 5-min bin spent in non-awake stages."""
    if df.empty or "start_time" not in df.columns:
        return pd.DataFrame(columns=["bin_start_utc", "asleep"])
    start = pd.to_datetime(df["start_time"], utc=True)
    end = pd.to_datetime(df["end_time"], utc=True)
    stage = df.get("sleep_stage_state")
    if stage is None:
        return pd.DataFrame(columns=["bin_start_utc", "asleep"])
    stage = stage.astype(str).str.lower()
    # non-awake = asleep for our binary channel
    is_sleep = ~stage.isin({"awake", "wake", "unknown", "nan", "none", ""})
    ok = (end > start) & is_sleep.to_numpy()
    start, end = start[ok], end[ok]
    if len(start) == 0:
        return pd.DataFrame(columns=["bin_start_utc", "asleep"])

    # accumulate seconds of sleep per bin
    acc: dict[pd.Timestamp, float] = {}
    bin_sec = 300.0
    for s, e in zip(start, end):
        b0 = s.floor(BIN)
        b1 = (e - pd.Timedelta(microseconds=1)).floor(BIN)
        if b1 < b0:
            b1 = b0
        n_bins = int((b1 - b0) / pd.Timedelta(minutes=5)) + 1
        for k in range(n_bins):
            bk = b0 + pd.Timedelta(minutes=5 * k)
            bin_end = bk + pd.Timedelta(minutes=5)
            ov0 = max(s, bk)
            ov1 = min(e, bin_end)
            sec = max((ov1 - ov0).total_seconds(), 0.0)
            acc[bk] = acc.get(bk, 0.0) + sec
    if not acc:
        return pd.DataFrame(columns=["bin_start_utc", "asleep"])
    bins = sorted(acc.keys())
    asleep = np.clip([acc[b] / bin_sec for b in bins], 0.0, 1.0)
    return pd.DataFrame({"bin_start_utc": bins, "asleep": asleep})


def choose_subwindow_start(
    wear_valid: np.ndarray,
    t_bins: int,
) -> tuple[int, int]:
    """CGM-free max wear-density contiguous window.

    Returns (start_idx, end_idx) half-open into the dense grid array.
    If len < t_bins, returns (0, len) — caller right-pads.
    Tie-break: earliest start.
    """
    w = np.asarray(wear_valid, dtype=bool)
    n = int(w.size)
    if n <= 0:
        return 0, 0
    if n <= t_bins:
        return 0, n
    # prefix sums of wear
    c = np.cumsum(w.astype(np.int32))
    # sum on [i, i+t_bins) = c[i+t_bins-1] - (c[i-1] if i else 0)
    best_i = 0
    best_s = -1
    for i in range(0, n - t_bins + 1):
        s = int(c[i + t_bins - 1] - (c[i - 1] if i > 0 else 0))
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, best_i + t_bins


def compute_grid_for_pid(
    cfg: dict,
    pid: int,
    zone: str,
    *,
    min_hr_samples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build dense 5-min grid for one person. Returns (grid_df, person_stats)."""
    hr = _read_pid(clean_path(cfg, "heart_rate"), pid)
    stress = _read_pid(clean_path(cfg, "stress"), pid)
    rr = _read_pid(clean_path(cfg, "respiratory_rate"), pid)
    act = _read_pid(clean_path(cfg, "physical_activity"), pid)
    sleep = _read_pid(clean_path(cfg, "sleep"), pid)
    cgm = _read_pid(clean_path(cfg, "cgm"), pid)

    hr_b = _point_mean_bins(hr, "heart_rate").rename(
        columns={"heart_rate": "hr", "n_heart_rate": "n_hr"}
    )
    st_b = _point_mean_bins(stress, "stress_level").rename(
        columns={"stress_level": "stress", "n_stress_level": "n_stress"}
    )
    rr_b = _point_mean_bins(rr, "respiratory_rate").rename(
        columns={"respiratory_rate": "rr", "n_respiratory_rate": "n_rr"}
    )
    cgm_b = _point_mean_bins(cgm, "blood_glucose").rename(
        columns={"blood_glucose": "cgm", "n_blood_glucose": "n_cgm"}
    )
    act_b = _activity_bins(act)
    sl_b = _sleep_asleep_bins(sleep)

    # union of bin starts from all sources that exist
    frames = [hr_b, st_b, rr_b, cgm_b, act_b, sl_b]
    starts = []
    for f in frames:
        if not f.empty and "bin_start_utc" in f.columns:
            starts.append(f["bin_start_utc"])
    if not starts:
        empty = pd.DataFrame(columns=META_BIN_COLS + CHANNEL_COLS + MASK_COLS)
        stats = {
            "person_id": int(pid),
            "n_bins": 0,
            "n_wear_valid": 0,
            "n_cgm_valid": 0,
            "n_both_valid": 0,
            "concurrent_hours": 0.0,
            "frac_cgm_with_wear": np.nan,
            "frac_wear_with_cgm": np.nan,
            "t0_utc": pd.NaT,
            "t1_utc": pd.NaT,
            "zone": zone,
        }
        return empty, stats

    t0 = min(s.min() for s in starts)
    t1 = max(s.max() for s in starts)
    # inclusive dense range
    idx = pd.date_range(t0, t1, freq=BIN, tz="UTC")
    grid = pd.DataFrame({"bin_start_utc": idx})
    for f, cols in [
        (hr_b, ["hr", "n_hr"]),
        (st_b, ["stress", "n_stress"] if "n_stress" in st_b.columns else ["stress"]),
        (rr_b, ["rr", "n_rr"] if "n_rr" in rr_b.columns else ["rr"]),
        (cgm_b, ["cgm", "n_cgm"] if "n_cgm" in cgm_b.columns else ["cgm"]),
        (act_b, ["steps", "intensity"]),
        (sl_b, ["asleep"]),
    ]:
        if f.empty:
            continue
        use = [c for c in cols if c in f.columns]
        grid = grid.merge(f[["bin_start_utc"] + use], on="bin_start_utc", how="left")

    for c, default in [
        ("hr", np.nan),
        ("stress", np.nan),
        ("rr", np.nan),
        ("steps", 0.0),
        ("intensity", np.nan),
        ("asleep", 0.0),
        ("cgm", np.nan),
        ("n_hr", 0),
        ("n_cgm", 0),
        ("n_stress", 0),
        ("n_rr", 0),
    ]:
        if c not in grid.columns:
            grid[c] = default
        else:
            if c.startswith("n_"):
                grid[c] = grid[c].fillna(0).astype(np.int32)
            elif c in ("steps", "asleep"):
                grid[c] = grid[c].fillna(0.0)

    # masks
    grid["hr_bin_valid"] = grid["n_hr"].astype(int) >= int(min_hr_samples)
    grid["cgm_bin_valid"] = grid["n_cgm"].astype(int) >= 1
    grid["wear_bin_valid"] = grid["hr_bin_valid"]
    if "n_stress" in grid.columns:
        grid["stress_bin_valid"] = grid["n_stress"].astype(int) >= 1
    else:
        grid["stress_bin_valid"] = False
    if "n_rr" in grid.columns:
        grid["rr_bin_valid"] = grid["n_rr"].astype(int) >= 1
    else:
        grid["rr_bin_valid"] = False

    # site-local ToD (always defined for every bin)
    local = to_site_local(grid["bin_start_utc"], zone)
    hour = hour_float(local).to_numpy(dtype=float)
    grid["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    grid["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)

    grid["person_id"] = int(pid)

    wear = grid["wear_bin_valid"].to_numpy(dtype=bool)
    cgm_v = grid["cgm_bin_valid"].to_numpy(dtype=bool)
    both = wear & cgm_v
    n_cgm = int(cgm_v.sum())
    n_wear = int(wear.sum())
    n_both = int(both.sum())
    stats = {
        "person_id": int(pid),
        "n_bins": int(len(grid)),
        "n_wear_valid": n_wear,
        "n_cgm_valid": n_cgm,
        "n_both_valid": n_both,
        "concurrent_hours": float(n_both * 5 / 60.0),
        "frac_cgm_with_wear": float(n_both / n_cgm) if n_cgm else np.nan,
        "frac_wear_with_cgm": float(n_both / n_wear) if n_wear else np.nan,
        "t0_utc": grid["bin_start_utc"].iloc[0] if len(grid) else pd.NaT,
        "t1_utc": grid["bin_start_utc"].iloc[-1] if len(grid) else pd.NaT,
        "zone": zone,
    }

    keep = META_BIN_COLS + CHANNEL_COLS + MASK_COLS
    # ensure all keep cols
    for c in keep:
        if c not in grid.columns:
            grid[c] = np.nan if c not in MASK_COLS else False
    grid = grid[keep].copy()
    return grid, stats


def _worker(args: tuple) -> tuple[pd.DataFrame, dict]:
    cfg, pid, zone, min_hr = args
    return compute_grid_for_pid(cfg, pid, zone, min_hr_samples=min_hr)


def build_grid_5min(
    cfg: dict,
    pool_masks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full-cohort grid + person quality table.

    Pool default: wearable_core (same as watch_daily). CGM may be empty for non-aux.
    """
    fcfg = (cfg.get("features") or {}).get("grid_5min") or {}
    min_hr = int(fcfg.get("min_hr_samples_per_bin", 2))
    require = fcfg.get("require_pool", "wearable_core")
    workers = int((cfg.get("runtime") or {}).get("fe_workers", 1) or 1)

    m = pool_masks.copy()
    m["person_id"] = m["person_id"].astype(int)
    if require and require in m.columns:
        pids = m.loc[m[require].astype(bool), "person_id"].astype(int).tolist()
    else:
        pids = m["person_id"].astype(int).tolist()

    zones = load_zone_map(cfg)

    jobs = [
        (cfg, int(pid), zones.get(int(pid), "America/Los_Angeles"), min_hr)
        for pid in pids
    ]

    grids: list[pd.DataFrame] = []
    stats_rows: list[dict] = []
    progress_every = int((cfg.get("runtime") or {}).get("progress_every", 200) or 200)

    if workers <= 1:
        for i, job in enumerate(jobs):
            g, s = _worker(job)
            if not g.empty:
                grids.append(g)
            stats_rows.append(s)
            if (i + 1) % progress_every == 0:
                print(f"  grid_5min {i+1}/{len(jobs)} pids")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, job): job[1] for job in jobs}
            done = 0
            for fut in as_completed(futs):
                g, s = fut.result()
                if not g.empty:
                    grids.append(g)
                stats_rows.append(s)
                done += 1
                if done % progress_every == 0:
                    print(f"  grid_5min {done}/{len(jobs)} pids")

    if grids:
        grid = pd.concat(grids, ignore_index=True)
    else:
        grid = pd.DataFrame(columns=META_BIN_COLS + CHANNEL_COLS + MASK_COLS)
    person = pd.DataFrame(stats_rows)
    if not person.empty:
        person = person.sort_values("person_id").reset_index(drop=True)
    return grid, person


def write_grid_outputs(cfg: dict, grid: pd.DataFrame, person: pd.DataFrame) -> dict[str, str]:
    """Write single parquet (default) or shards if configured."""
    fcfg = (cfg.get("features") or {}).get("grid_5min") or {}
    feat_dir = Path(cfg["_paths"]["features_dir"])
    shard = bool(fcfg.get("shard", False))
    paths: dict[str, str] = {}

    person_path = feat_dir / "grid_5min_person.parquet"
    write_parquet(person, person_path)
    paths["grid_5min_person"] = str(person_path)

    if shard:
        shard_dir = feat_dir / "grid_5min"
        shard_dir.mkdir(parents=True, exist_ok=True)
        # clear old shards lightly
        index_rows = []
        for pid, g in grid.groupby("person_id", sort=True):
            out = shard_dir / f"{int(pid)}.parquet"
            write_parquet(g.reset_index(drop=True), out)
            index_rows.append(
                {
                    "person_id": int(pid),
                    "path": str(out.relative_to(feat_dir)),
                    "n_bins": int(len(g)),
                }
            )
        index = pd.DataFrame(index_rows)
        idx_path = feat_dir / "grid_5min_index.parquet"
        write_parquet(index, idx_path)
        paths["grid_5min_index"] = str(idx_path)
        paths["grid_5min_dir"] = str(shard_dir)
    else:
        out = feat_dir / "grid_5min.parquet"
        write_parquet(grid, out)
        paths["grid_5min"] = str(out)
    return paths
