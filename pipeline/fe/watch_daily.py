"""Daily wearable feature matrix for Path B sequence backbones (B1+).

person_id × day_local + features only. Site-local civil day/hour from UTC + zone.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.fe.local_time import day_str, hour_float, load_zone_map, to_site_local
from pipeline.io import clean_path


def _read_pid(path, pid: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        t = pq.read_table(path, filters=[("person_id", "=", int(pid))])
    except Exception:
        return pd.DataFrame()
    return t.to_pandas()


def _agg_mean_sd_min_max_n(x: np.ndarray) -> tuple[float, float, float, float, int]:
    n = int(x.size)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(x.mean())
    sd = float(x.std(ddof=0)) if n > 1 else 0.0
    return mean, sd, float(x.min()), float(x.max()), n


def _hr_daily(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    if df.empty or "heart_rate" not in df.columns:
        return pd.DataFrame()
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_local"
    local = to_site_local(df[ts_col], zone)
    hr = pd.to_numeric(df["heart_rate"], errors="coerce").to_numpy(dtype=float)
    day = day_str(local).to_numpy()
    hour = hour_float(local).to_numpy(dtype=float)
    mask = np.isfinite(hr)
    day, hour, hr = day[mask], hour[mask], hr[mask]
    if hr.size == 0:
        return pd.DataFrame()

    # sort by day for contiguous groups
    order = np.argsort(day, kind="mergesort")
    day, hour, hr = day[order], hour[order], hr[order]
    # group boundaries
    change = np.flatnonzero(day[1:] != day[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, day.size]

    rows = []
    for a, b in zip(starts, ends):
        x = hr[a:b]
        h = hour[a:b]
        mean, sd, mn, mx, n = _agg_mean_sd_min_max_n(x)
        night = x[(h >= 0) & (h < 6)]
        dayp = x[(h >= 8) & (h < 20)]
        rows.append(
            {
                "day_local": str(day[a]),
                "hr_mean": mean,
                "hr_sd": sd,
                "hr_min": mn,
                "hr_max": mx,
                "hr_n": n,
                "hr_nocturnal_mean": float(night.mean()) if night.size else np.nan,
                "hr_day_mean": float(dayp.mean()) if dayp.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _stress_daily(df: pd.DataFrame, zone: str, cfg: dict) -> pd.DataFrame:
    if df.empty or "stress_level" not in df.columns:
        return pd.DataFrame()
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_local"
    local = to_site_local(df[ts_col], zone)
    x = pd.to_numeric(df["stress_level"], errors="coerce").to_numpy(dtype=float)
    day = day_str(local).to_numpy()
    med = float(cfg["sentinels"]["stress_medium"])
    hi = float(cfg["sentinels"]["stress_high"])
    mask = np.isfinite(x)
    day, x = day[mask], x[mask]
    if x.size == 0:
        return pd.DataFrame()
    order = np.argsort(day, kind="mergesort")
    day, x = day[order], x[order]
    change = np.flatnonzero(day[1:] != day[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, day.size]
    rows = []
    for a, b in zip(starts, ends):
        s = x[a:b]
        mean, sd, _, _, n = _agg_mean_sd_min_max_n(s)
        rows.append(
            {
                "day_local": str(day[a]),
                "stress_mean": mean,
                "stress_sd": sd,
                "stress_pct_medium_plus": float((s >= med).mean()),
                "stress_pct_high": float((s >= hi).mean()),
                "stress_n": n,
            }
        )
    return pd.DataFrame(rows)


def _rr_daily(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    if df.empty or "respiratory_rate" not in df.columns:
        return pd.DataFrame()
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_local"
    local = to_site_local(df[ts_col], zone)
    x = pd.to_numeric(df["respiratory_rate"], errors="coerce").to_numpy(dtype=float)
    day = day_str(local).to_numpy()
    mask = np.isfinite(x)
    day, x = day[mask], x[mask]
    if x.size == 0:
        return pd.DataFrame()
    order = np.argsort(day, kind="mergesort")
    day, x = day[order], x[order]
    change = np.flatnonzero(day[1:] != day[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, day.size]
    rows = []
    for a, b in zip(starts, ends):
        s = x[a:b]
        mean, sd, _, _, n = _agg_mean_sd_min_max_n(s)
        rows.append(
            {
                "day_local": str(day[a]),
                "rr_mean": mean,
                "rr_sd": sd,
                "rr_n": n,
            }
        )
    return pd.DataFrame(rows)


def _sleep_daily(df: pd.DataFrame, zone: str, gap_min: float = 30.0) -> pd.DataFrame:
    """Onset-date sleep sessions; duration via Timedelta (unit-safe).

    sleep_n_bouts = number of *sessions* per onset day (gap ≥ gap_min),
    not stage rows. Duration = sum of non-awake bout lengths; all-awake
    sessions contribute no duration (day NaN if no asleep time).
    """
    if df.empty:
        return pd.DataFrame()
    start_c = "start_time" if "start_time" in df.columns else "start_time_local"
    end_c = "end_time" if "end_time" in df.columns else "end_time_local"
    if start_c not in df.columns or end_c not in df.columns:
        return pd.DataFrame()

    s = to_site_local(df[start_c], zone)
    e = to_site_local(df[end_c], zone)
    stage = (
        df["sleep_stage_state"].astype(str).to_numpy()
        if "sleep_stage_state" in df.columns
        else np.array(["unknown"] * len(df))
    )
    # Sort key only — unit of int64 is irrelevant for ordering.
    order = np.argsort(s.astype("int64").to_numpy(), kind="mergesort")
    s = s.iloc[order].reset_index(drop=True)
    e = e.iloc[order].reset_index(drop=True)
    stage = stage[order]
    n = len(s)
    if n == 0:
        return pd.DataFrame()

    # Unit-invariant: works for datetime64[ms] and [ns] (C1 fix).
    dur_h = (e - s).dt.total_seconds().to_numpy(dtype=float) / 3600.0

    sess = np.empty(n, dtype=np.int32)
    sess[0] = 0
    sid = 0
    for i in range(1, n):
        gap_m = (s.iloc[i] - e.iloc[i - 1]).total_seconds() / 60.0
        if gap_m >= gap_min:
            sid += 1
        sess[i] = sid
    asleep = np.array([str(x).lower() != "awake" for x in stage])
    onset_day = day_str(s).to_numpy()

    from collections import defaultdict

    day_dur: dict[str, float] = defaultdict(float)
    day_bouts: dict[str, int] = defaultdict(int)
    day_has_asleep: dict[str, bool] = defaultdict(bool)

    n_sess = int(sess.max()) + 1
    for si in range(n_sess):
        idx = np.flatnonzero(sess == si)
        if idx.size == 0:
            continue
        day = str(onset_day[idx[0]])
        day_bouts[day] += 1  # one count per session, not per stage row
        a_mask = asleep[idx]
        asleep_sum = float(dur_h[idx][a_mask].sum()) if a_mask.any() else 0.0
        if asleep_sum > 0.0:
            day_dur[day] += asleep_sum
            day_has_asleep[day] = True

    rows = [
        {
            "day_local": d,
            # All-awake day → NaN (impute later), not a true 0 h sleep.
            "sleep_duration_hours": day_dur[d] if day_has_asleep[d] else np.nan,
            "sleep_n_bouts": day_bouts[d],
        }
        for d in sorted(day_bouts)
    ]
    return pd.DataFrame(rows)


def _activity_daily(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    start_c = "start_time" if "start_time" in df.columns else "start_time_local"
    if start_c not in df.columns:
        return pd.DataFrame()
    local = to_site_local(df[start_c], zone)
    day = day_str(local).to_numpy()
    if "duration_minutes" in df.columns:
        dur = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        end_c = "end_time" if "end_time" in df.columns else "end_time_local"
        end_local = to_site_local(df[end_c], zone)
        dur = ((end_local - local).dt.total_seconds() / 60.0).fillna(0.0).to_numpy(dtype=float)
    steps = (
        pd.to_numeric(df["steps"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "steps" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    tier = (
        pd.to_numeric(df["intensity_tier"], errors="coerce").to_numpy(dtype=float)
        if "intensity_tier" in df.columns
        else np.full(len(df), np.nan)
    )
    order = np.argsort(day, kind="mergesort")
    day, dur, steps, tier = day[order], dur[order], steps[order], tier[order]
    change = np.flatnonzero(day[1:] != day[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, day.size]
    rows = []
    for a, b in zip(starts, ends):
        t = tier[a:b]
        d = dur[a:b]
        if np.isfinite(t).any():
            mvpa = float(d[t == 2].sum()) if (t == 2).any() else 0.0
            light = float(d[t == 1].sum()) if (t == 1).any() else 0.0
            sed = float(d[t == 0].sum()) if (t == 0).any() else 0.0
        else:
            mvpa = light = sed = np.nan
        rows.append(
            {
                "day_local": str(day[a]),
                "steps_sum": float(steps[a:b].sum()),
                "mvpa_min": mvpa,
                "light_min": light,
                "sedentary_min": sed,
            }
        )
    return pd.DataFrame(rows)


def compute_watch_daily_for_pid(
    cfg: dict, pid: int, zone: str, min_hr: int
) -> pd.DataFrame:
    hr = _hr_daily(_read_pid(clean_path(cfg, "heart_rate"), pid), zone)
    if hr.empty:
        return pd.DataFrame()

    st = _stress_daily(_read_pid(clean_path(cfg, "stress"), pid), zone, cfg)
    rr = _rr_daily(_read_pid(clean_path(cfg, "respiratory_rate"), pid), zone)
    sl = _sleep_daily(_read_pid(clean_path(cfg, "sleep"), pid), zone)
    ac = _activity_daily(_read_pid(clean_path(cfg, "physical_activity"), pid), zone)

    out = hr
    for other in (st, rr, sl, ac):
        if other is not None and not other.empty:
            out = out.merge(other, on="day_local", how="left")

    for c, default in [
        ("stress_mean", np.nan),
        ("stress_sd", np.nan),
        ("stress_pct_medium_plus", np.nan),
        ("stress_pct_high", np.nan),
        ("stress_n", 0),
        ("rr_mean", np.nan),
        ("rr_sd", np.nan),
        ("rr_n", 0),
        ("sleep_duration_hours", np.nan),
        ("sleep_n_bouts", 0),
        ("steps_sum", np.nan),
        ("mvpa_min", np.nan),
        ("light_min", np.nan),
        ("sedentary_min", np.nan),
    ]:
        if c not in out.columns:
            out[c] = default

    out["person_id"] = int(pid)
    out["watch_day_valid"] = out["hr_n"].astype(int) >= int(min_hr)
    # count-like columns: missing modality day → 0 not NaN
    for c in ("stress_n", "rr_n", "sleep_n_bouts"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    cols = [
        "person_id",
        "day_local",
        "hr_mean",
        "hr_sd",
        "hr_min",
        "hr_max",
        "hr_n",
        "hr_nocturnal_mean",
        "hr_day_mean",
        "stress_mean",
        "stress_sd",
        "stress_pct_medium_plus",
        "stress_pct_high",
        "stress_n",
        "rr_mean",
        "rr_sd",
        "rr_n",
        "sleep_duration_hours",
        "sleep_n_bouts",
        "steps_sum",
        "mvpa_min",
        "light_min",
        "sedentary_min",
        "watch_day_valid",
    ]
    return out[cols]


def _worker_watch(args: tuple) -> pd.DataFrame:
    """Process-pool worker: (cfg_paths_dict-like cfg, pid, zone, min_hr)."""
    cfg, pid, zone, min_hr = args
    return compute_watch_daily_for_pid(cfg, int(pid), zone, int(min_hr))


def build_watch_daily(cfg: dict, pool_masks: pd.DataFrame) -> pd.DataFrame:
    fcfg = (cfg.get("features") or {}).get("watch_daily") or {}
    min_hr = int(fcfg.get("min_hr_minutes", 60))
    require = fcfg.get("require_pool") or "wearable_core"
    if require and require in pool_masks.columns:
        pids = (
            pool_masks.loc[pool_masks[require].astype(bool), "person_id"]
            .astype(int)
            .tolist()
        )
    else:
        pids = pool_masks["person_id"].astype(int).tolist()

    zones = load_zone_map(cfg)
    progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))
    n_workers = int((cfg.get("runtime") or {}).get("fe_workers") or 4)

    jobs = [(cfg, int(pid), zones.get(int(pid), "America/Los_Angeles"), min_hr) for pid in pids]

    parts: list[pd.DataFrame] = []
    if n_workers <= 1 or len(jobs) < 8:
        for i, job in enumerate(jobs, 1):
            part = _worker_watch(job)
            if not part.empty:
                parts.append(part)
            if progress_every and i % progress_every == 0:
                print(f"  [watch_daily] {i}/{len(jobs)}…", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_worker_watch, job) for job in jobs]
            for fut in as_completed(futs):
                part = fut.result()
                if part is not None and not part.empty:
                    parts.append(part)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  [watch_daily] {done}/{len(jobs)}…", flush=True)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
