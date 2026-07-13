"""GREEN wearable summary features (Path A floor).

Feature matrices contain person_id + features only. Labels/splits/sites/pool
flags live in meta/pool_masks.parquet and are joined at train time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.io import clean_path


def _read_pid(path, pid: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        t = pq.read_table(path, filters=[("person_id", "=", int(pid))])
    except Exception:
        return pd.DataFrame()
    return t.to_pandas()


def _hr_features(df: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    if df.empty or "heart_rate" not in df.columns:
        return out
    hr = df["heart_rate"].astype(float)
    out["hr_mean"] = float(hr.mean())
    out["hr_sd"] = float(hr.std(ddof=0)) if len(hr) > 1 else 0.0
    out["hr_cv"] = float(out["hr_sd"] / out["hr_mean"]) if out["hr_mean"] else np.nan
    out["hr_min"] = float(hr.min())
    out["hr_max"] = float(hr.max())
    out["hr_range"] = out["hr_max"] - out["hr_min"]
    out["hr_n"] = int(len(hr))

    ts = df["timestamp_local"] if "timestamp_local" in df.columns else df["timestamp"]
    ts = pd.to_datetime(ts)
    hours = ts.dt.hour + ts.dt.minute / 60.0
    day = hr[(hours >= 8) & (hours < 20)]
    night = hr[(hours >= 0) & (hours < 6)]
    if len(day) and len(night):
        out["hr_nocturnal_dip"] = float(day.mean() - night.mean())
    else:
        out["hr_nocturnal_dip"] = np.nan

    h0, h1 = cfg["time"]["rhr_local_hours"]
    mask = (hours >= h0) & (hours < h1)
    sub = df.loc[mask].copy()
    if len(sub) >= 5:
        tcol = "timestamp_local" if "timestamp_local" in sub.columns else "timestamp"
        sub = sub.sort_values(tcol)
        sub = sub.set_index(pd.to_datetime(sub[tcol])).sort_index()
        roll = sub["heart_rate"].rolling("30min", min_periods=10).mean()
        out["rhr"] = float(roll.min()) if roll.notna().any() else np.nan
    else:
        out["rhr"] = np.nan
    return out


def _stress_features(df: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    if df.empty or "stress_level" not in df.columns:
        return out
    x = df["stress_level"].astype(float)
    med = float(cfg["sentinels"]["stress_medium"])
    hi = float(cfg["sentinels"]["stress_high"])
    out["stress_mean"] = float(x.mean())
    out["stress_sd"] = float(x.std(ddof=0)) if len(x) > 1 else 0.0
    # Denominator = valid (rest-only) stress samples — Garmin stress is not 24h
    out["stress_pct_medium_plus"] = float((x >= med).mean())
    out["stress_pct_high"] = float((x >= hi).mean())
    out["stress_n"] = int(len(x))

    ts = df["timestamp_local"] if "timestamp_local" in df.columns else df["timestamp"]
    ts = pd.to_datetime(ts)
    hours = ts.dt.hour
    night = x[(hours >= 0) & (hours < 6)]
    out["stress_nocturnal_mean"] = float(night.mean()) if len(night) else np.nan
    return out


def _sleep_features(df: pd.DataFrame, cfg: dict) -> dict:
    out = {
        "sri": np.nan,
        "sleep_onset_sd_hours": np.nan,
        "sleep_duration_mean_hours": np.nan,
        "sleep_duration_dev_7_5": np.nan,
        "sleep_short_frac": np.nan,
        "sleep_long_frac": np.nan,
        "sleep_n_nights": 0,
    }
    if df.empty:
        return out

    start_c = "start_time_local" if "start_time_local" in df.columns else "start_time"
    end_c = "end_time_local" if "end_time_local" in df.columns else "end_time"
    df = df.copy()
    df[start_c] = pd.to_datetime(df[start_c])
    df[end_c] = pd.to_datetime(df[end_c])

    df["mid"] = df[start_c] + (df[end_c] - df[start_c]) / 2
    # strftime avoids DST ambiguous floor('D') failures on fall-back nights
    df["night_id"] = pd.to_datetime(df["mid"]).dt.strftime("%Y-%m-%d")

    stage = (
        df["sleep_stage_state"].astype(str).str.lower()
        if "sleep_stage_state" in df.columns
        else None
    )
    asleep = df.loc[stage != "awake"].copy() if stage is not None else df.copy()
    if asleep.empty:
        return out

    asleep["dur_h"] = (asleep[end_c] - asleep[start_c]).dt.total_seconds() / 3600.0
    night_dur = asleep.groupby("night_id")["dur_h"].sum()
    out["sleep_n_nights"] = int(len(night_dur))
    if len(night_dur):
        mean_d = float(night_dur.mean())
        out["sleep_duration_mean_hours"] = mean_d
        target = float(cfg["features"]["watch_green"].get("sleep_target_hours", 7.5))
        out["sleep_duration_dev_7_5"] = abs(mean_d - target)
        out["sleep_short_frac"] = float((night_dur < 7.0).mean())
        out["sleep_long_frac"] = float((night_dur > 8.0).mean())

    onset_df = df.loc[stage != "awake"] if stage is not None else df
    if len(onset_df):
        onset = onset_df.groupby("night_id")[start_c].min()
        mins = onset.dt.hour * 60 + onset.dt.minute + onset.dt.second / 60.0
        mins_adj = mins.where(mins >= 12 * 60, mins + 24 * 60)
        out["sleep_onset_sd_hours"] = (
            float(mins_adj.std(ddof=0) / 60.0) if len(mins_adj) > 1 else 0.0
        )

    try:
        out["sri"] = _compute_sri_adjacent(df, start_c, end_c, stage)
    except Exception:
        out["sri"] = np.nan
    return out


def _compute_sri_adjacent(
    df: pd.DataFrame, start_c: str, end_c: str, stage: pd.Series | None
) -> float:
    """Phillips 2017-style SRI: mean agreement between *adjacent* day pairs (0–100)."""
    if stage is not None:
        asleep = df.loc[stage.to_numpy() != "awake", [start_c, end_c]]
    else:
        asleep = df[[start_c, end_c]]
    if asleep.empty:
        return np.nan

    smin = pd.Timestamp(asleep[start_c].min())
    emax = pd.Timestamp(asleep[end_c].max())
    # Normalize to UTC for arithmetic grid only (SRI matrix indexing)
    if smin.tzinfo is not None:
        smin = smin.tz_convert("UTC").tz_localize(None)
        emax = emax.tz_convert("UTC").tz_localize(None)
    t0 = smin.floor("D")
    t1 = emax.ceil("D")
    n_days = int((t1 - t0).days)
    if n_days < 2:
        return np.nan
    n_days = min(n_days, 21)
    mat = np.zeros((n_days, 1440), dtype=np.uint8)

    starts = asleep[start_c].to_numpy()
    ends = asleep[end_c].to_numpy()
    for s, e in zip(starts, ends):
        s = pd.Timestamp(s)
        e = pd.Timestamp(e)
        if s.tzinfo is not None:
            s = s.tz_convert("UTC").tz_localize(None)
            e = e.tz_convert("UTC").tz_localize(None)
        s_min = int((s - t0).total_seconds() // 60)
        e_min = int((e - t0).total_seconds() // 60)
        if e_min <= s_min:
            continue
        s_min = max(s_min, 0)
        e_min = min(e_min, n_days * 1440)
        for mabs in range(s_min, e_min):
            d, m = divmod(mabs, 1440)
            if 0 <= d < n_days:
                mat[d, m] = 1

    # Adjacent pairs only (Phillips); skip pairs where both days empty
    agreements = []
    for i in range(n_days - 1):
        if mat[i].sum() == 0 and mat[i + 1].sum() == 0:
            continue
        agreements.append(float((mat[i] == mat[i + 1]).mean()))
    if not agreements:
        return np.nan
    sri = 200.0 * (float(np.mean(agreements)) - 0.5)
    return float(np.clip(sri, 0, 100))


def _activity_features(df: pd.DataFrame, cfg: dict, wear_days: float | None = None) -> dict:
    """MVPA / sedentary minutes per wear-day.

    Denominator defaults to max(unique activity calendar days, wear_days from HR).
    Sedentary = sum of Garmin sedentary-tier intervals (not 1440-active).
    """
    out = {
        "mvpa_min_per_day": np.nan,
        "light_min_per_day": np.nan,
        "sedentary_min_per_day": np.nan,
        "steps_mean_per_day": np.nan,
        "activity_n_days": 0,
    }
    if df.empty:
        return out
    start_c = "start_time_local" if "start_time_local" in df.columns else "start_time"
    df = df.copy()
    df[start_c] = pd.to_datetime(df[start_c])
    if "duration_minutes" not in df.columns:
        end_c = "end_time_local" if "end_time_local" in df.columns else "end_time"
        df["duration_minutes"] = (
            pd.to_datetime(df[end_c]) - df[start_c]
        ).dt.total_seconds() / 60.0
    df["day"] = pd.to_datetime(df[start_c]).dt.strftime("%Y-%m-%d")
    act_days = int(df["day"].nunique())
    # Prefer HR wear-day denominator when provided (more stable across pids)
    denom = float(wear_days) if wear_days and wear_days > 0 else float(max(act_days, 1))
    out["activity_n_days"] = int(round(denom))

    tier = df["intensity_tier"] if "intensity_tier" in df.columns else None
    if tier is not None:
        out["sedentary_min_per_day"] = float(
            df.loc[tier == 0, "duration_minutes"].sum() / denom
        )
        out["light_min_per_day"] = float(df.loc[tier == 1, "duration_minutes"].sum() / denom)
        out["mvpa_min_per_day"] = float(df.loc[tier == 2, "duration_minutes"].sum() / denom)
    if "steps" in df.columns:
        steps = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
        out["steps_mean_per_day"] = float(steps.sum() / denom)
    return out


def _rar_features(hr_df: pd.DataFrame) -> dict:
    """Cosinor on hourly mean HR using *real local hours*, not row index.

    Model: y = mesor + A cos(w t) + B sin(w t)
         = mesor + Amp cos(w t - φ),  φ = atan2(B, A)
    Peak (acrophase) at t = φ / w  (hours), not -φ/w.
    """
    out = {"rar_amplitude": np.nan, "rar_mesor": np.nan, "rar_acrophase_hour": np.nan}
    if hr_df.empty or "heart_rate" not in hr_df.columns:
        return out
    ts = hr_df["timestamp_local"] if "timestamp_local" in hr_df.columns else hr_df["timestamp"]
    ts = pd.to_datetime(ts)
    tmp = pd.DataFrame({"hr": hr_df["heart_rate"].astype(float).to_numpy(), "ts": ts})
    # Hourly means keyed by real timestamps
    g = tmp.set_index("ts")["hr"].groupby(pd.Grouper(freq="h")).mean().dropna()
    if len(g) < 24:
        return out
    y = g.to_numpy(dtype=float)
    # Hours since first sample (real gaps preserved as missing hours already dropped;
    # use absolute local hour-of-day continuum via timestamp)
    t0 = g.index[0]
    t_hours = np.array([(ix - t0).total_seconds() / 3600.0 for ix in g.index], dtype=float)
    w = 2 * np.pi / 24.0
    X = np.column_stack([np.ones(len(y)), np.cos(w * t_hours), np.sin(w * t_hours)])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        mesor, A, B = coef
        amp = float(np.hypot(A, B))
        phase = float(np.arctan2(B, A))  # φ in radians
        # peak at +φ/w modulo 24h, expressed as local hour-of-day of first sample + offset
        acro_offset_h = (phase / w) % 24.0
        # convert to local clock hour: first sample's local hour + offset
        t0_hour = t0.hour + t0.minute / 60.0 + t0.second / 3600.0
        acro_hour = (t0_hour + acro_offset_h) % 24.0
        out["rar_mesor"] = float(mesor)
        out["rar_amplitude"] = amp
        out["rar_acrophase_hour"] = float(acro_hour)
    except Exception:
        pass
    return out


def compute_watch_green_for_pid(cfg: dict, pid: int) -> dict:
    feat = {"person_id": int(pid)}
    fcfg = cfg["features"]["watch_green"]
    wear_days = None

    hr = pd.DataFrame()
    if fcfg.get("hr_stats", True) or fcfg.get("rhr", True) or fcfg.get("rar_cosinor", True):
        hr = _read_pid(clean_path(cfg, "heart_rate"), pid)
        if fcfg.get("hr_stats", True) or fcfg.get("rhr", True):
            feat.update(_hr_features(hr, cfg))
        if fcfg.get("rar_cosinor", True):
            feat.update(_rar_features(hr))
        if not hr.empty:
            ts = hr["timestamp_local"] if "timestamp_local" in hr.columns else hr["timestamp"]
            ts = pd.to_datetime(ts)
            wear_days = float(ts.dt.strftime("%Y-%m-%d").nunique())

    if fcfg.get("stress_stats", True):
        st = _read_pid(clean_path(cfg, "stress"), pid)
        feat.update(_stress_features(st, cfg))

    if fcfg.get("sleep_sri", True) or fcfg.get("sleep_duration", True):
        sl = _read_pid(clean_path(cfg, "sleep"), pid)
        feat.update(_sleep_features(sl, cfg))

    if fcfg.get("activity_mvpa", True):
        ac = _read_pid(clean_path(cfg, "physical_activity"), pid)
        feat.update(_activity_features(ac, cfg, wear_days=wear_days))

    return feat


def build_watch_green(cfg: dict, pool_masks: pd.DataFrame) -> pd.DataFrame:
    """Return person_id + GREEN features only (no label/split/site/pool flags)."""
    fcfg = cfg["features"]["watch_green"]
    require = fcfg.get("require_pool") or None
    if require and require in pool_masks.columns:
        pids = pool_masks.loc[pool_masks[require].astype(bool), "person_id"].astype(int).tolist()
    else:
        pids = pool_masks["person_id"].astype(int).tolist()

    progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))
    rows = []
    for i, pid in enumerate(pids, 1):
        rows.append(compute_watch_green_for_pid(cfg, pid))
        if progress_every and i % progress_every == 0:
            print(f"  [watch_green] {i}/{len(pids)}…", flush=True)

    return pd.DataFrame(rows)
