"""Coverage stats and pool membership masks (post-clean)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import MODALITY_RELPATH
from pipeline.io import clean_path, load_participants, write_parquet


def _parse_ts(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=False)


def _day_keys(t: pd.Series) -> np.ndarray:
    """Integer YYYYMMDD from wall-clock components (DST-safe; no floor)."""
    return (
        t.dt.year.to_numpy(dtype=np.int32) * 10000
        + t.dt.month.to_numpy(dtype=np.int32) * 100
        + t.dt.day.to_numpy(dtype=np.int32)
    )


def _minute_keys(t: pd.Series) -> np.ndarray:
    """Unique minute id = day*1440 + hour*60 + minute (DST-safe)."""
    day = _day_keys(t).astype(np.int64)
    return day * 1440 + t.dt.hour.to_numpy(dtype=np.int64) * 60 + t.dt.minute.to_numpy(
        dtype=np.int64
    )


def _valid_days_from_ts(ts: pd.Series) -> int:
    if ts is None or len(ts) == 0:
        return 0
    t = _parse_ts(ts)
    return int(len(np.unique(_day_keys(t))))


def _span_hours(a: pd.Series, b: pd.Series) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    a0, a1 = pd.to_datetime(a).min(), pd.to_datetime(a).max()
    b0, b1 = pd.to_datetime(b).min(), pd.to_datetime(b).max()
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return 0.0
    return float((end - start).total_seconds() / 3600.0)


def _hr_day_stats(ts: pd.Series, min_minutes_per_day: int) -> tuple[int, float]:
    """Return (n_valid_days, mean_minute_frac_on_valid_days)."""
    if ts is None or len(ts) == 0:
        return 0, 0.0
    t = _parse_ts(ts)
    day = _day_keys(t)
    minute = _minute_keys(t)
    # unique minutes per day via sort+reduce (faster than groupby object keys)
    order = np.lexsort((minute, day))
    day_s = day[order]
    min_s = minute[order]
    # unique (day, minute) pairs
    if len(min_s) == 0:
        return 0, 0.0
    pair_change = np.empty(len(min_s), dtype=bool)
    pair_change[0] = True
    pair_change[1:] = (day_s[1:] != day_s[:-1]) | (min_s[1:] != min_s[:-1])
    days_u = day_s[pair_change]
    # count minutes per day
    day_change = np.empty(len(days_u), dtype=bool)
    day_change[0] = True
    day_change[1:] = days_u[1:] != days_u[:-1]
    day_starts = np.flatnonzero(day_change)
    counts = np.diff(np.append(day_starts, len(days_u)))
    valid = counts[counts >= int(min_minutes_per_day)]
    n_valid = int(len(valid))
    if n_valid == 0:
        return 0, 0.0
    mean_frac = float(valid.mean() / 1440.0)
    return n_valid, mean_frac


def _load_clean_ts(cfg: dict, modality: str, pid: int) -> pd.Series | None:
    path = clean_path(cfg, modality)
    if not path.exists():
        return None
    try:
        table = pq.read_table(path, filters=[("person_id", "=", pid)])
    except Exception:
        return None
    if table.num_rows == 0:
        return None
    df = table.to_pandas()
    for col in ("timestamp_local", "timestamp", "start_time_local", "start_time"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True) if "local" not in col else pd.to_datetime(df[col])
    return None


def compute_pool_masks(cfg: dict, participant_index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one-row-per-pid pool mask table using cleaned series on disk."""
    parts = load_participants(cfg) if participant_index is None else participant_index.copy()
    cov = cfg["coverage"]
    pids = parts["person_id"].tolist()

    # Pre-load cleaned modalities into pid->stats by scanning once each file
    stats = {pid: {} for pid in pids}
    pid_set = set(pids)

    modalities_ts = {
        "heart_rate": ("timestamp_local", "timestamp"),
        "stress": ("timestamp_local", "timestamp"),
        "respiratory_rate": ("timestamp_local", "timestamp"),
        "oxygen_saturation": ("timestamp_local", "timestamp"),
        "cgm": ("timestamp_local", "timestamp"),
        "sleep": ("start_time_local", "start_time"),
    }

    for mod, cols in modalities_ts.items():
        path = clean_path(cfg, mod)
        if not path.exists():
            for pid in pids:
                stats[pid][f"{mod}_n"] = 0
                stats[pid][f"{mod}_days"] = 0
                stats[pid][f"{mod}_tmin"] = pd.NaT
                stats[pid][f"{mod}_tmax"] = pd.NaT
                stats[pid][f"{mod}_minute_frac"] = 0.0
            continue

        pf = pq.ParquetFile(path)
        # Only load pid + timestamp columns (full-row scans of stress/RR are huge)
        schema_names = set(pf.schema_arrow.names)
        ts_col = cols[0] if cols[0] in schema_names else cols[1]
        read_cols = ["person_id", ts_col]
        n_rg = pf.metadata.num_row_groups
        progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))
        for i in range(n_rg):
            tbl = pf.read_row_group(i, columns=read_cols)
            if tbl.num_rows == 0:
                continue
            pid = int(tbl.column(0)[0].as_py())
            if pid not in pid_set:
                continue
            df = tbl.to_pandas()
            ts = pd.to_datetime(df[ts_col])
            n_days = _valid_days_from_ts(ts)
            stats[pid][f"{mod}_n"] = int(len(df))
            stats[pid][f"{mod}_days"] = n_days
            stats[pid][f"{mod}_tmin"] = ts.min()
            stats[pid][f"{mod}_tmax"] = ts.max()
            if mod == "heart_rate":
                min_m = int(cfg["coverage"].get("min_minutes_per_hr_day", 60))
                vd, frac = _hr_day_stats(ts, min_m)
                stats[pid]["heart_rate_valid_days"] = vd
                stats[pid]["heart_rate_minute_frac"] = frac
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [pools:{mod}] {i + 1}/{n_rg}…", flush=True)

    rows = []
    for _, pr in parts.iterrows():
        pid = int(pr["person_id"])
        st = stats.get(pid, {})
        hr_days = int(st.get("heart_rate_valid_days", st.get("heart_rate_days", 0)) or 0)
        hr_n = int(st.get("heart_rate_n", 0) or 0)
        hr_frac = float(st.get("heart_rate_minute_frac", 0.0) or 0.0)
        stress_days = int(st.get("stress_days", 0) or 0)
        sleep_days = int(st.get("sleep_days", 0) or 0)
        rr_days = int(st.get("respiratory_rate_days", 0) or 0)
        cgm_days = int(st.get("cgm_days", 0) or 0)

        has_hr_valid = hr_n > 0
        has_stress_valid = int(st.get("stress_n", 0) or 0) > 0
        has_sleep_valid = int(st.get("sleep_n", 0) or 0) > 0
        has_rr_valid = int(st.get("respiratory_rate_n", 0) or 0) > 0
        has_cgm_valid = int(st.get("cgm_n", 0) or 0) > 0

        hr_cov_ok = (
            hr_days >= int(cov["min_hr_valid_days"])
            and hr_frac >= float(cov["min_hr_minute_frac"])
        )
        stress_ok = stress_days >= int(cov.get("min_stress_valid_days", 1)) and has_stress_valid
        sleep_ok = has_sleep_valid
        sleep_nights_ok = sleep_days >= int(cov.get("min_sleep_nights", 7))
        if cov.get("enforce_sleep_nights"):
            sleep_ok = sleep_ok and sleep_nights_ok

        wearable_core = bool(hr_cov_ok and stress_ok and sleep_ok)
        # Always expose a strict companion for sensitivity (even if enforce is false)
        wearable_core_strict = bool(
            hr_cov_ok and stress_ok and has_sleep_valid and sleep_nights_ok
        )

        # overlap HR ∩ CGM
        overlap_h = 0.0
        if has_hr_valid and has_cgm_valid:
            a0, a1 = st.get("heart_rate_tmin"), st.get("heart_rate_tmax")
            b0, b1 = st.get("cgm_tmin"), st.get("cgm_tmax")
            if pd.notna(a0) and pd.notna(b0):
                start = max(a0, b0)
                end = min(a1, b1)
                if end > start:
                    overlap_h = float((end - start).total_seconds() / 3600.0)

        cgm_ok = cgm_days >= int(cov["min_cgm_days"]) and has_cgm_valid
        aux_modalities = bool(wearable_core and has_rr_valid and cgm_ok)
        aux_eligible = bool(
            aux_modalities and overlap_h >= float(cov["min_cgm_hr_overlap_hours"])
        )

        row = {
            "person_id": pid,
            "label": int(pr["label"]) if pd.notna(pr["label"]) else np.nan,
            "recommended_split": pr["recommended_split"],
            "clinical_site": pr["clinical_site"],
            "age": pr["age"],
            "hr_n": hr_n,
            "hr_valid_days": hr_days,
            "hr_minute_frac": hr_frac,
            "stress_n": int(st.get("stress_n", 0) or 0),
            "stress_valid_days": stress_days,
            "sleep_n": int(st.get("sleep_n", 0) or 0),
            "sleep_valid_nights": sleep_days,
            "rr_n": int(st.get("respiratory_rate_n", 0) or 0),
            "rr_valid_days": rr_days,
            "cgm_n": int(st.get("cgm_n", 0) or 0),
            "cgm_valid_days": cgm_days,
            "cgm_hr_overlap_hours": overlap_h,
            "has_hr_valid": has_hr_valid,
            "has_stress_valid": has_stress_valid,
            "has_sleep_valid": has_sleep_valid,
            "has_rr_valid": has_rr_valid,
            "has_cgm_valid": has_cgm_valid,
            "hr_coverage_ok": hr_cov_ok,
            "sleep_nights_ok": sleep_nights_ok,
            "wearable_core": wearable_core,
            "wearable_core_strict": wearable_core_strict,
            "aux_modalities": aux_modalities,
            "aux_eligible": aux_eligible,
        }
        for h in cov.get("overlap_sensitivity_hours") or []:
            row[f"aux_overlap_ge_{int(h)}h"] = bool(
                aux_modalities and overlap_h >= float(h)
            )
        for thr in cov.get("hr_minute_frac_sensitivity") or []:
            row[f"hr_cov_frac_ge_{thr}"] = bool(
                hr_days >= int(cov["min_hr_valid_days"]) and hr_frac >= float(thr)
            )
        rows.append(row)

    return pd.DataFrame(rows)


def coverage_survival_table(masks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Summarize n at each gate for the report."""
    n = len(masks)
    rows = [
        {"stage": "all_labeled", "n": n, "frac": 1.0},
        {"stage": "has_hr_valid", "n": int(masks["has_hr_valid"].sum()), "frac": masks["has_hr_valid"].mean()},
        {"stage": "hr_coverage_ok", "n": int(masks["hr_coverage_ok"].sum()), "frac": masks["hr_coverage_ok"].mean()},
        {"stage": "wearable_core", "n": int(masks["wearable_core"].sum()), "frac": masks["wearable_core"].mean()},
        {"stage": "aux_modalities", "n": int(masks["aux_modalities"].sum()), "frac": masks["aux_modalities"].mean()},
        {"stage": "aux_eligible", "n": int(masks["aux_eligible"].sum()), "frac": masks["aux_eligible"].mean()},
    ]
    for h in cfg["coverage"].get("overlap_sensitivity_hours") or []:
        col = f"aux_overlap_ge_{int(h)}h"
        if col in masks.columns:
            rows.append(
                {
                    "stage": col,
                    "n": int(masks[col].sum()),
                    "frac": float(masks[col].mean()),
                }
            )
    return pd.DataFrame(rows)
