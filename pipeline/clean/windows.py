"""Shared per-participant analysis windows (HR-anchored)."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from pipeline.clean.series import clean_heart_rate
from pipeline.clean.timestamps import add_local_time, pick_window_bounds
from pipeline.constants import site_tz_map
from pipeline.io import load_participants, raw_path, write_parquet


def _zone_for_pid(pid: int, site_by_pid: dict[int, str], tz_map: dict[str, str]) -> str:
    site = site_by_pid.get(int(pid))
    if site is None or site not in tz_map:
        # Prefer explicit fail-soft: default Pacific but flag in info
        return tz_map.get("UW", "America/Los_Angeles")
    return tz_map[site]


def compute_shared_windows(cfg: dict) -> pd.DataFrame:
    """Compute one [start,end) local window per pid, anchored on cleaned HR.

    Writes meta/shared_windows.parquet and returns the frame.
    """
    parts = load_participants(cfg)
    site_by_pid = dict(zip(parts["person_id"].astype(int), parts["clinical_site"].astype(str)))
    allow = set(parts["person_id"].astype(int))
    tz_map = site_tz_map(cfg)
    win_cfg = cfg["time"]["window"]
    force = bool(win_cfg.get("force_all_to_window", False))
    policy = win_cfg.get("policy", "best_coverage")
    days = float(win_cfg.get("days", 14))
    long_wear = float(win_cfg.get("long_wear_days", 60))

    in_path = raw_path(cfg, "heart_rate")
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rows = []
    pf = pq.ParquetFile(in_path)
    progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))
    n_seen = 0

    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        if df.empty:
            continue
        pid = int(df["person_id"].iloc[0])
        if pid not in allow:
            continue
        n_seen += 1
        zone = _zone_for_pid(pid, site_by_pid, tz_map)
        site = site_by_pid.get(pid, "")

        df, _ = clean_heart_rate(df, cfg)
        if df.empty:
            rows.append(
                {
                    "person_id": pid,
                    "clinical_site": site,
                    "zone": zone,
                    "window_start_local": pd.NaT,
                    "window_end_local": pd.NaT,
                    "truncated": False,
                    "span_days_pre": 0.0,
                    "n_hr_valid": 0,
                    "policy_applied": policy,
                    "anchor": "heart_rate",
                    "empty": True,
                }
            )
            continue

        df = add_local_time(df, ["timestamp"], zone)
        start, end, info = pick_window_bounds(
            df["timestamp_local"],
            policy=policy,
            days=days,
            long_wear_days=long_wear,
            force_all_to_window=force,
        )
        rows.append(
            {
                "person_id": pid,
                "clinical_site": site,
                "zone": zone,
                "window_start_local": start,
                "window_end_local": end,
                "truncated": bool(info.get("truncated")),
                "span_days_pre": float(info.get("span_days", 0.0)),
                "n_hr_valid": int(len(df)),
                "best_count": info.get("best_count"),
                "policy_applied": info.get("policy_applied"),
                "anchor": "heart_rate",
                "empty": False,
            }
        )
        if progress_every and n_seen % progress_every == 0:
            print(f"  [shared_windows] {n_seen}…", flush=True)

    # pids with no HR row group
    have = {int(r["person_id"]) for r in rows}
    for pid in sorted(allow - have):
        site = site_by_pid.get(pid, "")
        zone = _zone_for_pid(pid, site_by_pid, tz_map)
        rows.append(
            {
                "person_id": pid,
                "clinical_site": site,
                "zone": zone,
                "window_start_local": pd.NaT,
                "window_end_local": pd.NaT,
                "truncated": False,
                "span_days_pre": 0.0,
                "n_hr_valid": 0,
                "policy_applied": policy,
                "anchor": "heart_rate",
                "empty": True,
            }
        )

    out = pd.DataFrame(rows)
    path = cfg["_paths"]["meta_dir"] / "shared_windows.parquet"
    write_parquet(out, path)
    return out


def windows_lookup(shared: pd.DataFrame) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """pid -> (start, end) for non-empty windows."""
    out: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, r in shared.iterrows():
        if r.get("empty") or pd.isna(r["window_start_local"]) or pd.isna(r["window_end_local"]):
            continue
        out[int(r["person_id"])] = (r["window_start_local"], r["window_end_local"])
    return out
