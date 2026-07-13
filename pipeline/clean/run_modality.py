"""Clean one modality end-to-end (stream row groups → write cleaned parquet)."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.clean.intervals import INTERVAL_CLEANERS
from pipeline.clean.series import CLEANERS
from pipeline.clean.timestamps import add_local_time, apply_window
from pipeline.constants import site_tz_map
from pipeline.io import PidParquetWriter, clean_path, load_participants, raw_path


def _zone_for_pid(pid: int, site_by_pid: dict[int, str], tz_map: dict[str, str]) -> str:
    site = site_by_pid.get(pid)
    if site is None or site not in tz_map:
        return tz_map.get("UW", "America/Los_Angeles")
    return tz_map[site]


def clean_modality(
    cfg: dict,
    modality: str,
    shared_windows: dict[int, tuple] | None = None,
) -> dict:
    """Clean a single modality; return stats dict.

    If shared_windows is provided (pid -> (start, end) local), apply that
    HR-anchored window. Otherwise fall back to no window truncation (full
    cleaned span) — prefer always passing shared windows from run_clean.
    """
    in_path = raw_path(cfg, modality)
    out_path = clean_path(cfg, modality)
    if not in_path.exists():
        return {"modality": modality, "error": f"missing {in_path}"}

    parts = load_participants(cfg)
    site_by_pid = dict(zip(parts["person_id"].astype(int), parts["clinical_site"].astype(str)))
    allow = set(parts["person_id"].astype(int))
    tz_map = site_tz_map(cfg)
    zstd = int(cfg.get("runtime", {}).get("zstd_level", 1))
    progress_every = int(cfg.get("runtime", {}).get("progress_every", 200))

    cleaner = CLEANERS.get(modality) or INTERVAL_CLEANERS.get(modality)
    if cleaner is None:
        return {"modality": modality, "error": "no cleaner"}

    is_interval = modality in INTERVAL_CLEANERS
    ts_cols = ["start_time", "end_time"] if is_interval else ["timestamp"]
    primary_ts = "start_time_local" if is_interval else "timestamp_local"

    agg = {
        "modality": modality,
        "n_pids_in": 0,
        "n_pids_out": 0,
        "n_rows_in": 0,
        "n_rows_out": 0,
        "truncated_pids": 0,
        "no_shared_window_pids": 0,
        "cleaner": {},
    }
    window_rows = []

    if out_path.exists():
        out_path.unlink()

    pf = pq.ParquetFile(in_path)
    writer: PidParquetWriter | None = None

    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i)
        df = tbl.to_pandas()
        if df.empty:
            continue
        pid = int(df["person_id"].iloc[0])
        if pid not in allow:
            continue
        agg["n_pids_in"] += 1
        agg["n_rows_in"] += len(df)

        df, cstats = cleaner(df, cfg)
        for k, v in cstats.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    key = f"{k}.{kk}"
                    agg["cleaner"][key] = agg["cleaner"].get(key, 0) + (
                        vv if isinstance(vv, (int, float)) else 0
                    )
            elif isinstance(v, (int, float)):
                agg["cleaner"][k] = agg["cleaner"].get(k, 0) + v

        if df.empty:
            window_rows.append(
                {
                    "person_id": pid,
                    "modality": modality,
                    "n_rows": 0,
                    "truncated": False,
                    "window_start_local": pd.NaT,
                    "window_end_local": pd.NaT,
                }
            )
            continue

        zone = _zone_for_pid(pid, site_by_pid, tz_map)
        df = add_local_time(df, ts_cols, zone)
        if primary_ts not in df.columns:
            df[primary_ts] = df[ts_cols[0]]

        start = end = None
        truncated = False
        if shared_windows is not None:
            bounds = shared_windows.get(pid)
            if bounds is None:
                agg["no_shared_window_pids"] += 1
                # No HR-anchored window → drop series for this pid (cannot align)
                window_rows.append(
                    {
                        "person_id": pid,
                        "modality": modality,
                        "n_rows": 0,
                        "truncated": False,
                        "window_start_local": pd.NaT,
                        "window_end_local": pd.NaT,
                        "dropped_no_hr_window": True,
                    }
                )
                continue
            start, end = bounds
            n_before = len(df)
            df = apply_window(df, primary_ts, start, end)
            truncated = len(df) < n_before
            if truncated:
                agg["truncated_pids"] += 1

        window_rows.append(
            {
                "person_id": pid,
                "modality": modality,
                "n_rows": len(df),
                "truncated": truncated,
                "window_start_local": start,
                "window_end_local": end,
            }
        )

        if df.empty:
            continue

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = PidParquetWriter(out_path, schema=table.schema, zstd_level=zstd)
        writer.write_table(table)
        agg["n_pids_out"] += 1
        agg["n_rows_out"] += len(df)

        if progress_every and agg["n_pids_in"] % progress_every == 0:
            print(f"  [{modality}] {agg['n_pids_in']} pids…", flush=True)

    if writer is not None:
        writer.close()

    wdf = pd.DataFrame(window_rows)
    wpath = cfg["_paths"]["meta_dir"] / f"windows_{modality}.parquet"
    if len(wdf):
        wdf.to_parquet(wpath, index=False)

    agg["out_path"] = str(out_path)
    return agg
