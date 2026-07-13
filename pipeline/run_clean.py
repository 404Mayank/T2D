#!/usr/bin/env python3
"""Run row-level cleaning, clinical pivot, and pool masks.

Usage:
  .venv/bin/python -m pipeline.run_clean
  .venv/bin/python -m pipeline.run_clean --only heart_rate,stress,sleep
  .venv/bin/python -m pipeline.run_clean --max-participants 20
  .venv/bin/python -m pipeline.run_clean --skip-series
  .venv/bin/python -m pipeline.run_clean --skip-clinical
  .venv/bin/python -m pipeline.run_clean --skip-pools
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pipeline.catalog.build_source_map import run_build_source_map
from pipeline.clean.clinical import (
    leakage_column_scan,
    load_source_map,
    pivot_clinical,
    split_clinical_blocks,
)
from pipeline.clean.run_modality import clean_modality
from pipeline.clean.windows import compute_shared_windows, windows_lookup
from pipeline.config import ensure_out_dirs, load_config, modality_list
from pipeline.io import load_participants, load_person_yob, write_parquet
from pipeline.pools import compute_pool_masks, coverage_survival_table
from pipeline.validate import write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clean AI-READI → data/processed/")
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated modalities")
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--skip-series", action="store_true")
    ap.add_argument("--skip-clinical", action="store_true")
    ap.add_argument("--skip-pools", action="store_true")
    ap.add_argument("--skip-catalog", action="store_true", help="Do not refresh source_value_map")
    ap.add_argument(
        "--force-all-window",
        action="store_true",
        help="Override config: slice every pid to window.days (H3 strict)",
    )
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.max_participants is not None:
        overrides["runtime"] = {"max_participants": args.max_participants}
    if args.only:
        mods = [m.strip() for m in args.only.split(",") if m.strip()]
        overrides.setdefault("runtime", {})["only_modalities"] = mods
    if args.force_all_window:
        overrides.setdefault("time", {}).setdefault("window", {})["force_all_to_window"] = True

    cfg = load_config(args.config, overrides=overrides or None)
    ensure_out_dirs(cfg)
    report: dict = {"started": time.time(), "stages": {}}

    parts = load_participants(cfg)
    yob = load_person_yob(cfg)
    parts = parts.merge(yob, on="person_id", how="left")
    try:
        visit_year = pd.to_datetime(parts["study_visit_date"], errors="coerce").dt.year
        derived = visit_year - parts["year_of_birth"]
        parts["age_yob_derived"] = derived
        parts["age_discrepancy"] = (parts["age"] - derived).abs()
    except Exception:
        parts["age_yob_derived"] = np.nan
        parts["age_discrepancy"] = np.nan

    write_parquet(parts, cfg["_paths"]["meta_dir"] / "participant_index.parquet")
    report["n_participants"] = len(parts)
    print(f"Participants: {len(parts)}")

    if not args.skip_catalog and not args.skip_clinical:
        print("=== catalog ===")
        p = run_build_source_map(cfg)
        report["stages"]["catalog"] = {"path": str(p)}
        smap_preview = pd.read_csv(p)
        n_border = int((smap_preview["class"] == "borderline").sum())
        report["stages"]["catalog"]["n_borderline"] = n_border
        print(f"  source map → {p}  (borderline prefixes: {n_border})")

    shared_lookup = None
    if not args.skip_series:
        print("=== shared HR-anchored windows ===")
        t0 = time.time()
        shared = compute_shared_windows(cfg)
        shared_lookup = windows_lookup(shared)
        n_trunc = int(shared["truncated"].fillna(False).sum()) if "truncated" in shared.columns else 0
        n_empty = int(shared["empty"].fillna(False).sum()) if "empty" in shared.columns else 0
        report["stages"]["shared_windows"] = {
            "n": len(shared),
            "with_window": len(shared_lookup),
            "truncated": n_trunc,
            "empty_hr": n_empty,
            "force_all_to_window": bool(cfg["time"]["window"].get("force_all_to_window", False)),
            "seconds": round(time.time() - t0, 2),
        }
        print(
            f"  windows: {len(shared_lookup)}/{len(shared)}  "
            f"truncated={n_trunc} empty_hr={n_empty}  "
            f"({report['stages']['shared_windows']['seconds']}s)"
        )

        # Regression guard: no window starts before year 2000
        bad_years = 0
        for pid, (s, e) in shared_lookup.items():
            if pd.notna(s) and getattr(s, "year", 9999) < 2000:
                bad_years += 1
        if bad_years:
            raise RuntimeError(
                f"Shared-window unit bug: {bad_years} windows start before year 2000"
            )

        mods = modality_list(cfg)
        print(f"=== series clean: {mods} ===")
        for mod in mods:
            print(f"-- {mod}")
            t0 = time.time()
            stats = clean_modality(cfg, mod, shared_windows=shared_lookup)
            stats["seconds"] = round(time.time() - t0, 2)
            report["stages"][mod] = stats
            print(
                f"   pids {stats.get('n_pids_out')}/{stats.get('n_pids_in')}  "
                f"rows {stats.get('n_rows_out')}/{stats.get('n_rows_in')}  "
                f"truncated {stats.get('truncated_pids')}  ({stats['seconds']}s)"
            )

    if not args.skip_clinical:
        print("=== clinical pivot ===")
        smap = load_source_map(cfg)
        if smap is None:
            run_build_source_map(cfg)
            smap = load_source_map(cfg)
        wide, cstats = pivot_clinical(
            cfg, smap, person_ids=parts["person_id"].astype(int).tolist()
        )
        write_parquet(wide, cfg["_paths"]["clean_dir"] / "clinical_wide.parquet")
        blocks = split_clinical_blocks(wide, smap, parts, cfg)
        for name, bdf in blocks.items():
            bad = leakage_column_scan(list(bdf.columns), cfg)
            if bad:
                raise AssertionError(f"Leakage/meta columns in {name}: {bad}")
            write_parquet(bdf, cfg["_paths"]["features_dir"] / f"{name}.parquet")
            print(f"  block {name}: {bdf.shape}")
        report["stages"]["clinical"] = cstats
        print(f"  clinical_wide: {wide.shape}")

    if not args.skip_pools:
        print("=== pools ===")
        masks = compute_pool_masks(cfg, parts)
        write_parquet(masks, cfg["_paths"]["meta_dir"] / "pool_masks.parquet")
        surv = coverage_survival_table(masks, cfg)
        surv_path = cfg["_paths"]["reports_dir"] / "coverage_survival.csv"
        surv.to_csv(surv_path, index=False)
        report["stages"]["pools"] = {
            "n": len(masks),
            "wearable_core": int(masks["wearable_core"].sum()),
            "aux_eligible": int(masks["aux_eligible"].sum()),
            "survival": surv.to_dict(orient="records"),
        }
        print(surv.to_string(index=False))
        print(f"  wrote {surv_path}")

    report["finished"] = time.time()
    report["elapsed_s"] = round(report["finished"] - report["started"], 2)
    rpath = write_report(cfg, report)
    print(f"Report → {rpath} ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
