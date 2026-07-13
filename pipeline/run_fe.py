#!/usr/bin/env python3
"""Build feature matrices from cleaned data.

Feature files contain person_id + features only. Join labels/splits from
meta/pool_masks.parquet at train time.

Usage:
  .venv/bin/python -m pipeline.run_fe
  .venv/bin/python -m pipeline.run_fe --blocks watch
  .venv/bin/python -m pipeline.run_fe --max-participants 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pipeline.clean.clinical import leakage_column_scan
from pipeline.config import ensure_out_dirs, load_config
from pipeline.fe.watch_green import build_watch_green
from pipeline.io import write_parquet
from pipeline.validate import write_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Feature engineering from data/processed/clean")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--blocks",
        default="watch",
        help="Comma list: watch,onboarding,comorbidity,mood,diet",
    )
    ap.add_argument("--max-participants", type=int, default=None)
    args = ap.parse_args(argv)

    overrides = {}
    if args.max_participants is not None:
        overrides["runtime"] = {"max_participants": args.max_participants}
    cfg = load_config(args.config, overrides=overrides or None)
    ensure_out_dirs(cfg)

    masks_path = cfg["_paths"]["meta_dir"] / "pool_masks.parquet"
    if not masks_path.exists():
        print("Missing pool_masks.parquet — run: python -m pipeline.run_clean")
        return 1
    masks = pd.read_parquet(masks_path)
    if args.max_participants is not None:
        idx_path = cfg["_paths"]["meta_dir"] / "participant_index.parquet"
        if idx_path.exists():
            idx = pd.read_parquet(idx_path)
            allow = set(idx["person_id"].astype(int))
            masks = masks[masks["person_id"].isin(allow)].copy()

    report = {"started": time.time(), "blocks": {}}
    want = [b.strip() for b in args.blocks.split(",") if b.strip()]

    if "watch" in want and cfg["features"]["watch_green"].get("enabled", True):
        print("=== watch_green ===")
        t0 = time.time()
        feat = build_watch_green(cfg, masks)
        bad = leakage_column_scan(list(feat.columns), cfg)
        if bad:
            raise AssertionError(f"Leakage/meta columns in watch_green: {bad}")
        out = cfg["_paths"]["features_dir"] / "watch_green.parquet"
        write_parquet(feat, out)
        report["blocks"]["watch_green"] = {
            "shape": list(feat.shape),
            "seconds": round(time.time() - t0, 2),
            "path": str(out),
            "n_rows": len(feat),
        }
        print(f"  {feat.shape} → {out} ({report['blocks']['watch_green']['seconds']}s)")
        print(f"  columns: {list(feat.columns)}")

    for b in want:
        if b == "watch":
            continue
        path = cfg["_paths"]["features_dir"] / f"{b}.parquet"
        if not path.exists():
            print(f"  skip {b}: missing {path} (run run_clean without --skip-clinical)")
            continue
        df = pd.read_parquet(path)
        bad = leakage_column_scan(list(df.columns), cfg)
        report["blocks"][b] = {"shape": list(df.shape), "leakage": bad, "path": str(path)}
        if bad:
            raise AssertionError(f"Leakage/meta columns in {b}: {bad}")
        print(f"  {b}: ok {df.shape}")

    report["elapsed_s"] = round(time.time() - report["started"], 2)
    write_report(cfg, report, name="fe_report.json")
    print(f"Done in {report['elapsed_s']}s")
    print("Join labels at train time:")
    print("  feats = pd.read_parquet('data/processed/features/watch_green.parquet')")
    print("  meta  = pd.read_parquet('data/processed/meta/pool_masks.parquet')")
    print("  df = feats.merge(meta[['person_id','label','recommended_split',...]], on='person_id')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
