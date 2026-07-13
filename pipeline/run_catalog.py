#!/usr/bin/env python3
"""Build OMOP source_value classification map.

Usage:
  .venv/bin/python -m pipeline.run_catalog
  .venv/bin/python -m pipeline.run_catalog --config pipeline/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python -m pipeline.run_catalog` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.catalog.build_source_map import run_build_source_map
from pipeline.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build source_value_map.csv")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    path = run_build_source_map(cfg)
    print(f"Wrote {path}")
    # print class counts
    import pandas as pd

    m = pd.read_csv(path)
    print(m.groupby(["table", "class"]).size().to_string())
    n_border = int((m["class"] == "borderline").sum())
    if n_border:
        print(f"\n⚠ {n_border} borderline prefixes — review {path} and re-run catalog after edits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
