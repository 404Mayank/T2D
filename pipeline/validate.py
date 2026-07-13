"""Post-clean assertions and report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.clean.clinical import leakage_column_scan


def assert_no_leakage(df: pd.DataFrame, cfg: dict) -> list[str]:
    bad = leakage_column_scan(list(df.columns), cfg)
    if bad and cfg.get("runtime", {}).get("strict_leakage_assert", True):
        raise AssertionError(f"Leakage columns present: {bad}")
    return bad


def write_report(cfg: dict, report: dict[str, Any], name: str = "clean_report.json") -> Path:
    path = cfg["_paths"]["reports_dir"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # make JSON-safe
    def conv(o):
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        return o

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=conv)
    return path
