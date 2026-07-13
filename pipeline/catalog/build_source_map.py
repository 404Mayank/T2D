"""Build / refresh meta/source_value_map.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.clean.clinical import build_source_map_from_raw
from pipeline.config import ensure_out_dirs


def run_build_source_map(cfg: dict) -> Path:
    ensure_out_dirs(cfg)
    out_path = cfg["_paths"]["meta_dir"] / "source_value_map.csv"
    new_map = build_source_map_from_raw(cfg)

    if out_path.exists():
        old = pd.read_csv(out_path)
        # Preserve ONLY rows with non-empty notes (explicit human lock).
        # Auto-reclassify everything else so config rule updates take effect.
        if {"table", "prefix", "class"}.issubset(old.columns):
            key = ["table", "prefix"]
            old_idx = old.set_index(key)
            merged_rows = []
            for _, r in new_map.iterrows():
                k = (r["table"], r["prefix"])
                row = r.to_dict()
                if k in old_idx.index:
                    prev = old_idx.loc[k]
                    if isinstance(prev, pd.DataFrame):
                        prev = prev.iloc[0]
                    if str(prev.get("notes", "")).strip():
                        row["class"] = prev["class"]
                        row["block"] = prev.get("block", row.get("block", ""))
                        row["notes"] = prev["notes"]
                merged_rows.append(row)
            new_map = pd.DataFrame(merged_rows)

    new_map.to_csv(out_path, index=False)

    # summary side-car
    summary = (
        new_map.groupby(["table", "class"]).size().reset_index(name="n_prefixes")
    )
    summary_path = cfg["_paths"]["reports_dir"] / "source_value_class_summary.csv"
    summary.to_csv(summary_path, index=False)
    return out_path
