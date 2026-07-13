"""I/O helpers: participant index, row-group iteration, parquet write."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import MODALITY_RELPATH


def raw_path(cfg: dict, modality: str) -> Path:
    return cfg["_paths"]["raw_root"] / MODALITY_RELPATH[modality]


def clean_path(cfg: dict, modality: str) -> Path:
    return cfg["_paths"]["clean_dir"] / f"{modality}.parquet"


def load_participants(cfg: dict) -> pd.DataFrame:
    path = cfg["_paths"]["raw_root"] / "metadata" / "participants.parquet"
    df = pd.read_parquet(path)
    df["person_id"] = df["person_id"].astype(np.int64)
    # Optional filters for smoke tests
    rt = cfg.get("runtime") or {}
    ids = rt.get("participant_ids")
    max_n = rt.get("max_participants")
    prefer_hr = rt.get("prefer_hr_participants", True)
    if ids is not None:
        df = df[df["person_id"].isin(ids)].copy()
    if max_n is not None:
        max_n = int(max_n)
        # Prefer pids that actually have HR data (early person_ids often lack wearables)
        if prefer_hr:
            hr_path = cfg["_paths"]["raw_root"] / MODALITY_RELPATH["heart_rate"]
            if hr_path.exists():
                try:
                    hr_pids = set(row_group_person_ids(hr_path))
                    with_hr = df[df["person_id"].isin(hr_pids)].sort_values("person_id")
                    without = df[~df["person_id"].isin(hr_pids)].sort_values("person_id")
                    df = pd.concat([with_hr, without], ignore_index=True)
                except Exception:
                    df = df.sort_values("person_id")
            else:
                df = df.sort_values("person_id")
        else:
            df = df.sort_values("person_id")
        df = df.head(max_n).copy()
    return df.reset_index(drop=True)


def load_person_yob(cfg: dict) -> pd.DataFrame:
    path = cfg["_paths"]["raw_root"] / "clinical" / "person.parquet"
    cols = ["person_id", "year_of_birth"]
    df = pd.read_parquet(path, columns=cols)
    df["person_id"] = df["person_id"].astype(np.int64)
    return df


def iter_row_groups(
    path: Path,
    columns: list[str] | None = None,
) -> Iterator[tuple[int, pa.Table]]:
    """Yield (rg_index, table) for each row group."""
    pf = pq.ParquetFile(path)
    for i in range(pf.metadata.num_row_groups):
        yield i, pf.read_row_group(i, columns=columns)


def row_group_person_ids(path: Path) -> list[int]:
    """Assume one person_id per row group (canonical layout). Returns pid per RG."""
    pf = pq.ParquetFile(path)
    pids = []
    for i in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(i, columns=["person_id"]).column(0)
        # first value
        pids.append(int(col[0].as_py()))
    return pids


class PidParquetWriter:
    """Stream write tables as one row-group each (typically one pid)."""

    def __init__(self, path: Path, schema: pa.Schema | None = None, zstd_level: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.zstd_level = zstd_level
        self._writer: pq.ParquetWriter | None = None
        self.n_groups = 0
        self.n_rows = 0

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self._writer is None:
            schema = self.schema or table.schema
            self._writer = pq.ParquetWriter(
                where=str(self.path),
                schema=schema,
                compression="zstd",
                compression_level=self.zstd_level,
            )
            self.schema = schema
        # cast if needed
        if table.schema != self._writer.schema:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self.n_groups += 1
        self.n_rows += table.num_rows

    def write_df(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        self.write_table(pa.Table.from_pandas(df, preserve_index=False))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_parquet(df: pd.DataFrame, path: Path, zstd_level: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=zstd_level)


def source_value_prefix(val: str | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val)
    return s.split(",", 1)[0].strip()
