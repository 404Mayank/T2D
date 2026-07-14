"""One-off: extract smoke_ever / smoke_current from raw OMOP observation.

AI-READI codes use susmk* (not smok*). Writes data/processed/features/smoking.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SENTINELS = {555, 777, 888, 999, 99}


def _code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(",").str[0].str.strip()


def _clean_bin(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(~x.isin(list(SENTINELS)), np.nan)
    # keep only 0/1
    x = x.where(x.isin([0.0, 1.0]), np.nan)
    return x


def build_smoking(
    observation_path: Path,
    *,
    id_col: str = "person_id",
) -> pd.DataFrame:
    obs = pd.read_parquet(
        observation_path,
        columns=[id_col, "observation_source_value", "value_as_number", "value_as_string"],
    )
    code = _code(obs["observation_source_value"])
    ever = obs.loc[code == "susmkncf", [id_col, "value_as_number"]].copy()
    ever["smoke_ever"] = _clean_bin(ever["value_as_number"])
    ever = ever.drop(columns=["value_as_number"]).drop_duplicates(id_col, keep="first")

    cur = obs.loc[code == "susmkcdur", [id_col, "value_as_number"]].copy()
    cur["smoke_current_raw"] = _clean_bin(cur["value_as_number"])
    cur = cur.drop(columns=["value_as_number"]).drop_duplicates(id_col, keep="first")

    # person universe from ever (full questionnaire)
    out = ever.merge(cur, on=id_col, how="left")
    # never-smoker → current 0; ever missing → current NA; else raw current
    out["smoke_current"] = np.where(
        out["smoke_ever"] == 0,
        0.0,
        out["smoke_current_raw"],
    )
    out = out.drop(columns=["smoke_current_raw"])
    # dtypes
    out["smoke_ever"] = out["smoke_ever"].astype("float64")
    out["smoke_current"] = out["smoke_current"].astype("float64")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--observation",
        type=Path,
        default=Path("data/full/AI_READI/clinical/observation.parquet"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/features/smoking.parquet"),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.out.exists() and not args.force:
        raise SystemExit(f"refuse overwrite {args.out} (pass --force)")
    if not args.observation.exists():
        raise FileNotFoundError(args.observation)

    df = build_smoking(args.observation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} n={len(df)}")
    print(df[["smoke_ever", "smoke_current"]].isna().mean().to_dict())
    print("ever vc", df["smoke_ever"].value_counts(dropna=False).to_dict())
    print("current vc", df["smoke_current"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
