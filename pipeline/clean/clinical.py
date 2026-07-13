"""OMOP clinical cleaning: survey sentinels, pivot long→wide, anthropometrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.constants import block_for_prefix, classify_prefix
from pipeline.io import source_value_prefix, write_parquet


def _mask_survey_values(df: pd.DataFrame, cfg: dict, value_cols: list[str]) -> pd.DataFrame:
    codes = set(float(x) for x in cfg["sentinels"]["survey_codes"])
    str_codes = {str(int(c)) if float(c).is_integer() else str(c) for c in codes}
    df = df.copy()
    for c in value_cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64").mask(df[c].isin(codes))
        else:
            s = df[c].astype(str)
            df[c] = df[c].mask(s.isin(str_codes) | s.isin({f"{int(c)}" for c in codes}))
    return df


def load_source_map(cfg: dict) -> pd.DataFrame | None:
    path = cfg["_paths"]["meta_dir"] / "source_value_map.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_source_map_from_raw(cfg: dict) -> pd.DataFrame:
    """Enumerate observation + measurement source_values and auto-classify."""
    raw = cfg["_paths"]["raw_root"]
    rows = []
    for table, col in (
        ("observation", "observation_source_value"),
        ("measurement", "measurement_source_value"),
    ):
        path = raw / "clinical" / f"{table}.parquet"
        df = pd.read_parquet(path, columns=["person_id", col])
        df["prefix"] = df[col].map(source_value_prefix)
        g = df.groupby("prefix", dropna=False)
        for prefix, sub in g:
            prefix = prefix or ""
            cls = classify_prefix(prefix, cfg)
            block = block_for_prefix(prefix, cls, cfg)
            # sample full source_value
            sample = str(sub[col].iloc[0]) if len(sub) else ""
            rows.append(
                {
                    "table": table,
                    "prefix": prefix,
                    "source_value_sample": sample,
                    "n_rows": int(len(sub)),
                    "n_pids": int(sub["person_id"].nunique()),
                    "class": cls,
                    "block": block if block else "",
                    "notes": "",
                }
            )
    out = pd.DataFrame(rows).sort_values(["table", "class", "prefix"]).reset_index(drop=True)
    return out


def pivot_clinical(
    cfg: dict,
    source_map: pd.DataFrame,
    person_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Pivot keep-* rows to one row per person_id; return wide + stats."""
    stats: dict = {"tables": {}}
    raw = cfg["_paths"]["raw_root"]
    keep_classes = {
        "keep_onboarding",
        "keep_comorbidity",
        "keep_mood",
        "keep_diet",
        "keep_survey",
    }
    keep_map = source_map[source_map["class"].isin(keep_classes)].copy()

    allow = set(int(x) for x in person_ids) if person_ids is not None else None

    frames = []
    for table, scol, vnum, vstr in (
        (
            "observation",
            "observation_source_value",
            "value_as_number",
            "value_as_string",
        ),
        (
            "measurement",
            "measurement_source_value",
            "value_as_number",
            None,
        ),
    ):
        path = raw / "clinical" / f"{table}.parquet"
        cols = ["person_id", scol, vnum]
        if vstr:
            cols.append(vstr)
        df = pd.read_parquet(path, columns=cols)
        if allow is not None:
            df = df[df["person_id"].isin(allow)]
        df["prefix"] = df[scol].map(source_value_prefix)
        allowed = set(keep_map.loc[keep_map["table"] == table, "prefix"])
        # hard safety: never pivot hard_exclude even if map wrong
        hard = set(
            source_map.loc[
                (source_map["table"] == table) & (source_map["class"] == "hard_exclude"),
                "prefix",
            ]
        )
        df = df[df["prefix"].isin(allowed) & ~df["prefix"].isin(hard)].copy()
        value_cols = [vnum] + ([vstr] if vstr else [])
        df = _mask_survey_values(df, cfg, value_cols)

        # choose value: prefer numeric
        def pick_value(row):
            vn = row.get(vnum)
            if pd.notna(vn):
                return float(vn)
            if vstr and pd.notna(row.get(vstr)):
                s = str(row[vstr]).strip()
                try:
                    return float(s)
                except ValueError:
                    return s
            return np.nan

        df["value"] = df.apply(pick_value, axis=1)
        # one value per person_id x prefix (last non-null)
        df = df.sort_values(["person_id", "prefix"])
        piv = df.groupby(["person_id", "prefix"], as_index=False)["value"].last()
        wide = piv.pivot(index="person_id", columns="prefix", values="value")
        wide.columns = [str(c) for c in wide.columns]
        frames.append(wide)
        stats["tables"][table] = {
            "n_keep_prefixes": len(allowed),
            "n_feature_cols": wide.shape[1],
            "n_pids": int(wide.shape[0]),
        }

    if not frames:
        return pd.DataFrame(columns=["person_id"]), stats

    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, how="outer", rsuffix="_dup")
        # drop accidental dup cols
        wide = wide[[c for c in wide.columns if not str(c).endswith("_dup")]]

    wide = wide.reset_index()
    wide["person_id"] = wide["person_id"].astype(np.int64)

    # Anthropometric fixes
    b = cfg["bounds"]
    for col, vmin in (
        ("waist_vsorres", b.get("waist_cm_min", 1)),
        ("hip_vsorres", b.get("hip_cm_min", 1)),
    ):
        if col in wide.columns:
            s = pd.to_numeric(wide[col], errors="coerce")
            wide[col] = s.mask(s < float(vmin))

    if "waist_vsorres" in wide.columns and "hip_vsorres" in wide.columns:
        w = pd.to_numeric(wide["waist_vsorres"], errors="coerce")
        h = pd.to_numeric(wide["hip_vsorres"], errors="coerce")
        wide["whr_vsorres"] = (w / h).where(h > 0)

    if "bmi_vsorres" in wide.columns:
        bmi = pd.to_numeric(wide["bmi_vsorres"], errors="coerce")
        flag_above = b.get("bmi_flag_above", 60)
        stats["bmi_flagged"] = int((bmi > flag_above).fillna(False).sum())
        null_above = b.get("null_bmi_above")
        if null_above is not None:
            wide["bmi_vsorres"] = bmi.mask(bmi > float(null_above))
        else:
            wide["bmi_vsorres"] = bmi

    stats["n_cols_total"] = int(wide.shape[1] - 1)
    stats["n_pids"] = int(wide.shape[0])
    return wide, stats


def split_clinical_blocks(
    clinical_wide: pd.DataFrame,
    source_map: pd.DataFrame,
    participants: pd.DataFrame,
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    """Return block_name -> dataframe with person_id + feature columns ONLY.

    label / recommended_split / clinical_site live in meta/pool_masks and
    participant_index — never inside feature matrices (C1).
    """
    prefix_to_block = {}
    for _, r in source_map.iterrows():
        if str(r["class"]).startswith("keep"):
            blk = r["block"] or block_for_prefix(str(r["prefix"]), str(r["class"]), cfg)
            if blk:
                prefix_to_block[str(r["prefix"])] = blk

    # age is an onboarding feature (not a train-meta column)
    age_source = (cfg.get("clinical") or {}).get("age_source", "participants")
    age_df = participants[["person_id"]].copy()
    if age_source == "year_of_birth" and "age_yob_derived" in participants.columns:
        age_df["age"] = participants["age_yob_derived"].values
    else:
        age_df["age"] = participants["age"].values

    write_blocks = (cfg.get("features") or {}).get("blocks", {}).get("write") or []
    out: dict[str, pd.DataFrame] = {}

    for block in write_blocks:
        cols = [
            c
            for c in clinical_wide.columns
            if c != "person_id" and prefix_to_block.get(c) == block
        ]
        df = participants[["person_id"]].merge(
            clinical_wide[["person_id"] + cols], on="person_id", how="left"
        )
        if block == "onboarding":
            df = df.merge(age_df, on="person_id", how="left")
            # put age first after person_id
            ordered = ["person_id", "age"] + [c for c in df.columns if c not in ("person_id", "age")]
            df = df[ordered]
        out[block] = df

    keep_cols = [c for c in clinical_wide.columns if c == "person_id" or c in prefix_to_block]
    keep_all = participants[["person_id"]].merge(clinical_wide[keep_cols], on="person_id", how="left")
    # include age on clinical_keep_all as optional onboarding signal
    keep_all = keep_all.merge(age_df, on="person_id", how="left")
    out["clinical_keep_all"] = keep_all
    return out


# Columns that must never appear as *features* (join keys / eval / confounders)
FORBIDDEN_FEATURE_COLS = frozenset(
    {
        "label",
        "recommended_split",
        "clinical_site",
        "study_group",
        "study_visit_date",
        "wearable_core",
        "aux_eligible",
        "aux_modalities",
        "hr_valid_days",
        "cgm_hr_overlap_hours",
        "has_hr_valid",
        "has_stress_valid",
        "has_sleep_valid",
        "has_rr_valid",
        "has_cgm_valid",
        "hr_coverage_ok",
    }
)


def leakage_column_scan(columns: list[str], cfg: dict) -> list[str]:
    """Return column names that must not appear as model features."""
    hard = list(cfg["classify"].get("hard_exclude_prefixes") or [])
    hard += list(cfg["classify"].get("hard_exclude_exact") or [])
    extra_sub = list(
        (cfg.get("classify") or {}).get("leakage_substrings")
        or [
            "hba1c",
            "import_insulin",
            "import_glucose",
            "c_peptide",
            "mhterm_dm",
            "mhoccur_glc",
            "mhoccur_pdr",
            "mhoccur_rvo",
            "cmtrt_insln",
            "cmtrt_a1c",
            "cmtrt_glcs",
            "cmtrt_lfst",
        ]
    )
    meta_suffixes = tuple(
        (cfg.get("classify") or {}).get("metadata_suffixes")
        or ("startts", "cmpts", "cmpdat", "endts")
    )
    bad = []
    for c in columns:
        if c == "person_id":
            continue
        if c in FORBIDDEN_FEATURE_COLS:
            bad.append(c)
            continue
        if any(c == h or c.startswith(h) for h in hard):
            bad.append(c)
            continue
        low = c.lower()
        if any(k in low for k in extra_sub):
            bad.append(c)
            continue
        # survey instrument metadata timestamps
        if any(low.endswith(sfx) for sfx in meta_suffixes):
            bad.append(c)
            continue
    return sorted(set(bad))
