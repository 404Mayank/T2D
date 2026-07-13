"""Point-sampled modality cleaning (HR, stress, RR, SpO2, CGM, calories)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clean.dedup import dedup_instant


def _mask_outside(s: pd.Series, lo: float, hi: float) -> pd.Series:
    out = s.astype("float64")
    bad = (out < lo) | (out > hi)
    out = out.mask(bad)
    return out


def clean_heart_rate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["heart_rate"], keep=keep)
    stats["dedup"] = dstats

    s = cfg["sentinels"]
    b = cfg["bounds"]["heart_rate"]
    hr = df["heart_rate"].astype("float64")
    off = int((hr == s["heart_rate_off"]).sum())
    hr = hr.mask(hr == s["heart_rate_off"])
    before_valid = hr.notna().sum()
    hr = _mask_outside(hr, b[0], b[1])
    stats["sentinel_off"] = off
    stats["out_of_range"] = int(before_valid - hr.notna().sum())
    df = df.copy()
    df["heart_rate"] = hr
    # drop pure-null value rows? keep rows for coverage calc — actually drop all-null value
    # Keep rows even if null after mask so day coverage can see wear attempts? Prefer drop nulls
    # for clean storage; coverage uses cleaned valid counts.
    n_before = len(df)
    df = df[df["heart_rate"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def clean_stress(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["stress_level"], keep=keep)
    stats["dedup"] = dstats

    inv = set(cfg["sentinels"]["stress_invalid"])
    b = cfg["bounds"]["stress_level"]
    x = df["stress_level"].astype("float64")
    stats["sentinel_invalid"] = int(x.isin(inv).sum())
    x = x.mask(x.isin(inv))
    before = x.notna().sum()
    x = _mask_outside(x, b[0], b[1])
    stats["out_of_range"] = int(before - x.notna().sum())
    df = df.copy()
    df["stress_level"] = x
    n_before = len(df)
    df = df[df["stress_level"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def clean_respiratory_rate(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["respiratory_rate"], keep=keep)
    stats["dedup"] = dstats

    inv = set(cfg["sentinels"]["respiratory_rate_invalid"])
    b = cfg["bounds"]["respiratory_rate"]
    min_year = int(cfg["time"]["min_year"])

    df = df.copy()
    # corrupt timestamp drop
    ts = pd.to_datetime(df["timestamp"], utc=True)
    year_ok = ts.dt.year >= min_year
    stats["bad_year_dropped"] = int((~year_ok).sum())
    df = df.loc[year_ok].copy()
    ts = ts.loc[year_ok]

    x = df["respiratory_rate"].astype("float64")
    stats["sentinel_invalid"] = int(x.isin(inv).sum())
    x = x.mask(x.isin(inv))
    before = x.notna().sum()
    x = _mask_outside(x, b[0], b[1])
    stats["out_of_range"] = int(before - x.notna().sum())
    df["respiratory_rate"] = x
    df["timestamp"] = ts
    n_before = len(df)
    df = df[df["respiratory_rate"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def clean_spo2(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["oxygen_saturation"], keep=keep)
    stats["dedup"] = dstats

    off = cfg["sentinels"]["spo2_off"]
    b = cfg["bounds"]["oxygen_saturation"]
    x = df["oxygen_saturation"].astype("float64")
    stats["sentinel_off"] = int((x == off).sum())
    x = x.mask(x == off)
    before = x.notna().sum()
    x = _mask_outside(x, b[0], b[1])
    stats["out_of_range"] = int(before - x.notna().sum())
    df = df.copy()
    df["oxygen_saturation"] = x
    n_before = len(df)
    df = df[df["oxygen_saturation"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def clean_cgm(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["blood_glucose"], keep=keep)
    stats["dedup"] = dstats

    b = cfg["bounds"]["blood_glucose"]
    df = df.copy()
    # Prefer EGV only if column present
    if "event_type" in df.columns:
        egv = df["event_type"].astype(str).str.upper() == "EGV"
        stats["non_egv_dropped"] = int((~egv).sum())
        df = df.loc[egv].copy()

    x = df["blood_glucose"].astype("float64")
    before = x.notna().sum()
    x = _mask_outside(x, b[0], b[1])
    stats["out_of_range"] = int(before - x.notna().sum())
    df["blood_glucose"] = x
    n_before = len(df)
    df = df[df["blood_glucose"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


def clean_calories(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    stats: dict = {}
    keep = cfg["dedup"]["timestamp_keep"]
    df, dstats = dedup_instant(df, "timestamp", ["calories"], keep=keep)
    stats["dedup"] = dstats
    df = df.copy()
    df["calories"] = df["calories"].astype("float64")
    # optional diff cleaning is feature-time; just store cleaned counter
    n_before = len(df)
    df = df[df["calories"].notna()].copy()
    stats["dropped_null"] = n_before - len(df)
    stats["n_out"] = len(df)
    return df, stats


CLEANERS = {
    "heart_rate": clean_heart_rate,
    "stress": clean_stress,
    "respiratory_rate": clean_respiratory_rate,
    "oxygen_saturation": clean_spo2,
    "cgm": clean_cgm,
    "physical_activity_calorie": clean_calories,
}
