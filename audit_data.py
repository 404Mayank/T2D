#!/usr/bin/env python3
"""AI-READI full-v3 data audit — discovers everything that needs cleaning.

Read-only: scans the converted parquet tree and reports schema conformance,
sentinel prevalence, coverage distributions, timestamp gaps, leakage-field
presence, out-of-range physiology, interval validity, split balance, etc.
Outputs a structured report to stdout and logs/audit_report.txt.

Run after the relevant subset (garmin+dexcom+clinical+metadata) is downloaded:
    python audit_data.py
"""
import json, sys, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)

ROOT = Path("data/full/AI_READI")
OUT  = Path("logs/audit_report.txt")
OUT.parent.mkdir(exist_ok=True)

_lines = []
def w(s=""):
    _lines.append(str(s)); print(s)

def banner(t):
    w(); w("=" * 90); w(f"  {t}"); w("=" * 90)

def pct(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else f"{n}/0"

# ----------------------------------------------------------------------------
def scan_parquet(path):
    """Return (metadata, pf) or (None, None) if unreadable."""
    try:
        pf = pq.ParquetFile(path)
        return pf.metadata, pf
    except Exception as e:
        w(f"  [UNREADABLE] {path}: {type(e).__name__}: {e}")
        return None, None

# ============================================================================
banner("0. FILE INVENTORY & PARQUET INTEGRITY")
# ============================================================================
files = sorted(ROOT.rglob("*.parquet"))
w(f"parquet files found: {len(files)}")
total_rows = 0
for f in files:
    meta, _ = scan_parquet(f)
    if meta is None: continue
    rel = f.relative_to(ROOT)
    nrg = meta.num_row_groups
    nrows = meta.num_rows
    ncols = meta.num_columns
    total_rows += nrows
    sz = f.stat().st_size / 1048576
    w(f"  {rel}: {nrows:>9,} rows | {ncols:>3} cols | {nrg:>5} row-groups | {sz:>7.1f} MiB")
w(f"\nTOTAL rows across all parquet: {total_rows:,}")

# ============================================================================
banner("1. PARTICIPANTS / LABEL / SPLIT")
# ============================================================================
pt_path = ROOT / "metadata" / "participants.parquet"
if pt_path.exists():
    pt = pd.read_parquet(pt_path)
    w(f"participants: {len(pt)} rows, columns: {list(pt.columns)}")
    w(f"\nlabel distribution:")
    lbl = pt["label"].value_counts(dropna=False).sort_index()
    for k, v in lbl.items():
        w(f"  label={k}: {v} ({100*v/len(pt):.1f}%)")
    w(f"\nrecommended_split distribution:")
    spl = pt["recommended_split"].value_counts(dropna=False)
    for k, v in spl.items():
        w(f"  {k}: {v} ({100*v/len(pt):.1f}%)")
    w(f"\nlabel × split (to check imbalance is consistent across splits):")
    ct = pd.crosstab(pt["label"], pt["recommended_split"])
    w(ct.to_string())
    w(f"\nstudy_group raw values:")
    for k, v in pt["study_group"].value_counts(dropna=False).items():
        w(f"  {k}: {v}")
    w(f"\nclinical_site distribution (for UTC→local feasibility):")
    for k, v in pt["clinical_site"].value_counts(dropna=False).items():
        w(f"  {k}: {v}")
    w(f"\nnulls per column:")
    for c in pt.columns:
        n = pt[c].isna().sum()
        if n: w(f"  {c}: {n} nulls")
    w(f"\nage: min={pt['age'].min()}, max={pt['age'].max()}, mean={pt['age'].mean():.1f}, nulls={pt['age'].isna().sum()}")
    w(f"duplicate person_id: {pt['person_id'].duplicated().sum()}")
    PIDS = set(pt["person_id"].astype(int))
else:
    w("participants.parquet MISSING"); PIDS = set()

# ============================================================================
banner("2. DATASET_INFO.JSON")
# ============================================================================
di = ROOT / "metadata" / "dataset_info.json"
if di.exists():
    w(di.read_text()[:2000])
else:
    w("dataset_info.json missing")

# ============================================================================
# Helper: per-participant row-group scan for a garmin/dexcom modality
# ============================================================================
def per_participant_stats(path, value_col, ts_col="timestamp", sentinels=None,
                          phys_lo=None, phys_hi=None):
    """Stream row-groups; collect per-participant: n_rows, ts min/max, span_days,
    sentinel counts, out-of-range counts, nulls. Returns DataFrame."""
    pf = pq.ParquetFile(path)
    recs = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()
        if df.empty: continue
        # person_id may be the grouping key
        pid_col = "person_id" if "person_id" in df.columns else df.columns[0]
        for pid, g in df.groupby(pid_col):
            n = len(g)
            ts = pd.to_datetime(g[ts_col], errors="coerce", utc=True) if ts_col in g.columns else None
            row = {"person_id": int(pid), "n_rows": n}
            if ts is not None and ts.notna().any():
                row["ts_min"] = ts.min(); row["ts_max"] = ts.max()
                row["span_days"] = (ts.max() - ts.min()).total_seconds() / 86400
                row["ts_null"] = int(ts.isna().sum())
            else:
                row["ts_min"] = pd.NaT; row["ts_max"] = pd.NaT
                row["span_days"] = np.nan; row["ts_null"] = int(n)
            if value_col in g.columns:
                v = pd.to_numeric(g[value_col], errors="coerce")
                row["val_null"] = int(v.isna().sum())
                if sentinels:
                    row["sentinel"] = int(v.isin(sentinels).sum())
                else:
                    row["sentinel"] = 0
                if phys_lo is not None:
                    row["below_phys"] = int((v < phys_lo).sum())
                else:
                    row["below_phys"] = 0
                if phys_hi is not None:
                    row["above_phys"] = int((v > phys_hi).sum())
                else:
                    row["above_phys"] = 0
                row["val_min"] = float(v.min()) if v.notna().any() else np.nan
                row["val_max"] = float(v.max()) if v.notna().any() else np.nan
            recs.append(row)
    return pd.DataFrame(recs)

def summarize(df, name, sentinel_name=None):
    w(f"\n--- {name} ---  ({len(df)} participants with row-groups)")
    if df.empty: w("  EMPTY"); return
    if "span_days" in df:
        w(f"  span_days:  min={df.span_days.min():.2f}  p25={df.span_days.quantile(.25):.2f}  "
          f"median={df.span_days.median():.2f}  p75={df.span_days.quantile(.75):.2f}  max={df.span_days.max():.2f}")
        w(f"  span < 7d:  {pct((df.span_days<7).sum(), len(df))}   | <1d: {pct((df.span_days<1).sum(), len(df))}")
    if "n_rows" in df:
        w(f"  n_rows/participant: min={df.n_rows.min()}  median={int(df.n_rows.median())}  max={df.n_rows.max()}")
    if "val_null" in df:
        w(f"  val_null total: {int(df.val_null.sum())}  | participants with all-null values: {pct((df.val_null==df.n_rows).sum(), len(df))}")
    if "sentinel" in df and sentinel_name:
        w(f"  {sentinel_name}: total={int(df.sentinel.sum())}  | participants with >0: {pct((df.sentinel>0).sum(), len(df))}")
    if "below_phys" in df and "above_phys" in df:
        w(f"  out-of-range: below={int(df.below_phys.sum())}  above={int(df.above_phys.sum())}")
    if "val_min" in df:
        w(f"  val range: min={df.val_min.min():.2f}  max={df.val_max.max():.2f}")
    # participants in PIDS but missing this modality
    if PIDS:
        have = set(df.person_id.astype(int))
        missing = PIDS - have
        w(f"  modality coverage: {pct(len(have & PIDS), len(PIDS))}  | missing: {len(missing)}")

# ============================================================================
banner("3. GARMIN — HEART RATE (sentinel: ==0 ; phys: 20–220)")
# ============================================================================
p = ROOT / "garmin" / "heart_rate.parquet"
if p.exists():
    df = per_participant_stats(p, "heart_rate", sentinels={0}, phys_lo=20, phys_hi=220)
    summarize(df, "heart_rate", "HR==0 (sensor off)")
    HR = df

# ============================================================================
banner("4. GARMIN — STRESS (sentinel: {-1,-2} ; phys: 0–17)")
# ============================================================================
p = ROOT / "garmin" / "stress.parquet"
if p.exists():
    df = per_participant_stats(p, "stress_level", sentinels={-1, -2}, phys_lo=0, phys_hi=17)
    summarize(df, "stress", "stress∈{-1,-2}")
    STR = df

# ============================================================================
banner("5. GARMIN — RESPIRATORY_RATE (sentinel: {-1,-2} ; phys: 4–60)")
# ============================================================================
p = ROOT / "garmin" / "respiratory_rate.parquet"
if p.exists():
    df = per_participant_stats(p, "respiratory_rate", sentinels={-1, -2}, phys_lo=4, phys_hi=60)
    summarize(df, "respiratory_rate", "RR∈{-1,-2}")
    RR = df

# ============================================================================
banner("6. GARMIN — OXYGEN_SATURATION (sentinel: ==0 ; phys: 50–100)")
# ============================================================================
p = ROOT / "garmin" / "oxygen_saturation.parquet"
if p.exists():
    df = per_participant_stats(p, "oxygen_saturation", sentinels={0}, phys_lo=50, phys_hi=100)
    summarize(df, "oxygen_saturation", "SpO2==0")
    SPO = df
    # measurement_method distribution
    pf = pq.ParquetFile(p)
    methods = defaultdict(int)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg).to_pandas()
        for k, v in t["measurement_method"].value_counts(dropna=False).items():
            methods[k] += v
    w(f"  measurement_method: {dict(methods)}")

# ============================================================================
banner("7. GARMIN — PHYSICAL_ACTIVITY_CALORIE (interval ts; phys: 0–5000)")
# ============================================================================
p = ROOT / "garmin" / "physical_activity_calorie.parquet"
if p.exists():
    df = per_participant_stats(p, "calories", phys_lo=0, phys_hi=5000)
    summarize(df, "calories")
    CAL = df

# ============================================================================
# Interval modalities (sleep, physical_activity) — start/end validity
# ============================================================================
def interval_stats(path, name):
    banner(f"8/9. GARMIN — {name.upper()} (interval: start_time/end_time)")
    if not path.exists(): w(f"  {path} missing"); return None
    pf = pq.ParquetFile(path)
    recs = []
    activity_names = defaultdict(int); sleep_states = defaultdict(int)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if df.empty: continue
        s = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        e = pd.to_datetime(df["end_time"], errors="coerce", utc=True)
        dur = (e - s).dt.total_seconds() / 60
        bad_order = int((e < s).sum())
        zero_dur = int((dur == 0).sum())
        neg_dur = int((dur < 0).sum())
        huge_dur = int((dur > 24 * 60).sum())
        for pid, g in df.groupby("person_id"):
            ss = pd.to_datetime(g["start_time"], errors="coerce", utc=True)
            ee = pd.to_datetime(g["end_time"], errors="coerce", utc=True)
            dd = (ee - ss).dt.total_seconds() / 60
            recs.append({
                "person_id": int(pid), "n_intervals": len(g),
                "span_days": (ss.max() - ss.min()).total_seconds() / 86400 if ss.notna().any() else np.nan,
                "bad_order": int((ee < ss).sum()),
                "zero_dur": int((dd == 0).sum()),
                "neg_dur": int((dd < 0).sum()),
                "huge_dur": int((dd > 1440).sum()),
                "ts_null": int(ss.isna().sum() + ee.isna().sum()),
            })
        if "activity_name" in df.columns:
            for k, v in df["activity_name"].value_counts(dropna=False).items():
                activity_names[k] += v
        if "sleep_stage_state" in df.columns:
            for k, v in df["sleep_stage_state"].value_counts(dropna=False).items():
                sleep_states[k] += v
    dfo = pd.DataFrame(recs)
    if dfo.empty: w("  EMPTY"); return dfo
    w(f"  {len(dfo)} participants | intervals total={dfo.n_intervals.sum()}")
    w(f"  span_days: median={dfo.span_days.median():.2f}  min={dfo.span_days.min():.2f}  max={dfo.span_days.max():.2f}")
    w(f"  bad_order (end<start): {int(dfo.bad_order.sum())}  | neg_dur: {int(dfo.neg_dur.sum())}  "
      f"| zero_dur: {int(dfo.zero_dur.sum())}  | huge_dur(>24h): {int(dfo.huge_dur.sum())}")
    w(f"  ts_null: {int(dfo.ts_null.sum())}")
    if sleep_states:
        w(f"  sleep_stage_state values: {dict(sleep_states)}")
    if activity_names:
        w(f"  activity_name top 25:")
        for k, v in sorted(activity_names.items(), key=lambda x: -x[1])[:25]:
            w(f"    {k}: {v}")
    if PIDS:
        have = set(dfo.person_id.astype(int)); w(f"  coverage: {pct(len(have & PIDS), len(PIDS))}")
    return dfo

SLP = interval_stats(ROOT / "garmin" / "sleep.parquet", "sleep")
ACT = interval_stats(ROOT / "garmin" / "physical_activity.parquet", "physical_activity")

# ============================================================================
banner("10. DEXCOM CGM (event_type, glucose range, coverage)")
# ============================================================================
p = ROOT / "dexcom" / "cgm.parquet"
if p.exists():
    pf = pq.ParquetFile(p)
    recs = []; evt = defaultdict(int); units = defaultdict(int)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if df.empty: continue
        for k, v in df["event_type"].value_counts(dropna=False).items(): evt[k] += v
        for k, v in df["unit"].value_counts(dropna=False).items(): units[k] += v
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        bg = pd.to_numeric(df["blood_glucose"], errors="coerce")
        for pid, g in df.groupby("person_id"):
            gts = pd.to_datetime(g["timestamp"], errors="coerce", utc=True)
            gbg = pd.to_numeric(g["blood_glucose"], errors="coerce")
            recs.append({
                "person_id": int(pid), "n_rows": len(g),
                "span_days": (gts.max() - gts.min()).total_seconds() / 86400 if gts.notna().any() else np.nan,
                "bg_null": int(gbg.isna().sum()),
                "bg_min": float(gbg.min()) if gbg.notna().any() else np.nan,
                "bg_max": float(gbg.max()) if gbg.notna().any() else np.nan,
                "below40": int((gbg < 40).sum()), "above400": int((gbg > 400).sum()),
            })
    dfg = pd.DataFrame(recs)
    w(f"  event_type distribution: {dict(evt)}")
    w(f"  unit distribution: {dict(units)}")
    w(f"  {len(dfg)} participants with CGM")
    w(f"  span_days: median={dfg.span_days.median():.2f}  p25={dfg.span_days.quantile(.25):.2f}  "
      f"p75={dfg.span_days.quantile(.75):.2f}  min={dfg.span_days.min():.2f}  max={dfg.span_days.max():.2f}")
    w(f"  <7d CGM: {pct((dfg.span_days<7).sum(), len(dfg))}")
    w(f"  bg range: min={dfg.bg_min.min():.1f}  max={dfg.bg_max.max():.1f}  | below40: {int(dfg.below40.sum())}  above400: {int(dfg.above400.sum())}")
    w(f"  bg_null total: {int(dfg.bg_null.sum())}")
    if PIDS:
        have = set(dfg.person_id.astype(int)); w(f"  CGM coverage: {pct(len(have & PIDS), len(PIDS))}")

# ============================================================================
banner("11. CLINICAL — schema, row counts, leakage-field presence")
# ============================================================================
LEAKAGE_MEASUREMENT = ["import_hba1c","import_a1c","lbscat_a1c","import_glucose",
                       "import_insulin","import_c_peptide","mlcs","plcs","mssrf","via"]
LEAKAGE_OBSERVATION = ["mhterm_dm1","mhterm_dm2","mhterm_predm","mh_a1c","mh_dm_age",
                       "cmtrt_insulin","cmtrt_a1c","cmtrt","rt","via","dmg"]
RETINAL_MEASUREMENT = ["mlcs","plcs","mssrf","via"]
RETINAL_OBSERVATION = ["rt","via"]

for name in ["person","condition_occurrence","measurement","observation",
             "procedure_occurrence","visit_occurrence"]:
    p = ROOT / "clinical" / f"{name}.parquet"
    if not p.exists(): continue
    df = pd.read_parquet(p)
    w(f"\n--- clinical/{name} ---  {len(df):,} rows × {len(df.columns)} cols")
    w(f"  columns: {list(df.columns)}")
    if "person_id" in df.columns:
        w(f"  unique person_id: {df.person_id.nunique()}  | nulls: {df.person_id.isna().sum()}")
    # null fraction per column (top offenders)
    nf = (df.isna().mean().sort_values(ascending=False) * 100)
    hi = nf[nf > 50]
    if len(hi): w(f"  columns >50% null: {hi.round(1).to_dict()}")
    # leakage / retinal substring scan
    cols = list(df.columns)
    if name == "measurement":
        for tag in LEAKAGE_MEASUREMENT:
            hits = [c for c in cols if tag.lower() in c.lower()]
            if hits: w(f"  [LEAKAGE-CANDIDATE] '{tag}' matches: {hits}")
    if name == "observation":
        for tag in LEAKAGE_OBSERVATION:
            hits = [c for c in cols if tag.lower() in c.lower()]
            if hits: w(f"  [LEAKAGE-CANDIDATE] '{tag}' matches: {hits}")
    if name == "condition_occurrence":
        # look for diabetes ICD codes in any concept column
        for c in cols:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                vals = df[c].dropna().astype(str)
                diab = vals[vals.str.contains("E1[0-3]|diabetes|prediab|pre-diab|IGT|IFG", case=False, regex=True)]
                if len(diab):
                    w(f"  [LEAKAGE-CANDIDATE] {c}: {len(diab)} diabetes-ish values, e.g. {list(diab.unique()[:5])}")
    # survey missing-code sentinels in observation
    if name == "observation":
        sent_cols = {}
        for c in cols:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"): continue
            v = pd.to_numeric(df[c], errors="coerce")
            for code in [555, 777, 888, 999, 99]:
                n = int((v == code).sum())
                if n: sent_cols.setdefault(c, {})[code] = n
        if sent_cols:
            w(f"  [SURVEY-SENTINELS] numeric codes 555/777/888/999/99 found in {len(sent_cols)} cols:")
            for c, d in list(sent_cols.items())[:30]:
                w(f"    {c}: {d}")

# ============================================================================
banner("12. CROSS-MODALITY AVAILABILITY MATRIX")
# ============================================================================
have = {}
for mod, var in [("hr","HR"),("stress","STR"),("rr","RR"),("spo","SPO"),
                 ("cal","CAL"),("slp","SLP"),("act","ACT")]:
    d = globals().get(var)
    if d is not None and not d.empty:
        have[mod] = set(d.person_id.astype(int))
# cgm
p = ROOT / "dexcom" / "cgm.parquet"
if p.exists():
    pf = pq.ParquetFile(p)
    cgm_pids = set()
    for rg in range(pf.num_row_groups):
        cgm_pids.update(pf.read_row_group(rg).to_pandas()["person_id"].astype(int).unique())
    have["cgm"] = cgm_pids
allp = sorted(PIDS) if PIDS else []
w(f"  participants (from index): {len(PIDS)}")
for mod in ["hr","stress","rr","spo","cal","slp","act","cgm"]:
    if mod in have:
        w(f"  {mod:>5}: {pct(len(have[mod] & PIDS), len(PIDS))}")
# pairwise intersection table headline: who has ALL of hr+stress+slp+cgm
if PIDS:
    core = ["hr","stress","rr","slp","cgm"]
    avail = {m: have.get(m, set()) for m in core}
    inter = PIDS
    for m in core: inter &= avail[m]
    w(f"\n  intersection(hr∩stress∩rr∩sleep∩cgm): {pct(len(inter), len(PIDS))}")
    # wearable-only pool (no CGM requirement)
    wear = PIDS
    for m in ["hr","stress","slp"]: wear &= have.get(m, set())
    w(f"  wearable core (hr∩stress∩sleep, no CGM): {pct(len(wear), len(PIDS))}")

# ============================================================================
banner("13. TIMESTAMP SANITY — tz, duplicates, monotonicity")
# ============================================================================
p = ROOT / "garmin" / "heart_rate.parquet"
if p.exists():
    pf = pq.ParquetFile(p)
    sample_pids = list(PIDS)[:200] if PIDS else []
    dup_total = 0; nonmono = 0; checked = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg)
        df = tbl.to_pandas()
        if df.empty: continue
        for pid, g in df.groupby("person_id"):
            if sample_pids and int(pid) not in sample_pids: continue
            ts = pd.to_datetime(g["timestamp"], errors="coerce", utc=True).sort_values()
            dup_total += int(ts.duplicated().sum())
            nonmono += int(ts.is_monotonic_increasing is False)
            checked += 1
        if checked >= 200: break
    w(f"  HR sampled {checked} participants: duplicate-timestamps total={dup_total}, non-monotonic groups={nonmono}")
    # tz check: are all timestamps tz-aware UTC?
    w(f"  (timestamps stored as tz-aware UTC per DATA_STRUCTURE — verify dtype in schema above)")

# ============================================================================
OUT.write_text("\n".join(_lines))
w(f"\n\nReport written to {OUT}")
