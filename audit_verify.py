#!/usr/bin/env python3
"""Verify the critiquer's load-bearing claims against the actual data.
Each check prints VERIFIED / REFUTED / PARTIAL with the real numbers."""
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, pyarrow.parquet as pq
warnings.filterwarnings("ignore")
ROOT = Path("data/full/AI_READI")
PT = pd.read_parquet(ROOT/"metadata"/"participants.parquet")
PIDS = set(PT.person_id.astype(int))
def w(s=""): print(s)

# ---- O1: person.parquet demographics blank? ----
w("="*80); w("O1: person.parquet demographics"); w("="*80)
person = pd.read_parquet(ROOT/"clinical/person.parquet")
for c in ["gender_concept_id","race_concept_id","ethnicity_concept_id",
          "gender_source_value","race_source_value","ethnicity_source_value",
          "year_of_birth","month_of_birth","day_of_birth","birth_datetime"]:
    if c in person.columns:
        s = person[c]
        nun = s.nunique(dropna=True)
        if nun <= 5:
            w(f"  {c}: nunique={nun}  values={dict(s.value_counts(dropna=False).head(5))}")
        else:
            w(f"  {c}: nunique={nun}  min={s.min()}  max={s.max()}  sample={list(s.dropna().unique()[:3])}")
# search observation + measurement for any sex/gender/race/ethnicity field
obs = pd.read_parquet(ROOT/"clinical/observation.parquet")
meas = pd.read_parquet(ROOT/"clinical/measurement.parquet")
for label, df in [("observation", obs), ("measurement", meas)]:
    hits = [s for s in df[df.columns[-3] if df.columns[-3]=='observation_source_value' else 'observation_source_value'].dropna().unique() if label=='observation'] if label=='observation' else []
    col = "observation_source_value" if label=="observation" else "measurement_source_value"
    matches = [str(s) for s in df[col].dropna().unique() if any(k in str(s).lower() for k in ["sex","gender","race","ethnic"])]
    w(f"  {label} sex/gender/race/ethnic matches: {matches[:10]}")

# ---- O2: condition_occurrence ICD codes? ----
w("\n"+"="*80); w("O2: condition_occurrence source_values"); w("="*80)
co = pd.read_parquet(ROOT/"clinical/condition_occurrence.parquet")
w(f"  rows={len(co)}, unique condition_source_value={co['condition_source_value'].nunique()}")
sv = co["condition_source_value"].value_counts(dropna=False)
for k,v in sv.items():
    w(f"    {k}: {v}")
# check for ICD E10-E13 patterns
diab = co[co["condition_source_value"].astype(str).str.contains("E1[0-3]|diabet|predm|IFG|IGT", case=False, regex=True, na=False)]
w(f"  diabetes/ICD-pattern matches: {len(diab)} rows, source_values={diab['condition_source_value'].unique()[:10]}")

# ---- O3+O4: full-cohort exact + timestamp duplicates (HR, stress, RR, CGM) ----
w("\n"+"="*80); w("O3+O4: duplicates (full cohort)"); w("="*80)
for mod, fname, valcol, tscol in [("heart_rate","heart_rate","heart_rate","timestamp"),
                                   ("stress","stress","stress_level","timestamp"),
                                   ("respiratory_rate","respiratory_rate","respiratory_rate","timestamp"),
                                   ("cgm","../dexcom/cgm","blood_glucose","timestamp")]:
    p = ROOT/"garmin"/f"{fname}.parquet" if mod!="cgm" else ROOT/"dexcom"/"cgm.parquet"
    pf = pq.ParquetFile(p)
    exact_dup_rows = 0; ts_dup_pids = 0; ts_dup_extra = 0; checked = 0
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        for pid, g in df.groupby("person_id"):
            checked += 1
            # exact row dups (all columns)
            exact_dup_rows += int(g.duplicated().sum())
            # timestamp dups
            ts = g[tscol]
            d = int(ts.duplicated().sum())
            if d:
                ts_dup_pids += 1; ts_dup_extra += d
    w(f"  {mod}: checked={checked} pids | exact_dup_rows={exact_dup_rows} | ts_dup_pids={ts_dup_pids} | ts_dup_extra_rows={ts_dup_extra}")

# ---- O5: clinical date column dtypes (schema) ----
w("\n"+"="*80); w("O5: schema dtypes for date/timestamp columns"); w("="*80)
for rel, cols in [("metadata/participants.parquet",["study_visit_date"]),
                  ("clinical/observation.parquet",["observation_date","observation_datetime"]),
                  ("clinical/measurement.parquet",["measurement_date","measurement_datetime","measurement_time"]),
                  ("clinical/visit_occurrence.parquet",["visit_start_date","visit_start_datetime"]),
                  ("clinical/condition_occurrence.parquet",["condition_start_date","condition_start_datetime"])]:
    pf = pq.ParquetFile(ROOT/rel)
    sch = pf.schema_arrow
    for c in cols:
        try:
            f = sch.field(c)
            w(f"  {rel} :: {c} -> arrow {f.type}")
        except: pass
# show sample values
for rel, c in [("metadata/participants.parquet","study_visit_date"),
               ("clinical/observation.parquet","observation_date"),
               ("clinical/measurement.parquet","measurement_date"),
               ("clinical/observation.parquet","observation_datetime")]:
    df = pd.read_parquet(ROOT/rel, columns=[c])
    w(f"  sample {rel}.{c}: {list(df[c].dropna().unique()[:3])}")

# garmin/dexcom timestamp dtype
for rel, c in [("garmin/heart_rate.parquet","timestamp"),("dexcom/cgm.parquet","timestamp")]:
    pf = pq.ParquetFile(ROOT/rel); w(f"  {rel} :: {c} -> arrow {pf.schema_arrow.field(c).type}")

# ---- O6: post-sentinel zero-valid-reading counts ----
w("\n"+"="*80); w("O6: post-sentinel zero-valid participants"); w("="*80)
def zero_valid(path, valcol, sentinels):
    pf = pq.ParquetFile(path); zeros = 0; total = 0
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        for pid, g in df.groupby("person_id"):
            total += 1
            v = pd.to_numeric(g[valcol], errors="coerce")
            valid = v[~v.isin(sentinels) & v.notna()]
            if len(valid) == 0: zeros += 1
    return zeros, total
for mod, fname, valcol, sents in [("HR","heart_rate","heart_rate",{0}),
                                   ("stress","stress","stress_level",{-1,-2}),
                                   ("RR","respiratory_rate","respiratory_rate",{-1,-2})]:
    z, t = zero_valid(ROOT/"garmin"/f"{fname}.parquet", valcol, sents)
    w(f"  {mod}: {z}/{t} participants with ZERO valid readings after sentinel masking")

# ---- O7: CGM ↔ HR temporal overlap ----
w("\n"+"="*80); w("O7: CGM <-> HR temporal overlap"); w("="*80)
def windows(path, tscol="timestamp"):
    pf = pq.ParquetFile(path); out = {}
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        for pid, g in df.groupby("person_id"):
            ts = pd.to_datetime(g[tscol], errors="coerce", utc=True).dropna()
            if len(ts): out[int(pid)] = (ts.min(), ts.max())
    return out
hrw = windows(ROOT/"garmin/heart_rate.parquet")
cgmw = windows(ROOT/"dexcom/cgm.parquet")
both = set(hrw) & set(cgmw)
overlap_h = []
for pid in both:
    h0,h1 = hrw[pid]; c0,c1 = cgmw[pid]
    lo = max(h0,c0); hi = min(h1,c1)
    oh = (hi-lo).total_seconds()/3600 if hi>lo else 0
    overlap_h.append(oh)
overlap_h = np.array(overlap_h)
w(f"  CGM∩HR participants: {len(both)}")
w(f"  ≤0h overlap: {(overlap_h<=0).sum()}  | <24h: {(overlap_h<24).sum()}  | <72h: {(overlap_h<72).sum()}")
w(f"  overlap hrs: p5={np.percentile(overlap_h,5):.0f}  p50={np.percentile(overlap_h,50):.0f}  p95={np.percentile(overlap_h,95):.0f}")
# within aux pool
aux = PIDS & set(hrw) & set(cgmw)  # not full aux (needs stress/rr/sleep) but CGM∩HR subset
# build the real aux pool
have = {}
for mod, fname in [("hr","heart_rate"),("stress","stress"),("rr","respiratory_rate"),("slp","sleep")]:
    have[mod] = set()
    pf = pq.ParquetFile(ROOT/f"garmin/{fname}.parquet")
    for rg in range(pf.num_row_groups):
        have[mod].update(pf.read_row_group(rg).to_pandas()["person_id"].astype(int).unique())
have["cgm"] = set(cgmw)
auxpool = PIDS & have["hr"] & have["stress"] & have["rr"] & have["slp"] & have["cgm"]
aux_overlap = []
for pid in auxpool:
    h0,h1 = hrw[pid]; c0,c1 = cgmw[pid]
    lo = max(h0,c0); hi = min(h1,c0)  # intentional: min(h1,c1)? fix below
aux_overlap = []
for pid in auxpool:
    h0,h1 = hrw[pid]; c0,c1 = cgmw[pid]
    lo = max(h0,c0); hi = min(h1,c1)
    aux_overlap.append((hi-lo).total_seconds()/3600 if hi>lo else 0)
aux_overlap = np.array(aux_overlap)
w(f"  Within aux pool (n={len(auxpool)}): ≤0h overlap={(aux_overlap<=0).sum()}  | <24h={(aux_overlap<24).sum()}  | <72h={(aux_overlap<72).sum()}")

# ---- M4: anthropometric waist/hip=0 ----
w("\n"+"="*80); w("M4+M6: anthropometric + BMI outliers"); w("="*80)
for c in ["weight_vsorres","height_vsorres","bmi_vsorres","waist_vsorres","hip_vsorres","whr_vsorres"]:
    m = meas[meas["measurement_source_value"].astype(str).str.startswith(c.split("_")[0])]
    # need exact prefix match; filter by source_value containing the code
    sub = meas[meas["measurement_source_value"].astype(str).str.contains(c.replace("_vsorres",""), case=False, na=False)]
    if len(sub):
        v = pd.to_numeric(sub["value_as_number"], errors="coerce").dropna()
        if len(v):
            w(f"  {c}: n={len(v)} min={v.min():.2f} max={v.max():.2f} | ==0: {(v==0).sum()} | >60(if bmi): {(v>60).sum() if 'bmi' in c else '-'}")

# ---- M5: age vs year_of_birth discrepancy ----
w("\n"+"="*80); w("M5: age vs year_of_birth"); w("="*80)
pt = PT.merge(person[["person_id","year_of_birth"]], on="person_id", how="left")
pt["visit_year"] = pd.to_datetime(pt["study_visit_date"], errors="coerce").dt.year
pt["implied_age"] = pt["visit_year"] - pt["year_of_birth"]
pt["diff"] = pt["age"] - pt["implied_age"]
w(f"  exact match: {(pt['diff'].abs()<=1).sum()}  | off by >1y: {(pt['diff'].abs()>1).sum()}  (max diff={pt['diff'].abs().max():.0f})")
bad = pt[pt["diff"].abs()>1][["person_id","age","year_of_birth","visit_year","implied_age","diff"]].head(10)
w(bad.to_string())

# ---- M7: year-long wear split distribution ----
w("\n"+"="*80); w("M7: year-long-wear (>60d) split distribution"); w("="*80)
pf = pq.ParquetFile(ROOT/"garmin/heart_rate.parquet")
long_pids = []
for rg in range(pf.num_row_groups):
    df = pf.read_row_group(rg).to_pandas()
    for pid, g in df.groupby("person_id"):
        ts = pd.to_datetime(g["timestamp"], errors="coerce", utc=True).dropna()
        if len(ts) and (ts.max()-ts.min()).total_seconds()/86400 > 60:
            long_pids.append(int(pid))
lp = PT[PT.person_id.isin(long_pids)]
w(f"  year-long HR participants: {len(long_pids)}  by split: {dict(lp['recommended_split'].value_counts())}  by label: {dict(lp['label'].value_counts().sort_index())}")
