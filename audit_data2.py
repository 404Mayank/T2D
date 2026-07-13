#!/usr/bin/env python3
"""AI-READI audit — pass 2: deep-dive on anomalies found in pass 1.
- stress value distribution (max=101 vs documented 0-17)
- timestamp outliers (RR span 12215d, HR/stress 397d)
- OMOP source_value mapping (survey features + leakage by real names)
- SpO2 >100, HR duplicates
- fixed cross-modality intersection (pass-1 bug: set &= mutated PIDS)
"""
import warnings, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd, pyarrow.parquet as pq
warnings.filterwarnings("ignore")
ROOT = Path("data/full/AI_READI")
PIDS = set(pd.read_parquet(ROOT/"metadata"/"participants.parquet")["person_id"].astype(int))
PT = pd.read_parquet(ROOT/"metadata"/"participants.parquet")
SITE_TZ = {"UAB":"US/Central", "UW":"US/Pacific", "UCSD":"US/Pacific"}

def w(s=""): print(s)

# ---- 1. STRESS value distribution ----
w("="*80); w("  STRESS VALUE DISTRIBUTION (documented 0-17; observed max=101)"); w("="*80)
pf = pq.ParquetFile(ROOT/"garmin/stress.parquet")
vals = []
for rg in range(pf.num_row_groups):
    v = pd.to_numeric(pf.read_row_group(rg).to_pandas()["stress_level"], errors="coerce")
    vals.append(v)
vals = pd.concat(vals)
w(f"total stress rows: {len(vals):,}")
w(f"value_counts (top 30, sorted by value):")
vc = vals.value_counts(dropna=False).sort_index()
for k, v in vc.head(40).items():
    w(f"  {k:>6}: {v:>10,}  ({100*v/len(vals):.2f}%)")
w(f"\nbuckets:")
w(f"  ==-2: {(vals==-2).sum():,}  ({100*(vals==-2).mean():.1f}%)")
w(f"  ==-1: {(vals==-1).sum():,}  ({100*(vals==-1).mean():.1f}%)")
w(f"  0-17 (documented valid): {((vals>=0)&(vals<=17)).sum():,}  ({100*((vals>=0)&(vals<=17)).mean():.1f}%)")
w(f"  18-100: {((vals>17)&(vals<=100)).sum():,}  ({100*((vals>17)&(vals<=100)).mean():.1f}%)")
w(f"  >100: {(vals>100).sum():,}")
w(f"  NaN: {vals.isna().sum():,}")
# per-participant: how many have ANY value >17
w(f"\nper-participant stress max (sample first 1500 row-groups):")
pf = pq.ParquetFile(ROOT/"garmin/stress.parquet")
maxes = []
for rg in range(min(1500, pf.num_row_groups)):
    df = pf.read_row_group(rg).to_pandas()
    for pid, g in df.groupby("person_id"):
        v = pd.to_numeric(g["stress_level"], errors="coerce")
        maxes.append((int(pid), float(v.max()), float(v[v<=17].max()) if (v<=17).any() else np.nan))
mx = pd.DataFrame(maxes, columns=["pid","abs_max","max_le17"])
w(f"  participants with abs_max>17: {(mx.abs_max>17).sum()}/{len(mx)}")
w(f"  abs_max distribution: min={mx.abs_max.min():.0f} p50={mx.abs_max.median():.0f} p90={mx.abs_max.quantile(.9):.0f} max={mx.abs_max.max():.0f}")

# ---- 2. TIMESTAMP OUTLIERS ----
w("\n"+"="*80); w("  TIMESTAMP OUTLIERS (RR span 12215d, HR/stress 397d)"); w("="*80)
for mod, col, tsc in [("heart_rate","heart_rate","timestamp"),
                      ("respiratory_rate","respiratory_rate","timestamp"),
                      ("stress","stress_level","timestamp")]:
    pf = pq.ParquetFile(ROOT/f"garmin/{mod}.parquet")
    bad = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        for pid, g in df.groupby("person_id"):
            ts = pd.to_datetime(g[tsc], errors="coerce", utc=True)
            if ts.notna().any():
                span = (ts.max()-ts.min()).total_seconds()/86400
                if span > 60:  # >60 days is suspicious for a ~14d wear window
                    bad.append((int(pid), span, ts.min(), ts.max(), len(g)))
    w(f"\n  {mod}: {len(bad)} participants with span>60d")
    bad.sort(key=lambda x:-x[1])
    for pid, span, tmin, tmax, n in bad[:8]:
        w(f"    pid={pid} span={span:.1f}d  n={n}  {tmin} → {tmax}")

# ---- 3. OMOP source_value mapping (survey features + leakage by real names) ----
w("\n"+"="*80); w("  OBSERVATION source_value top 80 (find survey features + leakage)"); w("="*80)
obs = pd.read_parquet(ROOT/"clinical/observation.parquet")
w(f"observation rows: {len(obs):,}, unique person: {obs.person_id.nunique()}")
sv = obs["observation_source_value"].value_counts(dropna=False)
w(f"unique source_values: {obs['observation_source_value'].nunique()}")
w(f"\ntop 80 observation_source_value:")
for k, v in sv.head(80).items():
    w(f"  {k:>40}: {v:>7,}  (pids={obs[obs.observation_source_value==k].person_id.nunique()})")

w("\n--- leakage/retinal substring scan in observation_source_value ---")
for tag in ["hba1c","a1c","insulin","c_peptide","diabet","predm","dm1","dm2","mh_","cmtrt","cmtrt_","mhoccur","paidscore","cestl","ces","fh_dm","via","rt_","mlcs","plcs","mssrf","dmg","susmk","sualc","diet","pxfi"]:
    hits = [s for s in obs["observation_source_value"].dropna().unique() if tag.lower() in str(s).lower()]
    if hits:
        w(f"  '{tag}': {hits[:12]}")

w("\n"+"="*80); w("  MEASUREMENT source_value top 60 (find lab leakage + retinal)"); w("="*80)
meas = pd.read_parquet(ROOT/"clinical/measurement.parquet")
w(f"measurement rows: {len(meas):,}, unique person: {meas.person_id.nunique()}")
msv = meas["measurement_source_value"].value_counts(dropna=False)
w(f"unique source_values: {meas['measurement_source_value'].nunique()}")
for k, v in msv.head(60).items():
    w(f"  {k:>40}: {v:>7,}  (pids={meas[meas.measurement_source_value==k].person_id.nunique()})")
w("\n--- leakage/retinal substring scan in measurement_source_value ---")
for tag in ["hba1c","a1c","glucose","insulin","c_peptide","via","mlcs","plcs","mssrf","bmi","weight","waist","height","bp","blood_press","lipid","chol","ldl","hdl","trig","systol","diastol","sp02","spo2","oxygen"]:
    hits = [s for s in meas["measurement_source_value"].dropna().unique() if tag.lower() in str(s).lower()]
    if hits:
        w(f"  '{tag}': {hits[:12]}")

# ---- 4. SpO2 >100 + HR duplicates ----
w("\n"+"="*80); w("  SpO2 >100 and HR duplicate-timestamp detail"); w("="*80)
spo = pd.read_parquet(ROOT/"garmin/oxygen_saturation.parquet")
v = pd.to_numeric(spo["oxygen_saturation"], errors="coerce")
w(f"SpO2 >100: {(v>100).sum()}  | <50: {(v<50).sum()}  | ==0: {(v==0).sum()}  | NaN: {v.isna().sum()}")
w(f"SpO2 value_counts top 15: {v.value_counts().head(15).sort_index().to_dict()}")
# HR duplicates: how many participants, are they exact dup rows or just ts?
pf = pq.ParquetFile(ROOT/"garmin/heart_rate.parquet")
dup_pids = 0; dup_rows = 0; checked = 0
for rg in range(min(400, pf.num_row_groups)):
    df = pf.read_row_group(rg).to_pandas()
    for pid, g in df.groupby("person_id"):
        ts = pd.to_datetime(g["timestamp"], errors="coerce", utc=True)
        d = int(ts.duplicated().sum())
        if d: dup_pids += 1; dup_rows += d
        checked += 1
w(f"\nHR duplicates in {checked} participants: {dup_pids} pids with dup-timestamps, {dup_rows} dup rows total")

# ---- 5. FIXED cross-modality intersection ----
w("\n"+"="*80); w("  CROSS-MODALITY INTERSECTION (fixed: copy PIDS before &=)"); w("="*80)
have = {}
for mod, fname in [("hr","heart_rate"),("stress","stress"),("rr","respiratory_rate"),
                   ("spo","oxygen_saturation"),("cal","physical_activity_calorie"),
                   ("slp","sleep"),("act","physical_activity")]:
    pf = pq.ParquetFile(ROOT/f"garmin/{fname}.parquet")
    s = set()
    for rg in range(pf.num_row_groups):
        s.update(pf.read_row_group(rg).to_pandas()["person_id"].astype(int).unique())
    have[mod] = s
pf = pq.ParquetFile(ROOT/"dexcom/cgm.parquet")
cgm = set()
for rg in range(pf.num_row_groups):
    cgm.update(pf.read_row_group(rg).to_pandas()["person_id"].astype(int).unique())
have["cgm"] = cgm
N = len(PIDS)
for mod in ["hr","stress","rr","spo","cal","slp","act","cgm"]:
    w(f"  {mod:>5}: {len(have[mod] & PIDS)}/{N} ({100*len(have[mod]&PIDS)/N:.1f}%)")
def inter(mods):
    r = set(PIDS)
    for m in mods: r &= have[m]
    return r
w(f"\n  hr∩stress∩sleep (wearable core):              {len(inter(['hr','stress','slp']))}/{N} ({100*len(inter(['hr','stress','slp']))/N:.1f}%)")
w(f"  hr∩stress∩rr∩sleep:                            {len(inter(['hr','stress','rr','slp']))}/{N} ({100*len(inter(['hr','stress','rr','slp']))/N:.1f}%)")
w(f"  hr∩stress∩rr∩sleep∩cgm (aux pool):             {len(inter(['hr','stress','rr','slp','cgm']))}/{N} ({100*len(inter(['hr','stress','rr','slp','cgm']))/N:.1f}%)")
w(f"  hr∩stress∩slp∩act∩cgm (no RR):                 {len(inter(['hr','stress','slp','act','cgm']))}/{N} ({100*len(inter(['hr','stress','slp','act','cgm']))/N:.1f}%)")
w(f"  all 7 garmin + cgm:                            {len(inter(['hr','stress','rr','spo','cal','slp','act','cgm']))}/{N} ({100*len(inter(['hr','stress','rr','spo','cal','slp','act','cgm']))/N:.1f}%)")

# ---- 6. label balance per split (exact) ----
w("\n"+"="*80); w("  LABEL × SPLIT exact counts + insulin scarcity"); w("="*80)
ct = pd.crosstab(PT.label, PT.recommended_split)
w(ct.to_string())
w(f"\ninsulin (label=3): train={int(ct.loc[3,'train'])} val={int(ct.loc[3,'val'])} test={int(ct.loc[3,'test'])}  total={int(ct.loc[3].sum())}")
w(f"  → train insulin n={int(ct.loc[3,'train'])} is the binding constraint for the 4-class model")
for sp in ["train","val","test"]:
    w(f"  {sp} class proportions: " + "  ".join(f"{l}:{int(ct.loc[l,sp])}" for l in [0,1,2,3]))

# ---- 7. timezone mapping feasibility ----
w("\n"+"="*80); w("  CLINICAL_SITE → TIMEZONE mapping"); w("="*80)
w(f"sites: {PT.clinical_site.value_counts().to_dict()}")
w(f"mapped: {SITE_TZ}")
w(f"nulls in clinical_site: {PT.clinical_site.isna().sum()}")
# how many participants per site per label (confound check)
w(f"\nsite × label:")
w(pd.crosstab(PT.clinical_site, PT.label).to_string())
