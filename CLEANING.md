# Cleaning & Feature Pipeline

> **Single doc for how raw AI-READI becomes model-ready tables** (design + runbook + full-cohort results).
> Empirical data facts: `DATA_AUDIT.md`. Feature definitions / leakage: `FEATURES.md`.
> ML build order: `Training.md`. Config: `pipeline/config.yaml`. Code: `pipeline/`.
>
> **Training / analysis only?** Use **`PROCESSED.md`** (layout + join contract) — you do not need
> this whole file unless you rebuild or change cleaning policy.

Anyone (human or agent) should be able to run, modify, or extend cleaning/FE from this
file without re-deriving the design from code.

---

## Quick start

```bash
# 1) OMOP source_value map
.venv/bin/python -m pipeline.run_catalog

# 2) Shared HR window + row clean + clinical pivot + pool masks
.venv/bin/python -m pipeline.run_clean

# Smoke (HR-bearing pids preferred when max_participants is set):
.venv/bin/python -m pipeline.run_clean --max-participants 20 \
  --only heart_rate,stress,sleep,physical_activity,cgm

# Equal-length windows for all pids (sensitivity):
.venv/bin/python -m pipeline.run_clean --force-all-window

# 3) GREEN wearable features (person_id + features ONLY)
.venv/bin/python -m pipeline.run_fe --blocks watch

# 4) Path B daily matrices (B1; no re-clean; site-tz re-derived in FE)
.venv/bin/python -m pipeline.run_fe --blocks cgm_daily,watch_daily
```

**Train-time join** (feature files do **not** contain label/split/site):

```python
import pandas as pd

feats = pd.read_parquet("data/processed/features/watch_green.parquet")
meta  = pd.read_parquet("data/processed/meta/pool_masks.parquet")
df = feats.merge(
    meta[
        [
            "person_id",
            "label",
            "recommended_split",
            "clinical_site",       # reporting / stratification — NOT default X
            "wearable_core",
            "wearable_core_strict",
            "aux_eligible",
        ]
    ],
    on="person_id",
    how="inner",
)
train = df[df["recommended_split"] == "train"]
feature_cols = [c for c in feats.columns if c != "person_id"]
X_train, y_train = train[feature_cols], train["label"]
```

---

## 1. Purpose & scope

### In scope
- Read-only consumption of canonical parquet under `data/full/AI_READI/`
- Row-level cleaning (sentinels, dedup, bounds, intervals, corrupt timestamps)
- UTC → local conversion (site-confounded; mandatory before circadian features)
- **One shared, HR-anchored analysis window per participant**
- OMOP long→wide clinical pivot with source_value classification
- Post-clean pool masks (`wearable_core`, `aux_eligible`, …)
- Path A **View A** feature matrices: GREEN watch summaries + clinical blocks

### Out of scope (v1 clean stages)
- Mutating canonical raw data (`convert_pipeline.py` stays separate)
- Model training, Optuna, SHAP (consumers of `data/processed/features/`)
- ECG / environment modalities

**Note:** Path B View B **5-min multi-modal grid** is **built** as FE (not clean):
`python -m pipeline.run_fe --blocks grid_5min` → `features/grid_5min*.parquet` (2026-07-15).
Clean stages still do not re-window for concurrent minute overlap; FE emits concurrent masks.

### Design principles
1. **Raw is immutable.** All outputs land in `data/processed/`.
2. **Policy is config, not code.** Sweep thresholds via `pipeline/config.yaml` / CLI.
3. **Clean once, engineer many times.** Stages 2–5 never compute RHR/SRI/MVPA.
4. **Feature files are features only.** Labels, splits, site, pool flags join at train time
   from `meta/` — never ride inside `features/*.parquet` (prevents silent leakage).
5. **Streaming-safe.** Garmin/dexcom processed one row-group (≈ one pid) at a time for 16 GB RAM.

---

## 2. High-level data flow

```
data/full/AI_READI/                         # CANONICAL — read only
        │
        ▼
┌───────────────────┐
│ 0  Config         │  pipeline/config.yaml  (+ CLI overrides)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 1  Catalog        │  OMOP source_value → class map
│    run_catalog    │  → meta/source_value_map.csv
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 2  Shared windows │  HR-anchored [start, end) per pid
│                   │  → meta/shared_windows.parquet
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 3  Series clean   │  dedup → sentinel → bounds → local TZ → apply shared window
│    per modality   │  → clean/{modality}.parquet
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 4  Clinical       │  survey sentinels → filter by map → pivot → blocks
│                   │  → clean/clinical_wide.parquet
│                   │  → features/{onboarding,comorbidity,mood,…}.parquet
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 5  Pools          │  post-clean coverage + CGM∩HR overlap
│                   │  → meta/pool_masks.parquet
│                   │  → reports/coverage_survival.csv
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 6  FE View A      │  GREEN watch summaries (person_id + features only)
│    run_fe         │  → features/watch_green.parquet
└─────────┬─────────┘
          ▼
   Path A training (join features ⨝ pool_masks on person_id)
```

View B (Path B seq/aux) is a later stage that reuses `clean/` without re-masking.

---

## 3. Package layout

```
pipeline/
├── config.yaml              # all policy knobs
├── config.py                # load/merge/resolve paths
├── constants.py             # classify_prefix, activity intensity, site TZ
├── io.py                    # participants, row-group I/O, PidParquetWriter
├── pools.py                 # coverage + pool membership
├── validate.py              # report writer
│
├── catalog/
│   └── build_source_map.py  # enumerate + auto-classify source_values
│
├── clean/
│   ├── series.py            # HR, stress, RR, SpO₂, CGM, calories
│   ├── intervals.py         # sleep, physical_activity
│   ├── dedup.py
│   ├── timestamps.py        # UTC→local; pick_window_bounds (Timestamp-safe)
│   ├── windows.py           # shared HR-anchored windows
│   ├── run_modality.py      # stream one modality clean → write
│   └── clinical.py          # pivot, blocks, leakage_column_scan
│
├── fe/
│   └── watch_green.py       # GREEN summary features
│
├── run_catalog.py           # CLI stage 1
├── run_clean.py             # CLI stages 2–5
└── run_fe.py                # CLI stage 6
```

Related: `DATA_AUDIT.md` §B (checklist this pipeline implements).

---

## 4. Stage specifications

### Stage 1 — OMOP catalog (`run_catalog`)

**Inputs:** `clinical/observation.parquet`, `clinical/measurement.parquet`  
**Output:** `meta/source_value_map.csv`

| Column | Meaning |
|---|---|
| `table` | observation \| measurement |
| `prefix` | source_value before first comma |
| `source_value_sample` | example full string |
| `n_rows`, `n_pids` | prevalence |
| `class` | see taxonomy below |
| `block` | onboarding / comorbidity / mood / diet / other_keep / empty |
| `notes` | human lock — non-empty notes preserve class on re-run |

**Class taxonomy (priority order in `constants.classify_prefix`):**

| Class | Fate |
|---|---|
| `hard_exclude` | Never pivoted (label leakage / diabetes-defining) |
| `retinal_drop` | Dropped (retinal metadata / monofilament exam) |
| `labs_upper_bound` | Dropped from primary; optional upper-bound arm later |
| `metadata_drop` | Dropped (instrument timestamps, cognitive batteries, …) |
| `keep_onboarding` | → onboarding block |
| `keep_comorbidity` | → comorbidity block |
| `keep_mood` | → mood block |
| `keep_diet` | → diet block |
| `keep_survey` | → other_keep / survey |
| `borderline` | **Excluded from pivot** until human classifies (safe default) |

Hard-exclude includes (config-driven, non-exhaustive):  
`import_hba1c`, `import_glucose`, `import_insulin`, `import_c_peptide`, `mhterm_dm*`,
`mh_a1c`, `mh_dm_age`, `cmtrt_insln/a1c/glcs/lfst`, `mhoccur_pdr`, **`mhoccur_glc`**,
**`mhoccur_rvo`**, **`dri1`/`dri2`**, diabetes survey date fields.

Metadata drops include `*startts` / `*cmpts` / `*cmpdat` suffixes so instrument
completion times never become features.

**Human edit rule:** only rows with non-empty `notes` are preserved across catalog
re-runs. Everything else re-classifies from current `config.yaml` rules.

---

### Stage 2 — Shared windows (`clean/windows.py`)

**Why shared:** Independent per-modality “best 14d” windows misalign HR vs stress vs
activity vs CGM. Path A would mix non-contemporaneous signals; Path B overlap can go to zero.

**Algorithm:**
1. For each pid, clean HR (sentinels/bounds/dedup only — no window yet).
2. Convert to local time via `clinical_site` → TZ map.
3. `pick_window_bounds` on cleaned local HR timestamps.
4. Write one row per pid to `meta/shared_windows.parquet`.

**Window policy** (`time.window` in config):

| Policy | Behavior |
|---|---|
| `best_coverage` | Contiguous `days`-length window maximizing sample count |
| `first` | First `days` from first valid sample |
| `all` | Full span (no truncation) |

**When truncation fires:**
- Default: only if pre-window span > `long_wear_days` (60)
- Or if `force_all_to_window: true` — everyone sliced to `days`

**Unit safety:** window math uses timezone-aware `Timestamp` arithmetic only.
Never cast `datetime64[ms]` → int64 → `pd.Timestamp(int)` (that path interpreted ms as ns
and wiped year-long wearers into 1969–1970). `run_clean` fails if any window starts before year 2000.

**Site → TZ (mandatory, site×label confound):**

| Site | Zone |
|---|---|
| UAB | America/Chicago |
| UW | America/Los_Angeles |
| UCSD | America/Los_Angeles |

---

### Stage 3 — Series clean (`run_modality`)

Per modality, stream row-groups:

1. **Dedup** — exact rows; timestamp collisions keep `first` (config: `mean`/`last`)
2. **Sentinels → drop null value rows**
3. **Physio bounds**
4. **Corrupt TS** — RR drops year < `min_year` (2000)
5. **UTC → local** columns (`timestamp_local` / `start_time_local` / …)
6. **Apply shared window** from stage 2 (same `[start,end)` for every modality)

| Modality | Sentinel / bound highlights |
|---|---|
| heart_rate | `==0` off; clamp (1, 220] |
| stress | `∈{-1,-2}`; scale **0–100**; clamp [0,100] |
| respiratory_rate | `∈{-1,-2}`; [4,60]; min_year |
| oxygen_saturation | `==0` (no-op here); clamp (50,100]; drop >100 |
| cgm | EGV only; clamp [40,400] |
| sleep | drop bad intervals, optional drop `unknown` stage |
| physical_activity | drop zero/>24h; blank name → `unknown`; `intensity_tier` float64 |
| calories | dedup only; counter resets handled at FE time if needed |

**Interval intensity map (config):**

| activity_name | tier | role |
|---|---|---|
| sedentary | 0 | sedentary mins |
| walking | 1 | light |
| generic | 1 | light |
| running | 2 | MVPA |
| unknown | null | excluded from intensity sums |

Outputs: `clean/{modality}.parquet` (one row-group per pid) + `meta/windows_{modality}.parquet`.

---

### Stage 4 — Clinical pivot

1. Load `source_value_map`; keep only `keep_*` classes.
2. Mask survey codes `{555,777,888,999,99}` on value columns only (not IDs).
3. Pivot long→wide: one row per `person_id`, columns = prefixes.
4. Anthro fixes: waist/hip `< min` → NaN; recompute WHR; BMI > `null_bmi_above` (80) → NaN.
5. Split into block matrices via `split_clinical_blocks`.
6. **Leakage assert** on every block before write.

**Feature matrices contain only `person_id` + feature columns.**  
`label`, `recommended_split`, `clinical_site`, pool flags are **forbidden**
(`FORBIDDEN_FEATURE_COLS` in `clinical.py`). They live in:

- `meta/participant_index.parquet` — age, site, label, split, age_discrepancy
- `meta/pool_masks.parquet` — coverage + pool membership

`age` is an onboarding **feature** (from `participants.age` by default; config
`clinical.age_source: year_of_birth` uses derived age). Site is never a feature.

---

### Stage 5 — Pools

Built from **cleaned, windowed** series (not raw presence).

| Flag | Definition (defaults) |
|---|---|
| `has_hr_valid` | any cleaned HR rows |
| `hr_valid_days` | calendar days with ≥ `min_minutes_per_hr_day` (60) unique minutes |
| `hr_minute_frac` | mean (unique minutes / 1440) on valid HR days |
| `hr_coverage_ok` | `hr_valid_days ≥ 7` and `hr_minute_frac ≥ 0.35` |
| `sleep_nights_ok` | cleaned sleep nights ≥ 7 |
| `wearable_core` | hr_coverage_ok ∧ stress_ok ∧ sleep_ok (nights enforced if `enforce_sleep_nights`) |
| `wearable_core_strict` | always requires ≥7 sleep nights (sensitivity companion) |
| `aux_modalities` | wearable_core ∧ RR ∧ CGM days ≥ 8 |
| `cgm_hr_overlap_hours` | intersection of cleaned HR and CGM time spans (min/max) |
| `aux_eligible` | aux_modalities ∧ overlap ≥ 24h |

Sensitivity columns: `hr_cov_frac_ge_{0.25,0.35,0.5,0.8}`, `aux_overlap_ge_{0,24,72}h`.

**Note:** pool `cgm_hr_overlap_hours` is **span** intersection, not minute-level concurrent wear.
Adequate for Path A / aux gates. View B FE (`grid_5min`) emits **bin-level concurrent** masks
(`wear_bin_valid` ∩ `cgm_bin_valid`); train-time subwindows are wear-density (CGM-free).

`reports/coverage_survival.csv` summarizes n at each gate.

---

### Stage 6 — GREEN features (`run_fe --blocks watch`)

Computed only for pids with `require_pool` (default `wearable_core`).

| Family | Features | Notes |
|---|---|---|
| HR summary | mean, sd, cv, min, max, range, n, nocturnal dip | not clinical HRV |
| RHR | lowest 30-min mean in local 03:00–07:00 | config `rhr_local_hours` |
| Stress | mean, sd, %≥51, %≥76, nocturnal mean, n | % over **valid rest-only** samples |
| Sleep duration | mean hours, \|mean−7.5\|, short/long frac, n nights | U-shape coding |
| Sleep regularity | **SRI** (Phillips adjacent-day pairs), onset SD | not all-pairs |
| Activity | MVPA / light / sedentary min per **HR wear-day**, steps/day | walking = light |
| RAR cosinor | amplitude, mesor, acrophase hour | real local hours; peak at **+φ/w** |

Output: `features/watch_green.parquet` — **`person_id` + features only**.

---

## 5. Commands (full)

```bash
# Full path
.venv/bin/python -m pipeline.run_catalog
.venv/bin/python -m pipeline.run_clean
.venv/bin/python -m pipeline.run_fe --blocks watch

# Smoke
.venv/bin/python -m pipeline.run_clean --max-participants 20 \
  --only heart_rate,stress,sleep,physical_activity,cgm
.venv/bin/python -m pipeline.run_fe --blocks watch

# H3 sensitivity: fixed-length windows for all pids
.venv/bin/python -m pipeline.run_clean --force-all-window

# Clinical blocks only (series already cleaned)
.venv/bin/python -m pipeline.run_clean --skip-series
.venv/bin/python -m pipeline.run_fe --blocks onboarding,comorbidity,mood
```

CLI flags: `--config`, `--max-participants`, `--only`, `--force-all-window`,
`--skip-series`, `--skip-clinical`, `--skip-pools`, `--skip-catalog`.

---

## 6. Config surface (`pipeline/config.yaml`)

All of the following are intended to be swept without code changes.

| Area | Keys | Default |
|---|---|---|
| Paths | `paths.raw_root`, `out_root` | `data/full/AI_READI`, `data/processed` |
| Window | `time.window.policy/days/long_wear_days/force_all_to_window` | best_coverage / 14 / 60 / false |
| TZ | `time.site_tz` | UAB Central, UW/UCSD Pacific |
| RHR hours | `time.rhr_local_hours` | [3, 7] |
| Coverage | `min_hr_valid_days`, `min_hr_minute_frac`, `min_minutes_per_hr_day` | 7 / **0.35** / 60 |
| Sleep gate | `enforce_sleep_nights`, `min_sleep_nights` | **true** / 7 |
| Aux | `min_cgm_days`, `min_cgm_hr_overlap_hours` | 8 / 24 |
| Stress cuts | `sentinels.stress_medium/high` | 51 / 76 |
| Bounds | HR/stress/RR/SpO2/CGM/BMI | see yaml |
| Dedup | `dedup.timestamp_keep` | first |
| Activity map | `intervals.activity.intensity_map` | see §4 |
| Classify | `classify.hard_exclude_*`, retinal, metadata, keep_* | see yaml |
| FE toggles | `features.watch_green.*` | all GREEN families on |
| Runtime | `max_participants`, `only_modalities`, `prefer_hr_participants` | null / null / true |

---

## 7. Outputs tree

```
data/processed/
├── clean/
│   ├── heart_rate.parquet
│   ├── stress.parquet
│   ├── respiratory_rate.parquet
│   ├── oxygen_saturation.parquet
│   ├── sleep.parquet
│   ├── physical_activity.parquet
│   ├── physical_activity_calorie.parquet
│   ├── cgm.parquet
│   └── clinical_wide.parquet
├── meta/
│   ├── source_value_map.csv
│   ├── participant_index.parquet
│   ├── pool_masks.parquet
│   ├── shared_windows.parquet
│   └── windows_{modality}.parquet
├── features/
│   ├── watch_green.parquet      # person_id + GREEN only
│   ├── onboarding.parquet
│   ├── comorbidity.parquet
│   ├── mood.parquet
│   └── clinical_keep_all.parquet
└── reports/
    ├── clean_report.json
    ├── fe_report.json
    ├── coverage_survival.csv
    └── source_value_class_summary.csv
```

`data/` is gitignored; re-run the pipeline to regenerate.

---

## 8. Mapping to DATA_AUDIT §B

| Audit item | Pipeline coverage |
|---|---|
| B1.1 source_value map | Stage 1 catalog |
| B1.2 long→wide pivot | Stage 4 |
| B1.3 no sex/race | Not extracted; person demographics unused |
| B2.1–B2.5 sentinels | Stage 3 series + clinical survey codes |
| B2.6 undocumented sentinel scan | **Not implemented** (open) |
| B3.1–B3.4 physio bounds | Stage 3 |
| B3.6–B3.7 anthro / BMI | Stage 4; `null_bmi_above: 80` |
| B3.8 age discrepancy | Flagged in participant_index; `age_source` honored for onboarding |
| B3.9 calorie resets | Config exists; FE differencing not primary |
| B4.1 UTC→local | Stages 2–3 |
| B4.2 pid 4280 RR | `min_year` drop in RR cleaner |
| B4.3 year-long window | Shared HR window; policy config |
| B4.4 dedup | Stage 3 |
| B4.5 clinical date formats | Partial (study_visit_date for age check only) |
| B4.6 common grid View B | **Done as FE** (`run_fe --blocks grid_5min`, 2026-07-15) — not a clean stage |
| B5 pools post-sentinel | Stage 5 |
| B6.1–B6.2 hard-exclude | Catalog + leakage scan |
| B6.3 condition_occurrence | Not pivoted (self-report dup) |
| B6.4–B6.5 retinal / via* | Catalog rules; via1–3 keep |
| B6.6 mhoccur_* audit | glc/rvo hard-excluded; remaining keep_comorbidity |
| B6.7 post-clean assert | `leakage_column_scan` on every feature write |
| B7 intervals | Stage 3 sleep/activity |
| B8–B9 split / site | In meta only; site×label noted for paper |
| B10 decisions | Encoded as config defaults (see §6) |
| B11 doc fixes | Applied in FEATURES/DATA_STRUCTURE/etc. |

---

## 9. Leakage & safety model

### Hard rules
1. Hard-exclude prefixes never enter the pivot allow-list (double-checked at scan).
2. Feature matrices never contain `label`, `recommended_split`, `clinical_site`,
   `study_group`, or pool flags.
3. Survey instrument timestamps (`*startts`, `*cmpts`, `*cmpdat`) are metadata_drop.
4. `condition_occurrence` is not a second feature source.
5. CGM is never a deployable Path A feature (training supervision for Path B only).

### Soft / residual risks
- ~100+ **borderline** prefixes remain unclassified — excluded until reviewed.
- Other `mhoccur_*` (HTN, etc.) are intentional comorbidity features; re-audit if needed.
- Span-based CGM∩HR overlap overstates concurrent wear for aux.
- Window length still varies for pids with span ≤60d unless `force_all_to_window`.

---

## 10. Locked methodological decisions (v1 defaults)

These are deliberate defaults, not unfinished TODOs. Change via config for sensitivity.

| Decision | Default | Rationale |
|---|---|---|
| Shared window anchor | Cleaned HR | Common reference; stress/CGM align to it |
| Truncate only long wear | span > 60d → 14d best_coverage | Preserve median ~14d study window |
| HR density gate | mean minute_frac ≥ **0.35** | 0.80 unrealistic for consumer PPG on full-day denom |
| Sleep nights | **enforce ≥7** for wearable_core | SRI/onset unstable below that |
| Stress % denominator | valid rest samples | Garmin stress is rest-only |
| SRI definition | Phillips **adjacent-day** pairs | Matches literature thresholds |
| RAR acrophase | peak at **+φ/w** on real local hours | Correct cosinor peak |
| Walking intensity | light (not MVPA) | Conservative; map is config |
| BMI | NaN if >80 | Extreme outliers flagged by audit |
| Feature file contents | no label/site/split | Prevents silent train leaks |

---

## 11. Critiquer-driven fixes already landed

| Issue | Fix |
|---|---|
| `best_coverage` ms/ns unit wipe of year-long pids | Timestamp-only window math + year≥2000 guard |
| Independent per-modality windows | HR-anchored `shared_windows` |
| label/site/split inside feature matrices | Stripped; leakage scan forbids them |
| RAR sign error + `arange` time base | +φ/w on real hours |
| `mhoccur_glc` / `rvo` / `dri*` kept | hard_exclude |
| `paidstartts` / `paidcmpts` as mood features | metadata_drop suffixes |
| SRI all-pairs | adjacent-day Phillips |
| `intensity_tier` null schema crash risk | force float64 |
| MVPA denominator = activity-only days | HR wear-days |
| Sleep nights soft-only | `enforce_sleep_nights: true` |
| Catalog merge fighting config updates | preserve only `notes`-locked rows |
| DST ambiguous `floor("min")` crash in pools | wall-clock day/minute integer keys (no tz floor) |

---

## 12. Full-cohort run snapshot (v1 defaults)

First full pass completed on this machine (canonical `data/full/AI_READI/`, n=2280).
Numbers below are the **landscape after cleaning** — use them as the Path A baseline
pool, not the raw audit ceilings.

### 12.1 Coverage survival (`reports/coverage_survival.csv`)

| Stage | n | frac of 2280 |
|---|---|---|
| all_labeled | 2280 | 1.00 |
| has_hr_valid (post-clean, in shared window) | 1999 | 0.877 |
| hr_coverage_ok (≥7d & minute_frac≥0.35) | 1916 | 0.840 |
| **wearable_core** (Path A default) | **1824** | **0.800** |
| aux_modalities / **aux_eligible** | **1685** | **0.739** |

Shared windows: **1999** pids with an HR-anchored window; **281** empty_hr (no usable
HR after clean); **49** truncated (span&gt;60d → best_coverage 14d). Window starts span
2023-08 → 2025-05 local; **zero pre-2000 windows** (unit-bug regression clean).

Series clean row survival (illustrative): stress/RR drop ~half of raw rows to sentinels
(expected rest-only metrics); HR keeps ~20.7M/22.1M; sleep intervals drop unknown/bad
rows; activity heavily window-clipped to HR span (`truncated` high is normal under
shared window — activity often extends outside the HR analysis window).

### 12.2 `watch_green` matrix

| | |
|---|---|
| Path | `features/watch_green.parquet` |
| Shape | **(1824, 31)** = one row per `wearable_core` |
| Contents | `person_id` + 30 GREEN features only |
| Null rate | **0%** on all columns (full cohort) |
| Leakage scan | clean (`label` / `site` / `split` absent) |
| Duplicate `person_id` | 0 |
| FE wall time | ~12.5 min single-core (per-pid parquet filters; slow but correct) |

Join labels/splits only from `meta/pool_masks.parquet` at train time.

### 12.3 Feature distributions (sanity, full wearable_core)

| Feature | median | p25–p75 | min–max | Flag |
|---|---|---|---|---|
| `rhr` | 57.6 | 52.0–63.7 | 31.6–89.2 | all in [30,120] |
| `hr_mean` | 77.1 | 71.6–83.7 | 50.5–115.3 | OK |
| `hr_nocturnal_dip` | 11.7 | 7.3–16.2 | −12.5–33.0 | some negative = higher night HR |
| `stress_mean` | 50.9 | 37.2–65.5 | 9.7–95.7 | 0–100 scale OK |
| `stress_pct_medium_plus` | 0.51 | 0.31–0.76 | 0.03–1.0 | rest-only denom |
| `stress_pct_high` | 0.19 | 0.10–0.37 | 0.00–1.0 | rest-only denom |
| `sri` | 64.8 | 57.9–70.8 | 3.7–96.5 | continuous; not UKB-calibrated |
| `sleep_duration_mean_hours` | 5.44 | 4.69–6.15 | 1.74–11.5 | short vs self-report lit |
| `sleep_n_nights` | 11 | 10–12 | 6–42 | ~99.9% ≥7 |
| `mvpa_min_per_day` | **0.58** | 0.17–1.81 | 0–89.9 | only `running` = MVPA |
| `light_min_per_day` | 204 | 160–254 | 26–1108 | walking+generic |
| `sedentary_min_per_day` | 1148 | 1076–1263 | 228–**3691** | **~12% &gt; 1440** |
| `steps_mean_per_day` | 7769 | 5870–10411 | 745–31487 | OK |
| `rar_amplitude` | 8.05 | 5.69–10.32 | 0.04–20.4 | OK |
| `rar_mesor` | 75.8 | 70.1–82.4 | 48.9–115.0 | ≈ hr_mean |
| `rar_acrophase_hour` | 14.2 | 12.7–15.6 | 0.76–23.97 | local clock; all in [0,24) |

Range flags (fraction of non-null passing): RHR/stress/SRI/acrophase/MVPA≥0 all **1.0**;
sleep_nights≥7 **0.999**; sedentary≤1440 **0.876**.

### 12.4 Label / split / site after wearable_core filter

| | n |
|---|---|
| label 0 / 1 / 2 / 3 | 636 / 453 / 536 / **199** |
| train / val / test | 1277 / 270 / 277 |
| **train insulin (label 3)** | **80** (full-split train insulin was 105 — filter bites class 3) |
| UAB / UW / UCSD | 681 / 671 / 472 |

**Path A implication:** class weights / focal loss remain mandatory; do not expect the
unfiltered train-insulin count. Report filtered n=1824 (not 2280) in the paper.

### 12.5 Series clean throughput (full run, sequential)

| Modality | pids out/in | rows out/in (approx) | wall |
|---|---|---|---|
| shared HR windows | 1999/2280 | — | ~97s |
| heart_rate | 1999/2104 | 20.7M/22.1M | ~62s |
| stress | 1993/2112 | 20.6M/47.6M | ~72s |
| respiratory_rate | 1992/2113 | 22.8M/47.6M | ~90s |
| oxygen_saturation | 1617/1638 | 3.0M/3.0M | ~21s |
| sleep | 1960/2109 | 0.38M/0.57M | ~18s |
| physical_activity | 1997/2118 | 8.7M/9.6M | ~89s |
| physical_activity_calorie | 1992/1994 | 2.4M/2.4M | ~19s |
| cgm | 1924/2245 | 5.1M/6.2M | ~36s |
| clinical pivot | 2280 | 91 feature cols wide | fast |
| pools (after DST fix) | 2280 mask rows | — | ~4 min |
| watch_green FE | 1824 | 31 cols | ~12.5 min |

Catalog: 136 borderline prefixes still excluded; hard_exclude observation=17,
measurement=5; keep_comorbidity=27; keep_mood=20; keep_onboarding=4+12.

---

## 13. Nuances, flags & config levers (post full-cohort)

Each item is a known behavior or residual risk, not a silent bug. **Config path** is
what to turn when you want a sensitivity run.

### 13.1 Pools & windows

| Nuance | Why it looks this way | Config / action |
|---|---|---|
| `wearable_core` 1824 ≪ 2280 | Needs post-clean HR density + stress + sleep≥7, not raw presence | `coverage.min_hr_*`, `enforce_sleep_nights`, `min_sleep_nights` |
| `aux_eligible` == `aux_modalities` (1685) at 0/24/72h | After shared HR window, almost all CGM∩wearable pids clear ≥72h span overlap | Overlap gate barely binds; Path B still needs **concurrent** overlap later |
| 281 empty_hr | No valid HR after sentinel/window (dead sensor / no HR modality) | Expected; excluded from wearable models |
| 49 long-wear truncated | `span_days_pre > long_wear_days` → best-coverage 14d | `time.window.{policy,days,long_wear_days}` |
| Variable window length for span≤60d | Default does **not** force all pids to 14d | `time.window.force_all_to_window: true` or CLI `--force-all-window` |
| Activity `truncated` count very high | Shared window is HR-anchored; activity often longer than HR window | By design; not data loss of “good” HR-aligned activity |
| Stress/RR row counts ~half of raw | Sentinels −1/−2 are majority (rest-only) | Expected; %stress features use valid-rest denominator |

### 13.2 GREEN feature quirks

| Nuance | Observation (full cohort) | Config / landscape |
|---|---|---|
| **MVPA median ~0.6 min/day** | Only `running` is intensity tier 2; walking is light | `intervals.activity.intensity_map` — set `walking: 2` for “steps-like MVPA” sensitivity; literature MVPA will look weak under current map |
| **`sedentary_min_per_day` can exceed 1440** (~12% of pids; max ~3691) | Garmin sedentary intervals can overlap / stack; sum is not “exclusive minutes in day”; denominator is HR wear-days not activity calendar days | Trees OK; optional later: non-overlap merge or clamp to 1440. Not `1440 − active` (FEATURES ideal) |
| Stress `%medium+` / `%high` | Denominator = valid rest-only samples, not 24h wear | Matches Garmin semantics; do not interpret as clock-time fraction |
| SRI median ~65 | Adjacent-day Phillips-style; literature “regular” often &gt;80 | Thresholds from UKB not calibrated to Vivosmart; use continuous SRI |
| Sleep duration mean ~5.5h | Short vs self-report literature 7–8h | Device/staging + “asleep stages only”; U-shape uses `sleep_target_hours: 7.5` via `sleep_duration_dev_7_5` |
| `sleep_n_nights` min 6 on a tiny tail | Pool night count vs FE night clustering edge | Essentially all core pids ≥7; ignore |
| RAR acrophase median ~14.2h | Local clock peak of cosinor on hourly HR | Requires local TZ (site map); do not recompute in UTC |
| RHR median ~58 | Local 03:00–07:00 lowest 30-min mean | `time.rhr_local_hours` |
| Zero nulls in watch_green | Core pool requires modalities that feed all GREEN families | If you loosen pool gates, expect NaNs and need imputation policy for non-trees |

### 13.3 Clinical / catalog

| Nuance | Observation | Config / action |
|---|---|---|
| **136 borderline** OMOP prefixes | Still excluded from pivot (safe) | Human review `meta/source_value_map.csv`; lock with non-empty `notes` |
| `mhoccur_glc` / `rvo` / `dri*` | hard_exclude | `classify.hard_exclude_prefixes` |
| Remaining `mhoccur_*` | keep_comorbidity (HTN, etc.) | Re-audit before claiming “no diabetes leakage in survey block” |
| `paidstartts` / `*cmpts` | metadata_drop | `classify.metadata_drop_suffixes` |
| Monofilament `msslffl` / `mssrffl` | retinal_drop (diabetes-targeted exam) | B10.2 closed as drop |
| BMI NaN if &gt;80 | Extreme outliers removed from onboarding | `bounds.null_bmi_above` |
| No sex/race | `person.parquet` blank | Cannot stratify fairness by sex/race; onboarding = age/BMI/waist/FH/BP/smoking only |
| Site×label | UAB still enriched in filtered core (681/1824) | Never put `clinical_site` in X; report site×label table |

### 13.4 Runtime / engineering

| Nuance | Observation | Guidance |
|---|---|---|
| Pools DST crash (fixed) | `dt.floor` on ambiguous fall-back local times | Do not reintroduce tz-aware `floor` for day/minute counts |
| FE ~12 min / 1824 pids | Per-pid parquet filter seeks | Acceptable one-shot; speedup = single-pass FE, not multi-core first |
| Series clean ~8 min | 8 modalities sequential after shared windows | Multi-core modality pool only worth it if re-cleaning often (16 GB RAM risk) |
| Resume after pools/FE fail | Series already on disk | `run_clean --skip-series --skip-catalog --skip-clinical` then `run_fe` |
| `prefer_hr_participants` | Only affects `max_participants` smoke sampling | Full run uses all 2280 |

### 13.5 Path A training checklist (from this landscape)

1. Load `watch_green` ⨝ `pool_masks` on `person_id`; filter `recommended_split`.  
2. **X = feature columns only** — never site/split/pool flags.  
3. **Class weights** (train insulin n=80). Lock before any feature selection.  
4. Honest target still 4-class ~0.72–0.75; filtered n=1824 is the cohort.  
5. SHAP guardrail still applies when survey blocks are added later.  
6. Optional sensitivities (separate runs, same split):  
   - `force_all_to_window: true`  
   - `walking: 2` in intensity_map  
   - `min_hr_minute_frac` ∈ {0.25, 0.35, 0.50} via pool sensitivity cols already on masks  
   - `enforce_sleep_nights: false` vs `wearable_core_strict`

---

## 14. Open work (not blocking Path A floor)

1. Human pass on remaining **136 borderline** prefixes in `source_value_map.csv`
2. Sensitivity: `force_all_to_window: true` vs default (feature stability vs wear length)
3. Optional clamp / non-overlap fix for `sedentary_min_per_day` &gt; 1440
4. MVPA map sensitivity (`walking` as moderate)
5. ~~View B alignment for Path B~~ → **done as FE** (`grid_5min`; concurrent bin masks; B4 training)
6. B2.6 undocumented sentinel scan
7. Optional labs upper-bound matrix
8. FE single-pass speedup if re-FE becomes frequent

---

## 15. How agents should use this

- **Run cleaning:** commands in Quick start / §5; knobs in `config.yaml`.
- **Change a threshold:** edit `config.yaml`, re-run affected stage. Do not hardcode in Python.
- **Add a feature:** implement in `fe/`, keep person_id+features only, run leakage scan.
- **Add a clinical field:** classify in catalog rules or edit `source_value_map.csv`
  with a non-empty `notes` lock; re-run clinical stages.
- **Never** put `clinical_site` or `recommended_split` into X unless it is an explicit
  confound ablation reported as such.
- **Full-cohort landscape / flags:** §12–§13 — read before inventing new pool thresholds.
- **Authority on conflict:** `DATA_AUDIT.md` (what data is) → **`CLEANING.md`** (how we clean)
  → `FEATURES.md` (what we engineer) → `Training.md` (how we train). If code and this doc
  diverge, fix the lagging side.

---

## 16. Success criteria (“cleaning done”)

1. `meta/shared_windows.parquet` exists; no window starts before year 2000  
2. `meta/pool_masks.parquet` documents `wearable_core` / `aux_eligible` n  
3. `features/watch_green.parquet` has one row per wearable_core pid; leakage scan clean  
4. Clinical blocks have no hard-exclude, no site/label/split, no `*startts` metadata  
5. `reports/coverage_survival.csv` matches order-of-magnitude audit ceilings  
6. Re-run is idempotent on this machine without Drive  

**v1 full-cohort status:** criteria 1–6 met (wearable_core=1824, watch_green 1824×31,
0% nulls, 0 pre-2000 windows). Residual items are sensitivities / Path B (§14), not
blockers for Path A floor training.

