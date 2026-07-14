# Processed data — consumer contract

> **For training / analysis agents.** How to *use* cleaned outputs under
> `data/processed/`. Rebuild rules, stage design, and config levers live in
> **`CLEANING.md`** — open that only when re-cleaning or changing pipeline policy.

Raw canonical (read-only): `data/full/AI_READI/` — see `DATA_STRUCTURE.md`.  
ML methodology: `Training.md`. Feature definitions: `FEATURES.md`.

---

## 1. Layout

```
data/processed/                    # gitignored; regenerate via pipeline
├── clean/                         # row-cleaned, local-time, HR-windowed series
│   ├── heart_rate.parquet
│   ├── stress.parquet
│   ├── respiratory_rate.parquet
│   ├── oxygen_saturation.parquet
│   ├── sleep.parquet
│   ├── physical_activity.parquet
│   ├── physical_activity_calorie.parquet
│   ├── cgm.parquet
│   └── clinical_wide.parquet      # OMOP keep-* pivoted wide
├── meta/
│   ├── pool_masks.parquet         # labels, splits, pool flags, coverage stats
│   ├── participant_index.parquet  # age, site, label, split, age_discrepancy
│   ├── shared_windows.parquet     # HR-anchored [start,end) per pid
│   ├── source_value_map.csv       # OMOP prefix classification
│   └── windows_{modality}.parquet
├── features/
│   ├── watch_green.parquet        # Path A floor: person_id + GREEN only
│   ├── onboarding.parquet         # person_id + age/BMI/BP/FH… (no site/label)
│   ├── comorbidity.parquet
│   ├── mood.parquet
│   └── clinical_keep_all.parquet
└── reports/
    ├── coverage_survival.csv
    ├── clean_report.json
    └── fe_report.json
```

If `data/processed/` is missing: run the pipeline (`CLEANING.md` Quick start).

---

## 2. Path A train recipe (default)

```python
import pandas as pd

feats = pd.read_parquet("data/processed/features/watch_green.parquet")
meta  = pd.read_parquet("data/processed/meta/pool_masks.parquet")

df = feats.merge(
    meta[
        [
            "person_id",
            "label",                 # 0–3
            "recommended_split",     # train | val | test
            "clinical_site",         # report only — NOT default feature
            "wearable_core",
            "wearable_core_strict",
            "aux_eligible",
        ]
    ],
    on="person_id",
    how="inner",
)

# Default Path A cohort is already wearable_core (watch_green rows)
assert df["wearable_core"].all()

feature_cols = [c for c in feats.columns if c != "person_id"]
train = df[df["recommended_split"] == "train"]
X_train, y_train = train[feature_cols], train["label"]
```

### Hard rules for consumers
1. **X = feature columns only** — never `label`, `recommended_split`, `clinical_site`,
   pool flags, or `study_group` as predictors (site is label-confounded).
2. **Split by `recommended_split`**, never random row split / k-fold that mixes people.
3. **Class weights** — train insulin (label 3) is **80** after filter (not 105).
4. Report cohort **n=1824** wearable_core (not full 2280) unless you intentionally
   loosen gates and re-FE.

---

## 3. What to expect inside key files

### `features/watch_green.parquet` (Path A floor)

| | |
|---|---|
| Rows | **1824** (= `wearable_core`) |
| Cols | `person_id` + 30 GREEN features (31 total) |
| Nulls | **none** (full cohort under current gates) |
| Labels? | **No** — join `pool_masks` |

Feature columns (names only; definitions in `FEATURES.md` / `CLEANING.md`):

```
hr_mean, hr_sd, hr_cv, hr_min, hr_max, hr_range, hr_n, hr_nocturnal_dip, rhr,
rar_amplitude, rar_mesor, rar_acrophase_hour,
stress_mean, stress_sd, stress_pct_medium_plus, stress_pct_high, stress_n,
stress_nocturnal_mean,
sri, sleep_onset_sd_hours, sleep_duration_mean_hours, sleep_duration_dev_7_5,
sleep_short_frac, sleep_long_frac, sleep_n_nights,
mvpa_min_per_day, light_min_per_day, sedentary_min_per_day, steps_mean_per_day,
activity_n_days
```

**Quick distribution context (medians, full core):** RHR ~58 · stress_mean ~51 ·
SRI ~65 · sleep duration ~5.4 h · steps/day ~7.8k · MVPA ~0.6 min/day (only
`running` counts as MVPA) · acrophase ~14.2 h local.  
`sedentary_min_per_day` can exceed 1440 (~12% of pids) — overlapping Garmin
intervals; trees tolerate it.

### `meta/pool_masks.parquet` (one row per labeled pid, n=2280)

Important columns:

| Column | Meaning |
|---|---|
| `person_id` | join key |
| `label` | 0 healthy → 3 insulin |
| `recommended_split` | train / val / test |
| `clinical_site` | UAB / UW / UCSD — confound, not a default feature |
| `age` | from participants |
| `wearable_core` | Path A default membership |
| `wearable_core_strict` | always requires ≥7 sleep nights |
| `aux_eligible` | Path B aux pool (post-clean + overlap) |
| `hr_valid_days`, `hr_minute_frac`, `cgm_hr_overlap_hours`, … | coverage diagnostics |
| `hr_cov_frac_ge_*`, `aux_overlap_ge_*h` | sensitivity flags |

### Cohort sizes after clean (v1 defaults)

| Pool | n |
|---|---|
| all labeled | 2280 |
| has_hr_valid | 1999 |
| hr_coverage_ok | 1916 |
| **wearable_core** / watch_green | **1824** |
| **aux_eligible** | **1685** |
| watch_green label 0/1/2/3 | 636 / 453 / 536 / **199** |
| train / val / test (core) | 1277 / 270 / 277 |
| **train insulin** | **80** |

### Clinical feature blocks (Path A ablations)

| File | Contents |
|---|---|
| `onboarding.parquet` | `person_id`, age, BMI, waist, BP, family hx, … — **no site/label** |
| `comorbidity.parquet` | `mhoccur_*` keeps (not glc/pdr/rvo — hard-excluded) |
| `mood.parquet` | CES-D items, PAID score, via1–3 — no `*startts` metadata |
| `smoking.parquet` | **one-off** extract: `smoke_ever`, `smoke_current` from raw `susmk*` (not pipeline FE) |

Join to `watch_green` on `person_id` for block ablation; still pull `label` /
`recommended_split` only from `pool_masks`.

**Modeling status:** Path A tabular ladder on these blocks is **frozen** (2026-07-14).
Deployable secondary stack used mood + onboarding; comorbidity core failed the decision bar.
Post-freeze C1 sensitivities (smoking / `mhoccur_obs` / via1–3 / joint) also **bar-fail**.
See `training/path_a_blocks/REPORT_A_WRAP.md`. `diet` block (if produced by FE) was not run.
Build smoking: `python -m training.path_a_blocks.build_smoking_features`.

### `clean/*.parquet` (sequence / Path B later)

- Timestamps are **local** (`timestamp_local` or `start_time_local`) plus original UTC where kept.
- Already **deduped, sentinel-masked, HR-windowed** (shared window per pid).
- One row-group per `person_id` (filter-friendly).
- CGM is train-time supervision only — not a watch-only deployable feature.
- Do not re-apply raw sentinels; do not re-window unless you change policy and re-run clean.

---

## 4. What this is *not*

| Need | Where |
|---|---|
| How cleaning works / change thresholds | `CLEANING.md` + `pipeline/config.yaml` |
| Why a feature exists / literature | `FEATURES.md` |
| Models, metrics, Path B methods | `Training.md` |
| Raw AI-READI schema | `DATA_STRUCTURE.md` |
| Empirical audit findings | `DATA_AUDIT.md` |

---

## 5. Rebuild pointer (only if outputs missing/stale)

```bash
.venv/bin/python -m pipeline.run_catalog
.venv/bin/python -m pipeline.run_clean
.venv/bin/python -m pipeline.run_fe --blocks watch
```

Details, config defaults, and residual flags: **`CLEANING.md`**.
