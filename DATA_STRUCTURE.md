# AI-READI Canonical Data Structure

Reference for the converted AI-READI T2D dataset. Self-contained: an AI/agent can work from this
without exploring the filesystem or asking for commands.

**Related:** empirical quirks and cleaning plan in `DATA_AUDIT.md` (source of truth for what the
data *actually* contains); feature rules in `FEATURES.md`; access/compute in `COMPUTE.md`.

## TL;DR

- Two variants with an **identical internal layout** — swap only the variant segment:
  - `mini` — 100 participants (subset for pipeline testing)
  - `full` — 2,280 participants (complete dataset)
- Everything is **Apache Parquet (zstd)** for tabular/wearable data, plus one **NumPy `.npy`** for
  ECG waveforms.
- One file per modality/table (Drive-friendly). **One row group per participant** in every
  garmin/dexcom parquet → stream/filter by participant without loading the whole file.
  Clinical tables are small single-row-group files.
- **Timestamps:** garmin + dexcom are tz-aware UTC (`timestamp[ms, tz=UTC]`). Clinical date columns
  are **`large_string` with three formats** — not tz-aware (see §Timestamps).
- Stored on **Google Drive** at `AI_READI/{mini|full}/AI_READI/` (rclone remote on this machine:
  `gdrive_zyrus:`). Relevant full subset also local: `data/full/AI_READI/` (~784 MiB).

## Folder structure

```
AI_READI/                                   # root for one variant (mini or full)
│
├── clinical/                               # OMOP CDM tables (verbatim minus index cols)
│   ├── person.parquet                      # ⚠ demographics blank — see notes
│   ├── condition_occurrence.parquet        # self-report terms, NOT ICD codes
│   ├── measurement.parquet                 # long-format source_value rows
│   ├── observation.parquet                 # long-format: survey/comorbidity/lifestyle
│   ├── procedure_occurrence.parquet
│   └── visit_occurrence.parquet
│
├── environment/
│   └── environment.parquet                 # LeeLab Anura environmental sensor (usually dropped)
│
├── dexcom/
│   └── cgm.parquet                         # Dexcom G6 CGM readings
│
├── garmin/                                 # Garmin Vivosmart 5, one parquet per modality
│   ├── heart_rate.parquet
│   ├── oxygen_saturation.parquet
│   ├── respiratory_rate.parquet
│   ├── stress.parquet
│   ├── sleep.parquet
│   ├── physical_activity.parquet
│   └── physical_activity_calorie.parquet
│
├── ecg/
│   ├── recordings.npy                      # 12-lead ECG waveforms, float32 mV, NaN-padded
│   └── index.parquet                       # per-recording metadata + true signal length
│
└── metadata/
    ├── participants.parquet                # participant index + T2D label + train/val/test split
    ├── dataset_info.json                   # schema notes, sentinels, label map, ECG shape
    └── manifests/                          # original AI-READI file manifests (TSV), copied verbatim
        ├── file-manifest.tsv
        ├── clinical_data_manifest.tsv
        ├── environment_manifest.tsv
        ├── wearable_activity_monitor_manifest.tsv
        ├── wearable_blood_glucose_manifest.tsv
        └── cardiac_ecg_manifest.tsv
```

## What each file contains

### metadata/participants.parquet — the index
- `person_id` (int) — joins to every other table.
- `study_group` (str) — original group name.
- `label` (int 0–3) — **T2D label**: `0`=healthy, `1`=pre_diabetes_lifestyle_controlled,
  `2`=oral_medication_and_or_non_insulin_injectable_medication_controlled, `3`=insulin_dependent.
- `age`, `clinical_site` (`UAB` / `UW` / `UCSD`), `study_visit_date` (ISO string `YYYY-MM-DD`),
  `recommended_split` (`train`/`val`/`test`).
- **Split by `person_id`, never by row**, to avoid leakage.
- Full label counts: 0=776, 1=560, 2=686, 3=258. Split sizes: train 1576 / val 352 / test 352
  (train insulin = 105).

### clinical/*.parquet — OMOP CDM (6 standard tables)

**Long-format, not wide.** `observation` (707k rows, 361 unique `observation_source_value`) and
`measurement` (242k rows, 108 unique `measurement_source_value`) store features as **rows keyed by
`*_source_value`** (friendly `"code, description"` strings). Feature extraction = filter by
source_value prefix + pivot to one row per `person_id`. Friendly names in `FEATURES.md`
(`paidscore`, `mhoccur_*`, `cmtrt_*`, …) are those prefixes (before the comma), not column names.

- `observation.parquet` — comorbidities, depression/distress (CES-D, PAID), lifestyle, family
  history, meds self-report — the "survey" feature block.
- `measurement.parquet` — labs, anthropometrics, clinical scores (many leakage / retinal / upper-bound).
- `condition_occurrence.parquet` — **not ICD-coded.** All 30 `condition_source_value`s are
  self-report terms duplicating observation (`mhterm_dm2`, `mhoccur_hbp`, …). Treat as a shared
  source-value namespace with observation; do not build an ICD filter (zero E10–E13 codes).
- `person.parquet` — **demographics are blank / unusable for sex/race/ethnicity.**
  `gender/race/ethnicity_concept_id = 0`, source values `' '` (space), `birth_datetime` epoch
  placeholder for all 2,280. Only `year_of_birth` is real. Use `participants.age` for age.
  Sex/race features are **not available** in this release.

Survey numeric sentinels in `value_as_number` / `value_as_string`: `{555, 777, 888, 999, 99}` → NaN.
Do **not** mask these codes in ID columns (`observation_id`, `visit_occurrence_id`) — coincidental.

### environment/environment.parquet
Per-participant environmental sensor rows. Columns: `person_id`, `ts`, then sensor channels:
`lch0`–`lch3`, `lch6`–`lch11`, `pm1`, `pm2.5`, `pm4`, `pm10`, `hum`, `temp`, `voc`, `nox`,
`screen`, `ff`, `inttemp` (all float32). Usually dropped (no realistic watch-side deployment).

### dexcom/cgm.parquet
Continuous glucose readings. Columns: `person_id`, `timestamp` (UTC), `blood_glucose` (float),
`unit` (100% mg/dL), `event_type` (100% `EGV` — no calibration/carb events), `source_device_id`,
`transmitter_id`. ~1 reading / 5 min. 2,245 participants; median ~9.9 days.

### garmin/*.parquet
One parquet per modality. All have `person_id` + a timestamp column; value/unit columns vary:
- `heart_rate`: `timestamp`, `heart_rate`, `unit` — 1-min instantaneous PPG (not beat-to-beat RR)
- `oxygen_saturation`: `timestamp`, `oxygen_saturation`, `unit`, `measurement_method` — ~71.8% coverage
- `respiratory_rate`: `timestamp`, `respiratory_rate`, `unit` — ~2× denser than HR
- `stress`: `timestamp`, `stress_level`, `unit` — **scale 0–100** (Garmin/Firstbeat bands:
  0–25 resting / 26–50 low / 51–75 medium / 76–100 high), rest-only
- `sleep`: `start_time`, `end_time`, `sleep_stage_state` (interval, not instant)
- `physical_activity`: `start_time`, `end_time`, `activity_name`, `steps`, `step_unit` (interval)
- `physical_activity_calorie`: `timestamp`, `calories`, `unit` (counter; can reset — non-monotonic)

Not every participant has every modality — absence of a participant's row group means no data for
them there. Modalities are **not on a shared time grid**; sequence/aux views need explicit resample.

Wear windows: median ~14.6d HR / ~10 nights sleep; **~63 participants have true >60d (year-long)
wear** — not an aggregate artifact. Dedup exact + timestamp duplicates before aggregating
(worst offender: pid 1366). See `DATA_AUDIT.md` A.4 / A.6.

### ecg/
- `recordings.npy` — `float32` array, shape `(N, 12, max_len)`: N recordings, 12 leads, each
  recording NaN-padded to `max_len`. Values are **physical mV** (not bit-exact vs the int16 digital
  source; precision ~1e-6 mV, below noise).
- `index.parquet` — `rec_id`, `person_id`, `fs`, `sig_len` (**true length, use this not the
  padding**), `n_sig`, `sig_name`, `units`, `comments`.
- Deployable-out; optional paper upper-bound arm (clinical HRV vs Garmin stress proxy).

### metadata/dataset_info.json
Schema notes, sentinel rules (as written at convert time), label map, ECG shape, timestamp
convention. **Note:** convert-time notes are incomplete vs the full audit (stress scale, clinical
date strings, person demographics) — prefer this doc + `DATA_AUDIT.md` for ML cleaning rules.

## Excluded (not in this dataset)

All retinal imaging datatypes are intentionally excluded:
- `retinal_flio`, `retinal_oct`, `retinal_octa`, `retinal_photography` (DICOM, large, out of scope)

Only the five non-retinal datatypes are present: `clinical_data`, `environment`,
`wearable_blood_glucose`, `wearable_activity_monitor`, `cardiac_ecg`.

**However:** retinal *metadata* still leaked into `observation` / `measurement` as source_value
rows (`rt*`, clinical `via*`, `mlcs*`, `plcs*`, …) — filter at feature extraction
(`FEATURES.md` §3; keep self-report `via1–3` only).

## Sentinel values (mask before ML)

Raw values are preserved verbatim; ML code must convert these to NaN / clamp:

| Field | Rule | Notes (full cohort) |
|---|---|---|
| `heart_rate == 0` | → NaN (sensor off) | 5.5% of rows; 105 pids zero-valid after mask |
| `stress_level ∈ {-1, -2}` | → NaN (invalid / not-at-rest) | 56.5% of rows (expected); scale of **valid** values is **0–100** |
| `respiratory_rate ∈ {-1, -2}` | → NaN | 50.9% of rows; 114 pids zero-valid after mask |
| `oxygen_saturation == 0` | → NaN (safety) | **0 occurrences** in full data (no-op); real issue is `>100` clamp (191 rows) + 28% missing pids |
| `oxygen_saturation > 100` | → NaN | max 101 |
| `heart_rate > 220` | → NaN | 2 rows |
| `stress_level > 100` | → NaN | 2 rows (max 101) |
| CGM `blood_glucose` outside [40, 400] | clamp or NaN | residual High/Low markers |
| Survey `value_as_number` / `value_as_string` ∈ {555, 777, 888, 999, 99} | → NaN | not in ID columns |
| `waist_vsorres` / `hip_vsorres` == 0 | → NaN | 1 pid each (impossible 0 cm) |

**Stress high-threshold features:** use ≥51 (medium+) or ≥76 (high) — never a 0–17-era ≥7 cut.

## Timestamps & time zones

| Source | Type | Format / convention |
|---|---|---|
| garmin `timestamp` / sleep·activity start/end | `timestamp[ms, tz=UTC]` | tz-aware UTC |
| dexcom `timestamp` | `timestamp[ms, tz=UTC]` | tz-aware UTC |
| `participants.study_visit_date` | `large_string` | ISO `%Y-%m-%d` |
| `observation.observation_date` | `large_string` | M/D/YY `%m/%d/%y` |
| `observation.observation_datetime` | `large_string` | `M/D/YY 0:00` (midnight placeholder — time not real) |
| `measurement.measurement_date` | `large_string` | ISO `%Y-%m-%d` |

**UTC → local is mandatory** before any circadian / nocturnal / SRI / RHR / cosinor feature.
Site → zone: **UAB → US/Central**, **UW → US/Pacific**, **UCSD → US/Pacific**. Site is
label-confounded (UAB holds ~53% of insulin cases) — computing those features in UTC injects a
site-correlated artifact.

## Pools (quick reference)

Report **post-filter** n, not raw presence. Detail: `DATA_AUDIT.md` A.6 / `FEATURES.md` §1.

| Pool | Raw | Effective (approx) |
|---|---|---|
| All labeled | 2,280 | — |
| Wearable core (HR∩stress∩sleep) | 2,052 | ≤1,983 after sentinel zero-valid |
| Aux (HR∩stress∩RR∩sleep∩CGM) | 2,034 | ≤1,963 → **≤1,921** after ≥24h CGM↔HR overlap |
| CGM-haves | 2,245 | CGM∩HR 2,085 raw; 51 ≤0h overlap |
| SpO₂ present | 1,638 (71.8%) | do not gate primary pool |

## How to access

Canonical lives on Google Drive. Relevant full subset is also on this machine.

**Local (this repo):**
```text
data/full/AI_READI/     # ~784 MiB relevant (garmin+dexcom+clinical+metadata)
```

**Google Drive path:** `AI_READI/{mini|full}/AI_READI/`

**rclone (configured remote on this machine is `gdrive_zyrus:`, not `gdrive:`):**
```bash
# copy down to local disk first (recommended — don't train over a Drive mount)
rclone copy gdrive_zyrus:AI_READI/full/AI_READI ./data/full/AI_READI --progress
# or mount (slower for random access; fine for sequential reads)
rclone mount gdrive_zyrus:AI_READI/full/AI_READI ./AI_READI --vfs-cache-mode reads --daemon
```

`convert_pipeline.py` still defaults `GDRIVE_REMOTE` to `gdrive:…` — set
`AI_READI_GDRIVE=gdrive_zyrus:AI_READI/{full|mini}/AI_READI` before re-archiving
(see `COMPUTE.md`).

**Colab (mount Drive):**
```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/AI_READI/full/AI_READI'   # or .../mini/...
# better: copy once to /content then read local
```

**Read in Python (any location):**
```python
import pyarrow.parquet as pq, pandas as pd, numpy as np
ROOT = 'data/full/AI_READI'              # or the Drive path above

# whole modality
hr = pd.read_parquet(f'{ROOT}/garmin/heart_rate.parquet')

# one participant only (row-group scan — cheap, no full load)
one = pq.read_table(
    f'{ROOT}/garmin/heart_rate.parquet',
    filters=[('person_id', '=', pid)],
).to_pandas()

# clinical long → filter by source_value prefix then pivot
obs = pd.read_parquet(f'{ROOT}/clinical/observation.parquet')
# obs['observation_source_value'] looks like "paidscore, PAID total score"

# ECG (memmap, don't load all into RAM)
arr = np.load(f'{ROOT}/ecg/recordings.npy', mmap_mode='r')
idx = pd.read_parquet(f'{ROOT}/ecg/index.parquet')
```

**Tip:** For training, copy the canonical to local disk once (Colab `/content` or a VM), then read
from local — Drive mounts are slow for random access. This layout avoids the many-small-file trap
(few large files), but local is still faster.

## Where cleaning detail lives

Do **not** treat this structure doc as a complete cleaning plan. For sentinels prevalence, dups,
year-long wearers, leakage source_value lists, post-sentinel pool math, and the 50+ item checklist,
use **`DATA_AUDIT.md`**. For which fields are features vs leakage, use **`FEATURES.md` §3**.
