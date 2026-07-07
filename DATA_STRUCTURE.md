# AI-READI Canonical Data Structure

Reference for the converted AI-READI T2D dataset. Self-contained: an AI/agent can work from this without exploring the filesystem or asking for commands.

## TL;DR

- Two variants with an **identical internal layout** — swap only the variant segment:
  - `mini` — 100 participants (subset for pipeline testing)
  - `full` — 2,280 participants (complete dataset)
- Everything is **Apache Parquet (zstd)** for tabular/wearable data, plus one **NumPy `.npy`** for ECG waveforms.
- One file per modality/table (Drive-friendly). **One row group per participant** in every parquet → stream/filter by participant without loading the whole file.
- Timestamps are tz-aware UTC (`datetime64[ns, UTC]`).
- Stored on **Google Drive** at `AI_READI/{mini|full}/AI_READI/`.

## Folder structure

```
AI_READI/                                   # root for one variant (mini or full)
│
├── clinical/                               # OMOP CDM tables (verbatim minus index cols)
│   ├── person.parquet
│   ├── condition_occurrence.parquet
│   ├── measurement.parquet
│   ├── observation.parquet                 # demographics, comorbidities, survey-style features
│   ├── procedure_occurrence.parquet
│   └── visit_occurrence.parquet
│
├── environment/
│   └── environment.parquet                 # LeeLab Anura environmental sensor, per-participant rows
│
├── dexcom/
│   └── cgm.parquet                         # Dexcom G6 continuous glucose monitor readings
│
├── garmin/                                 # Garmin Vivosmart 5 wearable, one parquet per modality
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
- `label` (int 0–3) — **T2D label**: `0`=healthy, `1`=pre_diabetes_lifestyle_controlled, `2`=oral_medication_and_or_non_insulin_injectable_medication_controlled, `3`=insulin_dependent.
- `age`, `clinical_site`, `study_visit_date`, `recommended_split` (`train`/`val`/`test`).
- **Split by `person_id`, never by row**, to avoid leakage.

### clinical/*.parquet — OMOP CDM (6 standard tables)
Standard OMOP columns. `observation.parquet` holds demographics, comorbidities, depression/distress scores, lifestyle, family history — the "survey" feature block. Unchanged from source except dropping stray index columns.

### environment/environment.parquet
Per-participant environmental sensor rows. Columns: `person_id`, `ts`, then sensor channels: `lch0`–`lch3`, `lch6`–`lch11`, `pm1`, `pm2.5`, `pm4`, `pm10`, `hum`, `temp`, `voc`, `nox`, `screen`, `ff`, `inttemp` (all float32).

### dexcom/cgm.parquet
Continuous glucose readings. Columns: `person_id`, `timestamp`, `blood_glucose` (float), `unit`, `event_type`, `source_device_id`, `transmitter_id`. ~1 reading / 5 min.

### garmin/*.parquet
One parquet per modality. All have `person_id` + a timestamp column; value/unit columns vary:
- `heart_rate`: `timestamp`, `heart_rate`, `unit`
- `oxygen_saturation`: `timestamp`, `oxygen_saturation`, `unit`, `measurement_method`
- `respiratory_rate`: `timestamp`, `respiratory_rate`, `unit`
- `stress`: `timestamp`, `stress_level`, `unit`
- `sleep`: `start_time`, `end_time`, `sleep_stage_state` (interval, not instant)
- `physical_activity`: `start_time`, `end_time`, `activity_name`, `steps`, `step_unit` (interval)
- `physical_activity_calorie`: `timestamp`, `calories`, `unit`

Not every participant has every modality (some wore/used the device less) — absence of a participant's row group in a modality file means no data for them there.

### ecg/
- `recordings.npy` — `float32` array, shape `(N, 12, max_len)`: N recordings, 12 leads, each recording NaN-padded to `max_len`. Values are **physical mV** (not bit-exact vs the int16 digital source; precision ~1e-6 mV, below noise).
- `index.parquet` — `rec_id`, `person_id`, `fs`, `sig_len` (**true length, use this not the padding**), `n_sig`, `sig_name`, `units`, `comments`.

### metadata/dataset_info.json
Schema notes, the sentinel rules below, the label map, ECG shape, and timestamp convention.

## Excluded (not in this dataset)

All retinal imaging datatypes are intentionally excluded:
- `retinal_flio`, `retinal_oct`, `retinal_octa`, `retinal_photography` (DICOM, large, out of scope)

Only the five non-retinal datatypes are present: `clinical_data`, `environment`, `wearable_blood_glucose`, `wearable_activity_monitor`, `cardiac_ecg`.

## Sentinel values (mask before ML)

Raw values are preserved verbatim; ML code must convert these to NaN:
- `heart_rate == 0` → sensor off
- `stress in {-1, -2}` → invalid
- `respiratory_rate in {-1, -2}` → invalid
- `oxygen_saturation == 0` → invalid

## How to access

The canonical lives on Google Drive. Point your code at the variant root and the internal paths are identical.

**Google Drive path:** `AI_READI/{mini|full}/AI_READI/`

**Colab (mount Drive):**
```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/AI_READI/full/AI_READI'   # or .../mini/...
```

**rclone (the canonical was uploaded with rclone; remote name `gdrive`):**
```bash
# copy down to local disk first (recommended — don't train over a Drive mount)
rclone copy gdrive:AI_READI/full/AI_READI ./AI_READI --progress
# or mount (slower for random access; fine for sequential reads)
rclone mount gdrive:AI_READI/full/AI_READI ./AI_READI --vfs-cache-mode reads --daemon
```

**Read in Python (any location):**
```python
import pyarrow.parquet as pq, pandas as pd, numpy as np
ROOT = './AI_READI'                     # or the Drive path above
# whole modality
hr = pd.read_parquet(f'{ROOT}/garmin/heart_rate.parquet')
# one participant only (row-group scan — cheap, no full load)
one = pq.read_table(f'{ROOT}/garmin/heart_rate.parquet', filters=[('person_id','=',<pid>)]).to_pandas()
# ECG (memmap, don't load all into RAM)
arr = np.load(f'{ROOT}/ecg/recordings.npy', mmap_mode='r')
idx = pd.read_parquet(f'{ROOT}/ecg/index.parquet')
```

**Tip:** For training, copy the canonical to local disk once (Colab `/content` or a VM), then read from local — Drive mounts are slow for random access and can stall on many-small-file patterns. This layout avoids that trap (few large files), but local is still faster.
