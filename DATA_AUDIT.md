# AI-READI Full-v3 Data Audit & Cleaning Plan

> Empirical audit of the converted parquet tree (`data/full/AI_READI/`, 2,280 participants,
> relevant subset = garmin + dexcom + clinical + metadata, ~784 MiB). Produced by `audit_data.py`
> and `audit_data2.py` (reproducible). Full numeric output in `logs/audit_report.txt` and
> `logs/audit_report2.txt`.
>
> **Scope:** discover everything that must be cleaned/handled before training. No pipeline design,
> no actual cleaning here — that's the next step. Findings are empirical unless marked *inferred*.
> Doc facts that this audit contradicts are flagged **[DOC-FIX]**.
>
> **Verification:** independently re-verified by a fresh `critiquer` subagent + `audit_verify.py`
> (run 4970e22e; all 11 load-bearing claims confirmed). Corrections from that review are marked
> **[VERIFIED-FIX]**. Full verify log: `logs/verify_report.txt`.

## How to reproduce

```bash
.venv/bin/python audit_data.py     # pass 1: inventory, sentinels, coverage, leakage, intervals
.venv/bin/python audit_data2.py    # pass 2: stress scale, timestamp outliers, OMOP source_values, site×label
.venv/bin/python audit_verify.py   # pass 3: independent verification of critiquer findings
```

---

# A. Empirical findings

## A.1 Inventory & integrity

- 15 parquet files. Garmin + dexcom use **one row-group per participant** (stream/filter by
  `person_id` is cheap) — confirmed correct.
- Clinical tables are **single row-group** each (small files; fine).
- Total rows across all parquet: 140,054,362. Biggest: respiratory_rate 47.6M, stress 47.6M,
  HR 22.1M, physical_activity 9.6M, CGM 6.2M.

## A.2 Clinical tables are OMOP long-format, not wide (**structural**)

- `observation.parquet` (707,126 rows × 21 cols, 361 unique `observation_source_value`) and
  `measurement.parquet` (242,279 rows × 25 cols, 108 unique `measurement_source_value`) store
  features as **rows keyed by `*_source_value`** (friendly `"code, description"` strings), **not
  as columns**.
- **Implication:** building survey/clinical features requires filter-by-source-value + pivot-to-wide.
  This is a major, under-specified engineering step. The friendly names in FEATURES.md
  (`paidscore`, `mhoccur_hbp`, `cmtrt_*`, …) are the source_value *prefixes* (before the comma),
  not column names.

### A.2.1 `person.parquet` demographics are blank — sex/race/ethnicity unavailable **[VERIFIED-FIX]**

**[VERIFIED-FIX]** `person.parquet` does NOT contain usable demographics. Verified:
- `gender_concept_id` = 0 for all 2,280 rows; `race_concept_id` = 0; `ethnicity_concept_id` = 0.
- `gender_source_value` = `' '` (single space) for all; `race_source_value` = `' '`;
  `ethnicity_source_value` = `' '`.
- `birth_datetime` = `'1970-01-01 00:00:00'` for all (epoch placeholder); `month_of_birth` = 0;
  `day_of_birth` = 0.
- Only `year_of_birth` has real data (54 unique values, 1930–1983).
- No sex/gender/race/ethnicity field exists in `observation` or `measurement` either — the only
  race-adjacent fields are `pxrd*` (PhenX Racial/Ethnic Discrimination *survey responses*, not
  race itself) and `px281501_metadata` (survey method).

**Consequence:** **sex and race/ethnicity features are NOT available in this dataset.** This
originally contradicted FEATURES.md §3/§6 (sex/race in keep-list and hard onboarding) — **[DOC-FIX
applied]** in the post-audit doc sync (FEATURES/Training/T2D now omit sex/race). The
hard-onboarding block is smaller than planned — age, BMI, waist, BP, family history, smoking only.
The gender/race-by-label confound check is impossible.

## A.3 Sentinels — prevalence and corrections

| Modality | Sentinel rule | Prevalence | Notes |
|---|---|---|---|
| heart_rate | `==0` → sensor off | 1,214,595 rows (5.5%), 99.9% of participants | Confirmed. |
| stress | `∈{-1,-2}` → invalid | **26,881,018 rows (56.5%), 100% of participants** | Confirmed. High rate is expected (Garmin stress computed only at rest; -1/-2 = not-at-rest/unknown). |
| respiratory_rate | `∈{-1,-2}` → invalid | **24,202,120 rows (50.9%), 100%** | Confirmed. |
| oxygen_saturation | `==0` → invalid | **0 occurrences** | Sentinel not present in data (already clean, or Vivosmart 5 never reports 0). |

**[DOC-FIX applied] Garmin stress scale is 0–100, NOT 0–17.** Empirical distribution: valid mass
spans 0–100 (35.9% of rows are 18–100; per-participant median max = 99, p90 = 100, max = 101).
Garmin's published scale is 0–25 resting / 26–50 low / 51–75 medium / 76–100 high
(Firstbeat/HRV-derived, rest-only). FEATURES.md / DATA_STRUCTURE.md now use 0–100 with high
thresholds ≥51 (medium+) / ≥76 (high). The prior plan's ≥7 would flag nearly everyone. (The
parallel attempt used ">50", consistent with 0–100.)

## A.4 Timestamps & wear window — **[DOC-FIX] not everyone is ~14 days

- **Median wear window** HR 14.63d, stress 14.51d, RR 14.51d, sleep 10.20 nights, CGM 9.91d —
  matches FEATURES.md §1 medians.
- **BUT ~63–64 participants per modality have >60 days of data** (year-long), not 14. Worst:
  pid 4364 = 397d, pid 4340 = 381d, pid 1652 = 299d, pid 4186 = 273d, … (same participants across
  HR/stress/RR). Earlier docs claimed the year-long aggregate was only different visit dates —
  **false for these ~63 participants**, who genuinely carry ~1 year of wearable data. **[DOC-FIX
  applied]** in FEATURES/T2D/DATA_STRUCTURE. Must handle variable-length windows (truncate to a
  consistent window, or use all and document) — policy still open (B4.3 / B10.3).
- **Year-long-wear split distribution [VERIFIED-FIX]:** the 63 year-long HR participants break
  down as train=45, val=10, test=8; by label 0=21, 1=14, 2=20, 3=8. The truncation policy (B4.3)
  affects splits unevenly — lock before feature engineering.
- **Corrupt timestamp:** pid 4280 respiratory_rate spans 12,215 days (1991-02-14 → 2024-07-26). The
  1991 date is a bad timestamp (epoch/encoding error). Drop or repair that participant's RR.
- **Duplicate rows (full cohort) [VERIFIED-FIX]:** every modality has exact + timestamp
  duplicates. The original audit only sampled 400 HR participants; full-cohort numbers:

  | Modality | Pids with ts-dups | Ts-dup extra rows | Exact duplicate rows |
  |---|---|---|---|
  | heart_rate | 42 | 13,283 | 10,850 |
  | stress | 48 | 64,190 | 42,547 |
  | respiratory_rate | 49 | 64,571 | 38,843 |
  | cgm | 80 | 960 | 40 |

  **pid 1366 is a systematic ingestion pathology** — worst offender across both HR (10,794 dup
  groups) and stress (60,698 dup rows). Dedup is mandatory for all four modalities, not just HR.
  Without dedup, per-participant aggregations (mean stress, mean RR) are biased toward duplicated
  values.
- **Sampling grids differ by modality:** RR median 20,454 rows/participant vs HR 11,101 — RR is
  ~2× denser than HR. Modalities are **not on a shared grid**; alignment/resampling is required
  for the sequence/aux views.
- **[VERIFIED-FIX] Clinical date columns are strings with inconsistent formats — NOT tz-aware
  timestamps.** Garmin + dexcom timestamps are `timestamp[ms, tz=UTC]` (confirmed in arrow schema).
  But every clinical date column is `large_string` with **three different formats**:
  - `participants.study_visit_date`: ISO `2023-07-27`
  - `observation.observation_date` / `observation_datetime`: M/D/YY `12/12/23` and `12/12/23 0:00`
    (datetime always midnight — time component is placeholder, not real)
  - `measurement.measurement_date` / `measurement_datetime`: ISO `2023-12-12`
  Code that assumes one format or auto-parses as tz-aware will silently corrupt clinical dates.
  Parse each table with its explicit format string.

## A.5 Timezone — feasible, and confounded with label (**critical**)

- 3 clinical sites, 0 nulls: **UAB 800, UW 798, UCSD 682**.
- Mapping (clean): UAB → US/Central, UW → US/Pacific, UCSD → US/Pacific.
- **Site is confounded with the label:** UAB holds **137/258 insulin (53%)** and 277/686 oral-med
  (40%); UW+UCSD (Pacific) hold the milder cases. So timezone correlates with severity via site.
  → **UTC→local conversion is mandatory before any circadian feature.** Computing RHR/SRI/cosinor/
  nocturnal windows in UTC would inject a site-correlated (hence label-correlated) shift — a
  confound/leakage-of-artifact, not just a methodology nicety.

## A.6 Coverage & modality availability

| Modality | Participants | % of 2280 |
|---|---|---|
| cgm | 2245 | 98.5% |
| physical_activity | 2118 | 92.9% |
| respiratory_rate | 2113 | 92.7% |
| stress | 2112 | 92.6% |
| sleep | 2109 | 92.5% |
| heart_rate | 2104 | 92.3% |
| physical_activity_calorie | 1994 | 87.5% |
| oxygen_saturation | 1638 | 71.8% |

- **Wearable core (hr ∩ stress ∩ sleep): 2052 (90.0%).**
- **Aux pool (hr ∩ stress ∩ rr ∩ sleep ∩ cgm): 2034 (89.2%)** — raw presence only; effective n
  after sentinel + overlap is lower (A.6.1–A.6.2). Docs now cite effective pools, not this ceiling.
- **All 7 garmin + cgm: 1610 (70.6%)** — SpO₂ (71.8%) is the bottleneck; requiring it shrinks the
  pool by ~20%. Confirms keeping SpO₂/ODI exploratory (Tier 3).
- 176 participants have **no HR at all** → excludable from any wearable model.
- SpO₂: 642 participants have none; 191 readings >100 (max 101); 2,670 nulls.
- Wear-window tails: HR 56 participants <7d, stress 65 <7d, SpO₂ 152 <7d (9.3%), CGM 78 <7d (3.5%).

### A.6.1 Post-sentinel survival — pools are ceilings, not usable counts **[VERIFIED-FIX]**

The coverage numbers above are **raw modality presence**. After sentinel masking (A.3), some
participants have **zero valid readings** despite having rows (all rows are sentinels):

| Modality | Participants with data | Zero valid after masking | % lost |
|---|---|---|---|
| heart_rate | 2104 | **105** | 5.0% |
| stress | 2112 | **116** | 5.5% |
| respiratory_rate | 2113 | **114** | 5.4% |

These "null participants" are treated as having data by the raw-coverage pools but are unusable.
Within the 2034 aux pool: 71 have zero valid stress, 70 zero valid RR, 69 zero valid HR. **The
effective aux pool after sentinel masking is ≤1963, not 2034.** Similarly the 2052 wearable core
loses 71 (stress) + 69 (HR). Pool definitions must be computed **after** sentinel masking, not
before.

### A.6.2 CGM ↔ wearable temporal overlap — the aux task requires co-occurring data **[VERIFIED-FIX]**

The aux/seq2seq task (Path B) requires CGM and wearable data to be **temporally co-occurring** —
a participant with CGM in week 1 and HR in week 3 has no aligned data. Raw modality intersection
does not guarantee this.

- **CGM ∩ HR participants: 2085.** Of these: **51 have ≤0h overlap** (windows don't intersect at
  all); **60 have <24h**; 98 have <72h. Overlap distribution: p5=82h, p50=238h (~10d, CGM-limited),
  p95=238h.
- **Within the 2034 aux pool:** **33 have ≤0h CGM-HR overlap**, **42 have <24h**, 80 have <72h.
  **Effective temporally-aligned aux pool (≥24h overlap): ≤1992** — and that's before sentinel
  masking (A.6.1), which shrinks it further.

The pool number "2034" is a ceiling. The usable aligned aux pool is substantially smaller. Lock
the minimum-overlap threshold (≥24h? ≥72h?) and report the surviving n.

## A.7 Label & split

- **Label (full):** 0=776 (34.0%), 1=560 (24.6%), 2=686 (30.1%), 3=258 (11.3%). Imbalanced.
- **Split:** train 1576 / val 352 / test 352.
- **train insulin (label 3) = 105** — the binding constraint for the 4-class model. val=67, test=86.
- Splits are label-imbalanced: train 600/384/487/105, val 88/88/109/67, test 88/88/90/86. **test
  is balanced; train and val are not.** Insulin is severely scarce in train.
- age 40–94 (mean 60.9), 0 nulls; 0 duplicate person_id.

## A.8 Leakage fields — confirmed present by real `source_value` names

**`measurement` (hard-exclude — all present):** `import_hba1c`, `import_glucose`, `import_insulin`,
`import_c_peptide`, `lbscat_a1c`. (Exactly the FEATURES.md §3 list — verified.)

**`observation` (hard-exclude — all present):** `mhterm_dm1`, `mhterm_dm2`, `mhterm_predm`,
`mh_a1c`, `mh_dm_age`, `cmtrt_insln` (insulin injection), `cmtrt_a1c` (pills for A1C), `cmtrt_glcs`
(other injections), `cmtrt_lfst` (lifestyle control — **yes ⇒ diabetic, leakage**),
`mhoccur_pdr` (diabetic retinopathy — diabetes-defining + retinal), `dmlcmpdat`/`dmlstartts`/
`dmlcmpts` (diabetes survey dates — metadata, drop).

**`condition_occurrence` [VERIFIED-FIX]:** **NO ICD codes exist.** All 30 `condition_source_value`
entries are self-report terms (`mhterm_dm2`, `mhterm_predm`, `mhoccur_pdr`, `mh_a1c`, `mhoccur_hbp`,
…) — the same source_values already in `observation`. `condition_occurrence` is a **duplicate
projection** of observation's `mhoccur_*`/`mhterm_*` block, not a separately ICD-coded clinical
diagnosis table. Zero E10–E13 codes. Leakage exclusion for diabetes terms is already covered by
B6.2 (observation); treat the two tables as a shared source-value namespace (don't double-count,
don't miss the projection). `mhterm_dm1` is absent from condition_occurrence (present in
observation with all values=0).

**Retinal metadata leak — confirmed in both tables** (FEATURES.md §3):
- observation: `via*` (ophthalmic survey via1–6 + viaocmpdat/viastartts/viacmpts), `rt_dat`,
  `rtci_*`/`rtma_*`/`rttr_*` (OCT/OCTA), `mlcscmpdat`, `plcscmpdat`.
- measurement: `via*` (VA letter/logMAR scores, autorefractor sphere/cyl/axis), `mlcs*` (macular
  contrast sensitivity), `plcs*` (peripheral contrast sensitivity), `mssrf*`/`mssrl*` (foot
  monofilament — **diabetic neuropathy screen, borderline**).

## A.9 Survey sentinels (confirmed + quantified)

`observation.value_as_number` contains codes **555** (1,972), **777** (3,210), **888** (16),
**999** (690), **99** (60) — all → NaN. Same codes appear in `value_as_string`. (The parallel
attempt found and masked 555/777/888/999/99 — confirmed.) **Note:** `observation_id` and
`visit_occurrence_id` also contain 555/777/999 coincidentally — these are ID columns, **not**
features; do not mask them (false-positive guard).

## A.10 Out-of-range physiology & clinical outliers

- HR: max 235 (2 rows >220) → clamp/NaN. min 0 (sentinel).
- stress: max 101 (2 rows >100) → clamp.
- SpO₂: 191 rows >100 (max 101) → clamp/NaN. Range otherwise 69–100, peak at 93.
- CGM: bg range 39–401; 1 below 40, 2 above 400 (residual High/Low markers; pipeline maps High→400,
  Low→40) → clamp to [40,400] or NaN. 39,632 bg_null.
- RR: max 35.41 (physio-fine), min −2 (sentinel).
- **[VERIFIED-FIX] Anthropometric sentinels:** `waist_vsorres` min=0.0 and `hip_vsorres` min=0.0
  (1 participant each — impossible 0 cm circumference; WHR for that participant = 0/207).
- **[VERIFIED-FIX] BMI outliers:** max=95.24, **7 values >60**. Extreme obesity is possible but
  BMI=95 is implausible for most — flag for review.
- **[VERIFIED-FIX] Age discrepancy [VERIFIED-FIX]:** 41 participants have age off by >1 year vs
  `year_of_birth`-derived age (max diff 31: pid 1209, age=57, born 1956, visit 2024 → implied 68).
  Most are −2 (birthday timing), but a handful are real data-entry errors. Reconcile before using
  age as a feature.
- **Calorie counter behavior [flagged-by-reviewer]:** `physical_activity_calorie` is non-monotonic
  with resets (sampled pid 1023: 241 negative diffs out of 1,572 intervals). If differenced to get
  energy expenditure rate, resets produce negative values. **Verify on full cohort during cleaning.**

## A.11 Interval validity (sleep / activity)

- **sleep:** 9 `end<start` (bad order), 9 negative-duration, 11 zero-duration, 2 intervals >24h,
  956 `unknown` sleep_stage_state. Valid stages: light 274,460 / awake 110,009 / deep 99,880 /
  rem 81,619.
- **physical_activity:** 0 bad order, 0 negative, **33,520 zero-duration intervals**, 235 intervals
  >24h, **2,486 blank `activity_name`**. Top names: sedentary 7.14M, walking 1.87M, generic 543k,
  running 27.6k.

## A.12 CGM specifics

- `event_type` = 100% `EGV` (clean — no calibration/carb events). `unit` = 100% mg/dL. Good.
- 2245 participants, median 9.91d, 78 (3.5%) <7d.

## A.13 Borderline / decision-needed (flag, don't auto-decide)

- `cmtrt_lfst` (lifestyle control), `cmtrt_glcs` (other injections), `mhoccur_pdr` (diabetic
  retinopathy) → leakage (above).
- `mssrffl`/`msslffl` (foot monofilament) → diabetes-targeted neuropathy exam; physical finding,
  not self-report diagnosis, but diabetes-specific. **Flag for review.**
- **Lab biomarkers** in `measurement` (not diabetes-defining, but not watch-deployable):
  `import_nt_probnp`, `import_troponin_t`, `import_crp_hs`, `import_total_cholesterol`,
  `import_triglycerides`, `import_hdl_cholesterol`. FEATURES.md §3 says "lipids as lifestyle risk
  factors (keep)" — but these are **lab-measured**, not self-report. **Decision needed:** lab
  biomarkers are not deployable watch-side → exclude from watch-only primary, allow as an
  upper-bound arm (like ECG). Settle explicitly.
- **Anthropometrics** (weight/height/BMI/waist/hip/WHR/BP1/BP2/pulse): single-timepoint, ~2,273
  participants, clinic-measured. These are the "hard onboarding" block (FEATURES.md §6) — keep, but
  note they are clinic/onboarding, not watch-derived.
- `via1–3` (self-report vision difficulty) vs `via4–6` + `via*` clinical scores + autorefractor:
  FEATURES.md §3 says blanket-filter `via*`, but `via1–3` are **self-report** (non-leaking; the
  parallel attempt used `via1`). **Reconcile per-field**, not a blanket `via*` drop.
- MoCA/cognitive scores, memory trials, digit span, letter fluency, etc. → cognitive assessments,
  not in plan, not watch-deployable. Note as available-but-excluded.

---

# B. The hard cleaning plan (complete checklist)

Grouped by concern. Severity: 🔴 blocking / must do before any feature engineering;
🟠 correctness, do before modeling; 🟡 hygiene / robustness; 🔵 decision to lock.

## B.1 Schema & feature extraction 🔴
- [ ] **B1.1** Build an OMOP source_value → feature mapping table for `observation` (361 values) and
  `measurement` (108 values). Classify every value into: keep-survey / keep-clinical / hard-exclude
  (leakage) / retinal-leak / metadata-drop / borderline-review. No modeling until this is complete.
- [ ] **B1.2** Pivot `observation` + `measurement` long→wide per participant (one row per
  `person_id`) for the survey/clinical feature block.
- [ ] **B1.3** **[VERIFIED-FIX]** `person.parquet` demographics are blank — **sex and
  race/ethnicity are NOT available** (gender/race/ethnicity_concept_id=0, source_value=' ',
  birth_datetime='1970-01-01'). Only `year_of_birth` is real. Derive `age` from `participants.parquet`
  (already there, 0 nulls). **Do not** attempt sex/race mapping. Apply [DOC-FIX] to FEATURES.md §3/§6:
  strike "sex, race/ethnicity" from the keep/onboarding blocks. The gender/race-by-label confound
  check is impossible — note this as a limitation.

## B.2 Sentinel masking 🔴
- [ ] **B2.1** HR: `heart_rate == 0` → NaN.
- [ ] **B2.2** stress: `stress_level ∈ {-1,-2}` → NaN. **Use 0–100 scale** (not 0–17). Rescale
  high-stress thresholds (≥51 medium+ or ≥76 high; the prior ≥7 is invalid).
- [ ] **B2.3** respiratory_rate: `∈{-1,-2}` → NaN.
- [ ] **B2.4** SpO₂: clamp `>100` → NaN (191 rows); `==0` rule is a no-op here (0 occurrences) but
  keep it for safety.
- [ ] **B2.5** Survey: in `observation.value_as_number` and `value_as_string`, map
  `{555,777,888,999,99}` → NaN. **Do not** mask these codes in `observation_id`/`visit_occurrence_id`
  (ID columns, coincidental values).
- [ ] **B2.6** Verify no other undocumented sentinels (scan all numeric columns for implausible
  repeated codes like 111, 222, 666, 8888).

## B.3 Out-of-range physiology 🟠
- [ ] **B3.1** HR: clamp `>220` → NaN (2 rows).
- [ ] **B3.2** stress: clamp `>100` → NaN (2 rows).
- [ ] **B3.3** SpO₂: (covered in B2.4).
- [ ] **B3.4** CGM: clamp to `[40,400]` or NaN the 1 below-40 / 2 above-400 residuals.
- [ ] **B3.5** Define and apply physiological bounds for every derived feature (e.g. RR 4–60,
  sleep duration 0–16h, activity duration 0–24h).
- [ ] **B3.6** **[VERIFIED-FIX]** Anthropometric sentinels: `waist_vsorres`==0 and `hip_vsorres`==0
  (1 participant each) → NaN. Recompute WHR after masking.
- [ ] **B3.7** **[VERIFIED-FIX]** BMI outliers: 7 values >60 (max 95.24) → review/clamp. Flag the
  participants; confirm whether data-entry error or extreme obesity.
- [ ] **B3.8** **[VERIFIED-FIX]** Age discrepancy: 41 participants off by >1y vs `year_of_birth`.
  Most are −2 (birthday timing); investigate the extreme cases (e.g. pid 1209, diff=−11) as
  data-entry errors. Decide whether to use `participants.age` or `visit_year − year_of_birth`.
- [ ] **B3.9** **[flagged-by-reviewer]** Calorie counter: verify non-monotonic resets on the full
  cohort. If differencing to get energy rate, clamp negative diffs to 0 or detect reset points.
  Verify before deriving calorie-rate features.

## B.4 Timestamps & wear window 🔴
- [ ] **B4.1** **UTC → local conversion** via `clinical_site` → {UAB: US/Central, UW: US/Pacific,
  UCSD: US/Pacific}, applied to every garmin + dexcom timestamp **before** any circadian/nocturnal/
  SRI/RHR/cosinor computation. (Mandatory — site is label-confounded, A.5.)
- [ ] **B4.2** Repair/drop **pid 4280** respiratory_rate (1991 corrupt timestamp).
- [ ] **B4.3** Handle **~63 participants with year-long wear** (>60d). Decide: truncate to a fixed
  window (e.g. first 14d, or best-coverage 14d), or use all and document. Lock the policy.
- [ ] **B4.4** **[VERIFIED-FIX]** Dedup duplicate rows across ALL modalities (not just HR):
  HR (42 pids, 13,283 ts-dup extra / 10,850 exact), stress (48 pids, 64,190 / 42,547),
  respiratory_rate (49 pids, 64,571 / 38,843), CGM (80 pids, 960 / 40). **pid 1366** needs special
  scrutiny (systematic ingestion pathology — worst dup offender across HR+stress).
- [ ] **B4.5** **[VERIFIED-FIX]** Parse clinical date strings with explicit per-table formats (NOT
  auto-parse): `participants.study_visit_date` ISO `%Y-%m-%d`; `observation.observation_date`
  M/D/YY `%m/%d/%y`; `measurement.measurement_date` ISO `%Y-%m-%d`. `observation_datetime` is
  `12/12/23 0:00` (midnight placeholder — time is not real). These are NOT tz-aware timestamps
  (unlike garmin/dexcom which are `timestamp[ms, tz=UTC]`).
- [ ] **B4.6** Resample/align modalities to a common grid (HR 1-min, RR ~2× denser, SpO₂/CGM
  5-min, sleep/activity interval-based). Document the canonical grid for the sequence/aux views.

## B.5 Coverage filtering 🔴
- [ ] **B5.1** Lock coverage thresholds after the A.6 distributions: candidate ≥7d wear + ≥80%
  1-min HR + ≥8d CGM (FEATURES.md §10). Confirm full-cohort survival at those thresholds.
- [ ] **B5.2** **[VERIFIED-FIX]** Define pools **after** sentinel masking + temporal overlap,
  not on raw modality presence. Raw → effective:
  - Wearable core (hr∩stress∩sleep): 2052 → ≤1983 after post-sentinel zero-valid removal (A.6.1).
  - Aux pool (hr∩stress∩rr∩sleep∩cgm): 2034 → ≤1963 after sentinel masking → **≤1921 after
    ≥24h CGM-HR temporal overlap** (A.6.2). Lock the min-overlap threshold.
  Track which participants fall in which pool at each filtering stage.
- [ ] **B5.3** Exclude participants with no HR (176) or zero valid HR post-sentinel (105) from any
  wearable model.
- [ ] **B5.4** Decide SpO₂ handling: keep Tier-3/exploratory given 71.8% coverage (642 missing);
  do not let it gate the primary pool.
- [ ] **B5.5** **[VERIFIED-FIX]** Quantify post-sentinel coverage survival at the proposed
  thresholds (≥7d valid wear + ≥80% 1-min HR + ≥8d CGM). The raw coverage numbers overstate usable
  data — after sentinel masking, 51 HR participants have <7 valid days, 271 sleep participants
  have <7 valid nights. Confirm the surviving n before locking thresholds.

## B.6 Leakage exclusion 🔴
- [ ] **B6.1** `measurement` hard-exclude: `import_hba1c`, `import_a1c`, `lbscat_a1c`,
  `import_glucose`, `import_insulin`, `import_c_peptide`.
- [ ] **B6.2** `observation` hard-exclude: `mhterm_dm1`, `mhterm_dm2`, `mhterm_predm`, `mh_a1c`,
  `mh_dm_age`, `cmtrt_insln`, `cmtrt_a1c`, `cmtrt_glcs`, `cmtrt_lfst`, `mhoccur_pdr`,
  `dmlcmpdat`/`dmlstartts`/`dmlcmpts`.
- [ ] **B6.3** **[VERIFIED-FIX]** `condition_occurrence` has **NO ICD codes** — all 30
  `condition_source_value` entries are self-report terms identical to observation (`mhterm_dm2`,
  `mhterm_predm`, `mhoccur_pdr`, `mh_a1c`, …). Exclude diabetes terms here too (same list as B6.2),
  but do NOT build an ICD/concept_id filter — there are no ICD codes to match. Treat
  condition_occurrence + observation as a shared source-value namespace to avoid double-counting.
- [ ] **B6.4** Retinal leak filter — observation: `via*` survey metadata, `rt_dat`, `rtci_*`,
  `rtma_*`, `rttr_*`, `mlcscmpdat`, `plcscmpdat`. measurement: `via*` scores/autorefractor,
  `mlcs*`, `plcs*`, `mssrf*`/`mssrl*`.
- [ ] **B6.5** **Reconcile `via*` per-field**: keep `via1–3` (self-report vision difficulty,
  non-leaking); drop `via4–6` + clinical `via*` scores + autorefractor. Do not blanket-drop `via*`.
- [ ] **B6.6** Audit the rest of `cmtrt_*` and `mhoccur_*` blocks for any other diabetes-defining
  fields (FEATURES.md §12 open question — close it here).
- [ ] **B6.7** Post-cleaning assertion: no hard-excluded field survives into the feature matrix
  (automated check).

## B.7 Interval modalities (sleep / activity) 🟠
- [ ] **B7.1** sleep: drop/repair 9 `end<start`, 9 negative-duration, 11 zero-duration,
  2 >24h intervals.
- [ ] **B7.2** sleep: handle 956 `unknown` stage rows (drop, or merge into a neutral bucket) before
  SRI / stage-% features.
- [ ] **B7.3** activity: handle 33,520 zero-duration intervals (drop or keep as zero-length events
  depending on feature), 235 >24h intervals, 2,486 blank `activity_name` (drop or impute category).
- [ ] **B7.4** Map `activity_name` → MVPA intensity tiers (sedentary / walking / generic / running
  / blank) — define the mapping explicitly; "generic" and blank need a rule.

## B.8 Label & split 🟠
- [ ] **B8.1** Keep `recommended_split` as the primary split (held-out test is more honest than
  k-fold). **Document** that the parallel attempt used random 5-fold CV, so numbers aren't
  directly comparable.
- [ ] **B8.2** Acknowledge **train insulin n=105** as the binding constraint; lock the imbalance
  strategy (class weights / focal loss / augmentation) **before** feature selection.
- [ ] **B8.3** Decide whether to stratify-augment or re-split given train/val imbalance (test is
  balanced). Do not silently merge splits.
- [ ] **B8.4** Report 4-class, 3-class, binary, and ordinal formulations; AUPRC mandatory.

## B.9 Site/timezone confound 🟠
- [ ] **B9.1** Report site × label in the paper (UAB enriched for insulin). After UTC→local, verify
  circadian features no longer carry a site-correlated shift (sanity check: RHR distribution by
  site after local conversion should be comparable).

## B.10 Decisions to lock 🔵
- [ ] **B10.1** Lab biomarkers (lipids, troponin, NT-proBNP, CRP): watch-only primary excludes
  them; upper-bound arm allows them (mirrors the ECG upper-bound). Confirm.
- [ ] **B10.2** `mssrffl`/`msslffl` (foot monofilament): include or exclude? (diabetes-targeted exam).
- [ ] **B10.3** Year-long-wear ~63 participants: truncate vs use-all.
- [ ] **B10.4** Coverage thresholds (B5.1): lock after distribution review.
- [ ] **B10.5** Canonical time grid & window length for sequence/aux views (B4.5, B4.3).

## B.11 Doc corrections to apply [DOC-FIX] 🟠

> **Status:** FEATURES.md, DATA_STRUCTURE.md, T2D.md, Training.md, COMPUTE.md, and AGENTS.md were
> synced to these fixes (post-audit doc pass). Items below remain as the audit trail — checked =
> applied in docs; cleaning-code items stay open under B1–B10.

- [x] **B11.1** FEATURES.md §1/§4.1: Garmin stress is **0–100** (not 0–17); fix the scale, the
  validation framing, and the "%time high ≥7" threshold. → also DATA_STRUCTURE sentinels table.
- [x] **B11.2** FEATURES.md §1: ~63 participants have **year-long wear**, not 14 days; the
  "year-long aggregate = different participants on different dates" claim is false for them.
- [x] **B11.3** FEATURES.md §3: `via*` is not a blanket retinal-leak — `via1–3` are self-report
  (keep), `via4–6` + clinical scores are retinal (drop). Update the rule.
- [x] **B11.4** FEATURES.md §10: oxygen_saturation `==0` sentinel is a no-op in this data (0
  occurrences); the real SpO₂ cleaning issue is `>100` clamping + 28% missingness.
- [x] **B11.5** FEATURES.md: note that clinical features are OMOP long-format (source_value-keyed
  rows), so feature extraction = filter + pivot, not column selection. → also DATA_STRUCTURE.
- [x] **B11.6** **[VERIFIED-FIX]** FEATURES.md §3/§6: **sex and race/ethnicity are NOT available** —
  `person.parquet` demographics are blank (concept_id=0, source_value=' '). Strike "sex,
  race/ethnicity" from the keep-list and the hard-onboarding block. Note the confound-check
  limitation in the paper. → also Training.md §6 block hierarchy, T2D.md.
- [x] **B11.7** **[VERIFIED-FIX]** FEATURES.md §3: `condition_occurrence` is NOT ICD-coded — it's
  self-report source_values duplicating observation. Update the leakage rule accordingly.
  → also DATA_STRUCTURE.
- [x] **B11.8** **[VERIFIED-FIX]** Clinical date columns are `large_string` with three formats (ISO,
  M/D/YY, M/D/YY 0:00) — NOT tz-aware timestamps. Noted in DATA_STRUCTURE.md (garmin/dexcom only
  are tz-aware UTC).

---

# C. What is already clean (no action needed)

- CGM: 100% EGV, 100% mg/dL, no calibration/carb events (A.12).
- Parquet row-group-per-participant integrity for garmin + dexcom (A.1).
- `participants.parquet`: 0 duplicate person_id, 0 age nulls, label fully mapped (A.7).
- HR/stress/RR val_null = 0 (nulls only come from sentinels, not storage) (A.3) — but see A.6.1:
  after sentinel masking, 105–116 participants per modality have **zero valid** readings.
- clinical_site: 0 nulls, clean 3-value enumeration (A.5).
- `recommended_split` has no temporal leak: train/val/test all span 2023-07 to 2025-05
  (overlapping enrolment periods, not sequential) [VERIFIED-FIX].

---

# D. Verification appendix

This audit was independently verified by a fresh `critiquer` subagent (run 4970e22e) that ran its
own pyarrow/pandas queries against the parquet files, plus `audit_verify.py` (pass 3). All 11
load-bearing claims from the critique were confirmed against the data:

| # | Claim | Verdict |
|---|---|---|
| O1 | person.parquet demographics blank (no sex/race) | CONFIRMED |
| O2 | condition_occurrence has no ICD codes (self-report dup) | CONFIRMED |
| O3 | HR dups = 42 pids / 13,283 extra (not 4/10,825 from sample) | CONFIRMED |
| O4 | stress 42,547 + RR 38,843 exact dups (doc only checked HR) | CONFIRMED |
| O5 | clinical dates are strings, 3 formats (not tz-aware UTC) | CONFIRMED |
| O6 | 116 stress / 114 RR / 105 HR zero valid post-sentinel | CONFIRMED |
| O7 | 51 CGM∩HR ≤0h overlap; aux pool 33 ≤0h, 42 <24h | CONFIRMED |
| M4 | waist=0, hip=0 (1 participant each) | CONFIRMED |
| M5 | 41 participants age off >1y vs year_of_birth | CONFIRMED |
| M6 | 7 BMI >60, max 95.24 | CONFIRMED |
| M7 | year-long-wear split: train=45, val=10, test=8 | CONFIRMED |

The critiquer's recommendation was **revise** (not reject) — the doc's core claims are mostly
accurate but had three factual errors (condition_occurrence ICD, HR dup scope, person.parquet
demographics), three critical omissions (post-sentinel survival, CGM-wearable temporal overlap,
stress/RR duplicates), and pool definitions that were pre-sentinel ceilings. All corrections are
integrated above, marked **[VERIFIED-FIX]**. Verify log: `logs/verify_report.txt`.
