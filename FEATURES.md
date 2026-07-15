# T2D from Wearables — Canonical Feature & Plan Reference

> **Canonical feature/engineering reference** (merges the former `featurelist.md`
> literature source and `review.md` reconciliation pass, both deleted).
> Methodology authority: `Training.md`. Empirical cleaning facts: `DATA_AUDIT.md`.
> Layout: `DATA_STRUCTURE.md`. Pipeline: `convert_pipeline.py`. Project notes: `T2D.md`.
> Dataset: AI-READI v3.0.0, n=2,280. Wearable: Garmin Vivosmart 5 (PPG HR, accel, SpO₂, sleep staging).
>
> Data facts verified against the converted full parquet (`data/full/AI_READI/`, audit
> passes in `logs/`) unless marked *inferred*. Doc corrections from the full-cohort
> audit are applied here ([DOC-FIX] / [VERIFIED-FIX] in `DATA_AUDIT.md` §B11).

## 0. Decisions locked

- **Architecture bet (aligned with `Training.md`):**
  1. **Path A direct baseline first** (mandatory floor) — LightGBM + CatBoost on GREEN
     summary features; fixed `recommended_split` (+ person-bootstrap CIs for block Δ);
     multiclass/binary (+ ordinal metrics); calibration diagnostic. **Status (2026-07-14):** Path A
     tabular **frozen** — watch floor 0.666; deployable C1 (watch+onboarding+mood) 0.738 / 0.831.
     See `training/path_a_blocks/REPORT_A_WRAP.md`.
  2. **Path B headline cell = B4** — seq2seq traj + rep-distill under LUPI — **concluded null**
     for deployable raise vs C1 (2026-07-15; easy + hard teachers). See `REPORT_B4*.md`.
     Paper may still use rigorous-direct + negative-result framing for this cell.
  3. **B1 multi-task (scalar CGM summaries)** = controlled ablation only — **frozen 2026-07-15**.
     After sleep FE unit fix + input z-score: pure-seq test 4-AUC **0.652**; λ=0.5 multi-task
     **null** (paired boot CI lo≯0); GREEN late-fuse **no raise** (0.638). Pre-fix ~0.51 was
     broken inputs. Authority: `training/path_b/REPORT_B1.md`. Do not reopen B1 λ grids.
  4. **B2 two-stage** = ablation **frozen 2026-07-15** — no deployable arm beats C1; predicted handoff null; oracle +0.09 4-AUC non-deployable (`REPORT_B2.md`).
  5. **B3 logit-KD** = strong baseline to beat / reproduce **last** (Diasense method), not a contribution.
  6. **SSL / end-to-end 1-min** = gated later lever (makes raw sequence models learnable at
     this n); not cold-start CNN. **MOMENT** = one-afternoon side bet.
- **Scientific claim:** **watch-only is the paper headline.** Onboarding/self-report is the
  "deployable config" in a deployment section, not the claim.
- **Direct is built first regardless** — floor, reference for every aux Δ, bare-minimum model.
- **Execution ladder (Path B):** **B1 (frozen) → B2 (frozen) → B4 (concluded) → B3 last.**
  B2 predicted person-CGM handoff null + oracle headroom (`REPORT_B2.md`). B4 traj/rep-distill
  null for deployable C1 raise (`REPORT_B4*.md`). SSL/MOMENT only if watch-only leaves dynamics
  on the table.

## 1. Ground-truth data facts (verified on full cohort)

- **Per-participant wear window is mostly short, with a long tail:** median **14.6 days** HR
  (IQR 13–19), **9.9 days** CGM, **10 nights** sleep. **~63 participants per modality have
  >60 days of true year-long wear** (worst: pid 4364 = 397d) — not an aggregate artifact of
  different visit dates. Both direct and aux must handle variable-length windows; truncation
  policy (first 14d / best-coverage 14d / use-all) is a lock-before-FE decision
  (`DATA_AUDIT.md` B4.3 / B10.3). Year-long wearers split train=45 / val=10 / test=8.
- **HR is 1-min instantaneous** (median gap 60s, 25/75 pct 60/60s). **No beat-to-beat RR
  intervals exist in any Garmin file.** → clinical time-domain HRV (RMSSD/SDNN/pNN50) and
  frequency-domain (LF/HF) are **not computable** from `heart_rate.parquet`. At 60-s sampling,
  Nyquist limits frequency content to ≤0.008 Hz, below LF/HF bands.
- **Garmin `stress` is 0–100 (not 0–17)** — Firstbeat computes it from RMSSD during detected
  inactivity. Empirical valid mass spans 0–100 (per-participant median max ≈ 99). Garmin bands:
  0–25 resting / 26–50 low / 51–75 medium / 76–100 high. Mask `{-1,-2}` (56.5% of rows — expected,
  rest-only metric). 2025 validation work shows significant correlation with HRV/RMSSD/SDNN and
  ANS activity (biorxiv 2025; J Affective Disord 2025, DOI 10.1016/j.jadr.2025.100974). Validated
  as an ANS indicator; **not** validated against T2D outcomes; not interchangeable with Holter
  absolutes. Defensible as a **primary** T2D autonomic feature, with stated caveats.
  **High-stress threshold:** ≥51 (medium+) or ≥76 (high) — never the obsolete ≥7.
- **CGM is clean:** 5-min EGV only (`event_type=='EGV'`, ~2854 readings/participant). No
  calibration/carb events. Unit 100% mg/dL.
- **Coverage is uneven:** some participants have 872 HR readings / 0.9d → coverage filtering
  (Diasense 2,280→1,586) is necessary. Raw modality presence overstates usable n — see pools below.
- **Pools (ceilings → effective; lock thresholds before FE):**

  | Pool | Raw presence | After sentinel zero-valid | After ≥24h CGM↔HR overlap |
  |---|---|---|---|
  | Direct T2D (all labeled) | 2,280 | — | — |
  | Wearable core (HR∩stress∩sleep) | 2,052 | ≤1,983 | — |
  | Aux (HR∩stress∩RR∩sleep∩CGM) | 2,034 | ≤1,963 | **≤1,921** |
  | CGM-haves | 2,245 | — | CGM∩HR: 2,085 raw; 51 have ≤0h overlap |

  Post-sentinel: 105 HR / 116 stress / 114 RR participants have **zero valid** readings despite
  having rows. SpO₂ covers only 71.8% (1,638) — do not gate the primary pool on it.
  Min CGM↔wearable overlap for aux (≥24h vs ≥72h) is still a lock decision (`DATA_AUDIT.md` B5.2).
- **Anthropometrics are single-timepoint:** max 1 row/participant for weight, BMI, waist, height
  → **weight/BMI trajectory features are infeasible** (only baseline). Watch for waist/hip==0
  sentinels (1 pid each) and BMI outliers (7 values >60, max 95.24).
- **Sex and race/ethnicity are NOT available.** `person.parquet` demographics are blank
  (`gender/race/ethnicity_concept_id=0`, source values `' '`, birth_datetime epoch placeholder).
  Only `year_of_birth` is real; use `participants.age` (0 nulls). No sex/race fields in
  observation/measurement either (`pxrd*` is discrimination survey, not race). Gender/race-by-label
  confound checks are impossible — note as a limitation.
- **Clinical tables are OMOP long-format**, not wide. `observation` (361 `observation_source_value`s)
  and `measurement` (108 `measurement_source_value`s) store features as **rows keyed by source_value**,
  not columns. Feature extraction = filter-by-source-value + pivot-to-wide. Friendly names in this
  doc (`paidscore`, `mhoccur_*`, `cmtrt_*`, …) are source_value *prefixes* (before the comma).
- **PAID is NOT leakage:** `paidscore` is populated across all labels including healthy;
  non-monotonic across labels (oral-med > insulin in mini). A leakage proxy would cleanly separate;
  this doesn't. Usable self-report distress feature (same status as CES-D). *Oddity:* healthy
  participants have non-zero diabetes-distress scores — AI-READI likely administered universally;
  one-line instrument check warranted, but for ML it's non-separating.
- **SRI is feasible:** sleep covers median 10 nights/participant (≥ UK Biobank's 7); interval data
  tiles each night with awake-gaps, daytime between nights = awake → minute-level sleep/wake binary
  is constructible → Phillips SRI computable.
- **Dataset is small:** full = **6.36 GiB**; environment = 5.0 GiB (79%, dropped); garmin = 741 MiB;
  dexcom = 32 MiB; clinical = 4.8 MiB; ECG = 568 MiB (dropped from deployable, kept for upper-bound
  arm). Relevant pull ≈ **780 MiB** (present locally under `data/full/AI_READI/`).
- **Retinal metadata leaked into clinical tables** despite dropping retinal images — must filter
  (but `via1–3` self-report vision difficulty is keepable; see §3).
- **Comorbidity / survey sentinels:** `mhoccur_*` are `0`/`1` with **`777` = not-sure/refused**
  (and 555/888/999/99 also present) → map to missing, never as a number. Do **not** mask these
  codes in ID columns (`observation_id`, `visit_occurrence_id`) — coincidental values.
- **Site × label confound:** 3 sites (UAB 800, UW 798, UCSD 682). UAB holds 53% of insulin cases.
  Timezone map: UAB→US/Central, UW/UCSD→US/Pacific. **UTC→local is mandatory** before any
  circadian/nocturnal/SRI/RHR/cosinor feature — otherwise site-correlated shift leaks into labels.
- **Label (full):** 0=776 (34.0%), 1=560 (24.6%), 2=686 (30.1%), 3=258 (11.3%).
  Full-split train/val/test = 1576/352/352 (train insulin 105). **Post-clean wearable_core**
  train/val/test = **1277/270/277** with **train insulin n=80** (binding Path A constraint).
- **Duplicates & bad timestamps:** exact + timestamp dups across HR/stress/RR/CGM (pid 1366 is a
  systematic ingestion pathology). pid 4280 RR has a 1991 corrupt timestamp. Dedup + repair before FE.
- **Modalities are not on a shared grid:** RR ~2× denser than HR; SpO₂/CGM 5-min; sleep/activity
  interval-based. Sequence/aux views need an explicit resample/align policy.

## 2. Philosophy & framing

- **Watch-only is the floor and the paper claim.** The model must work on watch-only; everything
  else is additive and optional. Onboarding/self-report is the deployable config, reported in a
  deployment section.
- **Survey-dominates-wearable is the central risk.** Diasense found survey-only ≈ wearable LSTM
  (and survey features drove most of the 0.6846→0.7937 2-AUC jump). If comorbidities do the
  metabolic-phenotype job cheaper than the watch, the "from wearables" claim is strained regardless
  of AUC. **SHAP guardrail (§6):** after the final model is fit, if the top-10 SHAP features are
  all survey items, the claim is dead — revisit watch feature engineering or framing. Surface this
  in the paper, don't hide it.
- **Missing ≠ 0.** Absent or 555/777/888/999/99 self-report → `NaN`, never "no." Tree models handle
  NaN natively; neural models need masks/indicators.
- **Post-diagnosis caveat:** AI-READI wearables were recorded *after* participants knew their
  status → T2D-positive behavior may reflect post-diagnosis lifestyle change, not the disease
  signature. Limits external validity to true screening; reframe claim as **"risk stratification,"
  not "early diagnosis."** Belongs in the discussion.
- **Honest performance target** (from `Training.md` / §5): wearable-only ceilings binary
  0.75–0.86, 4-class **0.72–0.75**. For this mixed-control cohort on a held-out split expect binary
  0.78–0.82, 4-class 0.72–0.75. Do **not** anchor on ~0.92 UKB-style lab-biomarker figures.

## 3. Label & leakage rules

4-class label (`metadata/participants.parquet` → `label`): `0` healthy, `1` pre-diabetes
(lifestyle), `2` oral/non-insulin injectable, `3` insulin-dependent. Imbalanced → AUPRC required;
report binary (healthy-vs-not), 4-class, and ordinal (prefer **CORN** over classical
proportional-odds — see `Training.md`). Split by `person_id`, never by row; use
`recommended_split` (held-out test is stricter than k-fold averages; Diasense used random 5-fold —
numbers not directly comparable).

### Hard exclude (leakage — verified present by source_value)

| Source | Fields | Why |
|---|---|---|
| `measurement` | `import_hba1c`, `import_a1c`, `lbscat_a1c`, `import_glucose` | Diagnostic criteria for the label. |
| `measurement` | `import_insulin`, `import_c_peptide` | Define treatment group / insulin resistance. |
| `observation` | `mhterm_dm1`, `mhterm_dm2`, `mhterm_predm`, `mh_a1c`, `mh_dm_age` | Diabetes terms / age-at-diagnosis = the label or leakage by construction. |
| `observation` (meds) | `cmtrt_insln` (inject insulin), `cmtrt_a1c` (A1C pills), `cmtrt_glcs` (other injections), `cmtrt_lfst` (lifestyle control — yes ⇒ diabetic), other `cmtrt_*` T2D meds | Anyone on these is class 2/3 by definition. Audit remaining `cmtrt_*` fully. |
| `observation` | `mhoccur_pdr` (diabetic retinopathy), `dmlcmpdat`/`dmlstartts`/`dmlcmpts` | Diabetes-defining / diabetes-survey metadata. |
| `condition_occurrence` | Same self-report diabetes terms as observation (`mhterm_dm2`, `mhterm_predm`, `mhoccur_pdr`, `mh_a1c`, …) | **Not ICD-coded.** All 30 `condition_source_value`s are self-report terms duplicating observation — a shared source-value namespace. Exclude diabetes terms; do **not** build an ICD/concept_id filter (zero E10–E13 codes). Avoid double-counting with observation. |

### Keep without worry (risk factors, not criteria)

**Age** (from `participants`; reconcile the 41 pids off >1y vs `year_of_birth` before FE);
BMI/height/waist/hip/WHR (single timepoint, clinic-measured); family history T2D (`fh_dm2pt/sb`);
BP/smoking as lifestyle risk factors; CES-D depression; **PAID distress (verified non-leakage, §1)**;
non-metabolic comorbidities (HTN, hyperlipidemia, depression, etc.); `via1–3` self-report vision
difficulty (non-leaking).

**Not available:** sex, race/ethnicity (see §1).

**Borderline / decide explicitly (`DATA_AUDIT.md` B10 / A.13):**
- Lab biomarkers (lipids, troponin, NT-proBNP, CRP): **exclude from watch-only primary**; allow as
  upper-bound arm (mirrors ECG). These are lab-measured, not self-report lifestyle.
- `mssrffl`/`msslffl` (foot monofilament): diabetes-targeted neuropathy exam — flag for review.
- Cognitive batteries (MoCA, digit span, …): available but excluded (not watch-deployable, not in plan).

### Operational leakage rules

- **Retinal metadata leaked into clinical** — filter observation: `via*` *survey metadata*
  (`viaocmpdat`/…), `rt_dat`, `rtci_*`/`rtma_*`/`rttr_*`, `mlcscmpdat`, `plcscmpdat`;
  measurement: clinical `via*` scores/autorefractor, `mlcs*`, `plcs*`, `mssrf*`/`mssrl*`.
  **Per-field `via*` rule (not blanket):** keep `via1–3` (self-report vision difficulty);
  drop `via4–6` + clinical `via*` scores + autorefractor.
- **Survey sentinels → missing; missing ≠ 0** — map `{555,777,888,999,99}` in
  `value_as_number` / `value_as_string` only (not ID columns).
- **Auxiliary-data scope:** CGM (`dexcom`) is invasive, not a deployable feature. Training-time
  supervision for the aux emulator only (LUPI / Path B). CGM-derived metrics (TIR/TBR/TAR/CV/MAGE)
  are paper-only upper bounds when reported as features; at deployment they are emulated.
- **Audit rule:** before modeling, classify every `observation`/`measurement` source_value; if its
  value could plausibly have assigned `study_group` / `label`, it goes in hard-exclude. Automate a
  post-cleaning assertion that no hard-excluded field survives into the feature matrix.

## 4. Feature inventory (with literature backing)

### 4.1 Tier 1 — strong evidence, must-have (GREEN core)

**Resting heart rate (RHR).** Source: lowest 30-min moving percentile during sleep (03:00–07:00
*local*). Aune 2015 meta (n=119,915, 5,628 cases): RR **1.20 per 10 bpm** (1.07–1.34). Copenhagen
Male Study: 16% per 10 bpm, RHR>90 vs ≤50 HR 3.06. Mechanism: sympathetic overactivity → ↓insulin
secretion, ↓muscle glucose uptake via vasoconstriction, RAAS activation. Caveat: beta-blocker
confound (med list is leakage-excluded → unmodeled).

**Minute-scale HR features (the wearable-derived HRV substitute).** Mean, SD, CV, HR range,
nocturnal HR dip — these are *not* the clinical HRV the literature validates, but are what 1-min
PPG HR supports. Pair with Garmin stress (below) for the autonomic story. *Honest framing:*
"consumer-watch HR variability," not "HRV in the clinical sense."

**Garmin stress score (autonomic proxy).** Source: `garmin/stress.parquet` (**0–100**; mask
`{-1,-2}`). Features: mean, SD, %time medium+ (**≥51**), %time high (**≥76**), nocturnal stress,
variability, recovery. Literature-validated ANS/HRV indicator (§1). The closest substitute for the
infeasible clinical RMSSD/SDNN. **ECG-derived HRV (SDNN/RMSSD from the 12-lead snapshot) goes in
the paper upper-bound arm only** — enables a clean ablation: does clinical HRV beat the stress
proxy, and by how much?

**Sleep Regularity Index (SRI).** Source: `garmin/sleep.parquet` → minute-level sleep/wake binary
→ Phillips 2017 formula (feasible, §1). Chaput 2024, *Diabetes Care* (n=73,630 UK Biobank, 7-day,
8-yr follow-up): irregular (<71.6) vs regular (>87.3) HR **1.38** (1.20–1.59); dose-response
elevated below SRI≈80. **Independent of duration** — irregular sleepers getting ≥7h still had
elevated risk. DOI 10.2337/dc24-1208. SRI def: Phillips 2017, *Sci Rep*, DOI
10.1038/s41598-017-03171-4. Backup: SD of sleep-onset timing (UK Biobank 2026, n=72,562: HR 1.30,
1.07–1.58; SD of duration not significant in joint models).

**Sleep duration (U-shaped).** Source: total sleep time/night → participant mean. Shan 2015 meta
(n=482,502, 18,443 cases): vs 7h, RR **1.09/h shorter** (<7h), **1.14/h longer** (>7h); nadir
7–8h. Cappuccio 2010: short RR 1.28, long RR 1.48. Liu/Zhu 2025 (53 studies, 1.48M): <7h OR 1.18,
≥8h OR 1.13. **Code as |deviation from 7.5h| or short/long binaries — a linear term loses the U.**

**MVPA / activity dose.** Source: `garmin/physical_activity.parquet` → minutes/day
moderate-to-vigorous (map `activity_name` to intensity tiers). Boonpor 2023, *BMC Med* (n=40,431):
≥600 vs <150 min/wk → **71% lower T2D**; ≥75 vs <25 vigorous → 64% lower. Strain 2023, *Diabetes
Care* (n=90,096): 19% lower odds per 5 kJ/kg/day PAEE; 43–61% lower for 1–1.5 h/day MVPA. Caveat:
Garmin `activity_name`→MVPA is coarser than UK Biobank raw accel → attenuated effects. Explicit
mapping needed for sedentary / walking / generic / running / blank (2,486 blank names; 33,520
zero-duration intervals — see cleaning).

**Sedentary time.** Source: minutes/day with no activity bouts. Zhou 2025, *BMC Public Health*
(n=69,461): high vs low HR **1.37**. Independent of MVPA (include both).

**Circadian rest-activity rhythm (RAR).** Source: 1-min activity counts from HR+activity → cosinor
(amplitude/mesor/acrophase) + non-parametric (M10, L5, relative amplitude). Wu 2025, *Nutr
Diabetes* (n=74,165): low amplitude HR **1.48**, low mesor HR **1.55**, delayed acrophase HR
**1.25**. Wu 2023, *DOM* (n=97,503): lowest relative-amplitude quartile HR **2.06**. **Requires
local time per participant, not UTC** — map `clinical_site`→US time zone first (mandatory under
site×label confound, §1).

### 4.2 Tier 2 — include but don't lead with

Sleep efficiency (cross-sectional HbA1c r=-0.35; Tasali 2024 next-day glycaemia); WASO (objective,
well-defined; large incident-T2D evidence sparse); acrophase (if fitting cosinor anyway); Garmin
calories (active/resting ratio — semi-redundant with steps+HR; watch non-monotonic counter resets
before differencing); respiratory_rate (weak, watch-derived; 50.9% sentinel rate).

### 4.3 Tier 3 — exploratory, with caveats

**SpO₂ / ODI.** Nguyen 2025, *Nat Sci Sleep* (MAILES, n=536, 52 cases, 8yr): ≥19 non-supine ≥3%
desaturations → OR **2.41** (1.20–4.82). Mechanism: intermittent hypoxia → β-cell dysfunction via
oxidative stress + visceral adipose inflammation. **BUT: no peer-reviewed overnight validation for
Vivosmart 5;** other Garmin models often fail clinical accuracy (Jiang 2023, PLOS Digit Health;
Graversen 2023). Coverage only 71.8% (642 missing) — **keep exploratory; do not gate primary pool.**
If used, run a finger-oximeter validation subsample (optional side contribution). Cleaning: clamp
`>100` → NaN (191 rows); `==0` sentinel is a **no-op** in this data (0 occurrences) but keep for safety.

**%REM, %deep sleep.** HypnoLaus (n=2,164) shows univariate metabolic-syndrome association that
**vanishes after multivariable adjustment + OSA exclusion**; wearable stage accuracy 40–70% vs PSG.
**Don't use in primary model** — lean on duration + regularity + efficiency.

### 4.4 Drop entirely

Clinical HRV RMSSD/SDNN/pNN50 from HR (infeasible); LF/HF ratio (null: Benichou 2018 SMD 0.02,
p=0.914); HRR from passive data (needs maximal exercise tests — infeasible); step cadence &
sedentary fragmentation (no prospective incident-T2D evidence); weight/BMI trajectory (single
timepoint — infeasible); ECG (clinic, deployable-out; upper-bound arm only); environment sensor
(no realistic deployment story); retinal metadata (except `via1–3` self-report); driving; sex/race
(unavailable in dataset).

## 5. State of the art & Diasense (teammate's prior work — to beat/distinct-from)

- **Diasense** (GitHub-only, unpublished; re-derive before citing): AI-READI v3, n=1,586 after
  filtering. 41 wearable + 27 survey features; LSTM+Attention + Optuna LightGBM, **logit-KD** from
  CGM teacher (T=2, α=0.3 best). 4-AUC **0.7412**, 3-AUC 0.7490, 2-AUC **0.7937**. RF baseline
  4-AUC 0.6534. Wearable+KD-only 2-AUC **0.6846** — the jump to 0.7937 is overwhelmingly the survey
  block, not the glucose signal. Key findings: survey-only ≈ wearable LSTM; HRV / circadian-HR
  amplitude / sleep efficiency strongest wearable predictors; KD transfers glucose-correlated signal.
  Multi-task on scalar CGM summaries was tried and abandoned (+0.003 4-AUC, confounded backbone).
- **UK Biobank (Lam 2021, JMIR Diabetes, n=103,712):** binary T2D AUC **0.86** (clean controls;
  drops to 0.75–0.77 with impaired controls). DOI 10.2196/23364.
- **WEAR-ME (Nature 2026, s41586-026-10179-2):** Fitbit/Pixel, n=1,165; insulin resistance **0.70**
  wearable+demo, **0.87** with full biomarkers.
- **SweetDeep (Henriques 2025, arXiv/IEEE JBHI preprint):** Samsung Galaxy Watch 7, n=285, 82.5% acc.
- **Wearable-only ceilings:** binary 0.75–0.86, 4-class 0.72–0.75. For our 2,280 mixed-control
  cohort expect binary 0.78–0.82, 4-class 0.72–0.75. Do not anchor on ~0.92 lab-biomarker figures.

## 6. Methodology

Canonical training methodology lives in **`Training.md`**. Feature-facing rules only here:

- **Decision bar (pre-registered):** a feature block stays deployable iff ΔAUC **>+1.0**
  (i.e. **>+0.01** absolute) with CIs that do not freely overlap 0 **and** stable permutation
  importance. Implementation uses **fixed `recommended_split` + person-bootstrap CIs** (not
  reshuffled nested k-fold). Below that, dropped regardless of SHAP.
- **Block-ablation hierarchy** (package `training/path_a_blocks/`):
  1. **watch-only** (GREEN) — **done** (test 4-AUC 0.666)
  2. **+hard onboarding** — age, BMI/waist/height, family history, BP (**no sex/race**;
     smoking was FE-gap: raw codes `susmk*` not pivoted into onboarding) — **done / decision_bar_pass** (0.699)
  3. **+comorbidities** — HTN-first core — **done / decision_bar_pass=False** (0.709); not in stack
  4. **+mood** — CES-D-10 + PAID scores — **done / decision_bar_pass** (0.738 / binary 0.831);
     PAID carries the block; CES near-null alone
  5. **+diet** (if present and non-leaking) — **not run**; optional only, not required for freeze
  6. **Post-freeze C1 sensitivities** — smoking (`susmk*` ever+current), `mhoccur_obs`, `via1–3`,
     joint: **all bar-fail vs C1** (max joint ΔAUC +0.009). C1 not expanded. See
     `REPORT_A_WRAP.md` §8 / `PLAN_SENS_C1.md`.

  Report ΔAUC + ΔAUPRC per block with bootstrap CIs. **Path A tabular frozen** → secondary
  deployable = watch+onboarding+mood (C1). Wrap + sensitivities: `REPORT_A_WRAP.md`.
- **SHAP guardrail:** after the final fit, if top-10 SHAP are all survey items → "from wearables"
  claim dead; revisit watch engineering or framing. Report SHAP alongside permutation importance.
- **Evaluation variants (all required):** 4-class macro-OVR AUC + AUPRC; binary healthy-vs-not;
  ordinal (**CORN** preferred over classical proportional-odds / CORAL); per-class one-vs-rest AUC.
  **Calibration is required** (isotonic/Platt/Beta on val; report curves + Brier with every AUC) —
  risk stratification needs calibrated probabilities, not rank-only AUC.
- **Path A models:** LightGBM (baseline, first) + CatBoost (categoricals + ordered boosting at
  small n) as primary; LR/linear SVM as interpretability floor; RF as secondary robustness check
  (not because of an unverified "RF beats XGB" citation — local prior says RF was weakest).
- **Pools:** direct T2D = all labeled after wearable coverage filter; aux = post-sentinel +
  temporal-overlap CGM∩wearable subset (§1 table). **Do not silently restrict the T2D pool to
  CGM-haves.** Track n at each filter stage. **Train insulin n=80** on wearable_core (post-clean)
  is the binding constraint (was 105 before coverage filter).
- **Imbalance strategy** (class weights / focal loss / TS aug for sequence models) locked
  **before** feature selection so importance isn't confounded by the loss.
- **Weighting features:** don't hand-weight. GBMs learn natively (group-lasso/L1 optional to
  shrink soft blocks).

## 7. Path forward

Aligned with `Training.md` §4–§7. Feature/engineering view:

1. **Cleaning + GREEN feature matrix** — **done** (`data/processed/`; `watch_green` n=1824 +
   onboarding/comorbidity/mood blocks). Re-clean only if policy changes.
2. **Path A direct = floor + headline baseline** — **frozen (2026-07-14)**. Watch-only 0.666 is
   the aux / paper reference; deployable secondary is **C1 0.738 / 0.831** (not 1A alone).
   Ladder + wrap: `training/path_a_blocks/REPORT.md`, `REPORT_A_WRAP.md`. CORN optional; cal
   diagnostic.
3. **Controlled B1 ablation** — **done / frozen** (multi-task null; pure-seq ~0.652).
4. **B2 two-stage ablation** — **done / frozen** (predicted null; oracle +0.09 non-deployable).
5. **B4 traj + rep-distill** — **done / concluded null** for deployable C1 raise (A + B easy +
   hard teachers). Grid FE + `training/path_b/b4/`. See `REPORT_B4*.md`.
6. **B3 logit-KD baseline** — **next / last** (Diasense-style).
7. **SSL-pretrained backbone** (gated) — lever for raw 1-min end-to-end if hand features leave
   dynamics on the table; not a cold-start CNN (hourly already failed locally at 4-AUC 0.6454).
   MOMENT embeddings → LGBM as a one-afternoon high-variance side bet.
8. **Generative / TS augmentation + calibration polish** on the winning sequence/aux backbone if
   class-3 data hunger or miscalibration shows up.
9. **SpO₂ validation + ODI** (optional side contribution). Methodological, not predictive headliner.

**Novelty note:** plain logit-KD (B3) is Diasense's territory — not a contribution as-is. Multi-task
on scalar CGM (B1), two-stage point estimates (B2), and B4 traj/rep-distill are **controlled
nulls** for deployable raise under recipes run. Residual delta may be SSL+aux, attention fusion,
or the deployment angle. The deployment motivation (no CGM at inference) is real and publishable.

## 8. Build order

1. **Cleaning on GREEN + clinical blocks** (filter retinal rows per §3; mask sentinels + survey
   codes; dedup; coverage-filter participants; leakage exclusion; **UTC→local**; lock year-long
   truncation + min CGM overlap + coverage thresholds). Full checklist: `DATA_AUDIT.md` §B.
2. **Feature engineering → two views:**
   - **(a)** participant-level summary matrix (Path A direct T2D) — GREEN watch, then onboarding /
     comorbidity / mood / diet blocks as separate matrices for block ablation. **Done / frozen.**
   - **(b-daily, B1)** person × day wearable vector + daily CGM 8-summary targets — **done**
     (`features/watch_daily.parquet`, `features/cgm_daily.parquet`; site-local day from UTC+zone).
     Sleep duration uses `.dt.total_seconds()`; `sleep_n_bouts` = sessions/onset day (fixed 2026-07-15).
     See `training/path_b/PLAN_B1_DATA.md` / `PLAN_B1_FIX.md`.
   - **(b-grid, B4)** 5-min multi-modal grid — **built 2026-07-15** (`features/grid_5min.parquet`,
     6.88M × 1824; `grid_5min_person` quality). UTC bins + site-local ToD; concurrent aux median
     ~210 h. Train-time subwindow = wear-density only (CGM-free). Package: `training/path_b/b4/`.
3. **Path A baseline** — **frozen** (watch floor + 1A/1B/1C + wrap). Optional leftovers only:
   diet block, GREEN v2 FE, CORN, cal rewrite.
4. **Survey EXTRAS** — hierarchy executed: 1A pass, 1B fail, 1C pass; diet skipped. Do not reopen
   1B as claim without a new pre-registered protocol.
5. **Path B ladder:** **B1 frozen** → **B2 frozen** → **B4 concluded** → **B3 next**. B2: T1 0.735 ≯ C1;
   O1 0.823 ceiling. B1 pure-seq ~0.652. B4: no deployable arm beats C1 (`REPORT_B4*.md`).

## 9. Compute & storage

- **Storage:** relevant pull ≈ 780 MiB (garmin+dexcom+clinical+metadata; ECG & environment dropped).
  Present at `data/full/AI_READI/`. ECG 568 MiB pulled only if the upper-bound HRV ablation is run.
- **Cleaning (CPU, parquet):** local 5600G/16 GB handles via row-group streaming; Lightning 16–32
  core interruptible CPU VM cuts wall-time. No GPU.
- **Training:** Lightning L4 1× ($1.22/hr) dev → A100-80 ($3.32/hr) or L40S for full. Modal for
  short hyperparam sweeps. Details: `COMPUTE.md`.
- **Colab/Kaggle:** mini prototyping + the small ~2-week-window subset.
- Copy canonical to VM local disk once; never train over a Drive mount.

## 10. Cleaning checklist (per modality)

Full severity-tagged checklist with verification status: **`DATA_AUDIT.md` §B**. Feature-facing
summary:

| Block | Cleaning |
|---|---|
| All Garmin | Sentinels→NaN: HR==0, stress∈{-1,-2}, RR∈{-1,-2}; SpO₂ clamp `>100`→NaN (`==0` is no-op here). Dedup exact + timestamp dups (all modalities; scrutinize pid 1366). Coverage-filter **after** sentinel masking (candidate ≥7d valid wear + ≥80% 1-min HR + ≥8d CGM; confirm surviving n — raw coverage overstates). |
| Heart rate | 1-min grid. Minute-scale HR stats on clean windows; reject >20% missing; clamp `>220`→NaN. Restrict RHR to 03:00–07:00 **local**. Exclude 176 no-HR + 105 zero-valid-post-sentinel pids from wearable models. |
| Stress | Mask {-1,-2}; scale **0–100**; %time ≥51 / ≥76, nocturnal, variability. Clamp `>100`→NaN. |
| Sleep | Drop/repair bad intervals (end<start, neg/zero/>24h); handle 956 `unknown` stages. Expand intervals→minute binary for SRI; SD of onset as backup. Stage labels ordinal, not gold (40–70% vs PSG). |
| Activity | Map `activity_name`→intensity (explicit rule for generic/blank); handle 33,520 zero-duration + 235 >24h intervals. |
| Calories | Verify non-monotonic counter resets before differencing; clamp negative diffs. |
| Timestamps | **UTC→local via clinical_site→{UAB: US/Central, UW/UCSD: US/Pacific}** before any circadian feature. Repair pid 4280 RR. Lock year-long (~63 pid) truncation policy. Clinical dates are **strings** (3 formats) — not tz-aware; parse per table. |
| Clinical | OMOP long→wide pivot via source_value map. Filter retinal per §3 (`via1–3` keep). Survey sentinels→NaN. Leakage-exclude (§3). Anthropometric 0-cm waist/hip→NaN; review BMI>60. |
| CGM | 5-min EGV; clamp [40,400]. Aux pool = post-sentinel ∩ temporal overlap (lock ≥24h or ≥72h). |
| Align (seq/aux) | Resample modalities to a documented canonical grid (HR 1-min, CGM 5-min, …). |

## 11. Caveats (for the discussion)

1. Post-diagnosis data → "risk stratification," not "early diagnosis."
2. HRV→T2D not causal (Rotterdam 2023 Mendelian randomization, DOI 10.1210/clinem/dgad003) — HRV
   marks autonomic dysfunction.
3. Consumer SpO₂ unvalidated overnight — validate or caveat heavily; low coverage.
4. Vivosmart 5 sleep staging moderate-low for REM/deep — lean on duration/regularity.
5. Diasense AUCs unpublished; used 5-fold CV not `recommended_split` — re-run or position as
   distinct architectural angle; don't claim direct numeric beat without matched protocol.
6. Site×label confound: imputed time zone matters for all circadian features; report site×label.
7. Class imbalance (**train insulin n=80** on wearable_core) — weights locked (`balanced` / `Balanced`).
8. **No sex/race features** — hard-onboarding block is smaller than typical wearable-T2D papers;
   gender/race fairness and confound checks are impossible on this release.
9. Effective aux n (≤~1.9k post-mask/overlap) is smaller than raw CGM∩wearable counts — report the
   filtered n, not the ceiling.
10. Survey may dominate SHAP — guardrail in §6; surface honestly if triggered.

## 12. Open questions / decisions still to lock

**Still open:**
1. PAID instrument check (why do healthy have non-zero scores?) — one-line verification.
2. ~~Final coverage thresholds~~ → locked in `pipeline/config.yaml` / `CLEANING.md` (wearable_core=1824).
3. Min CGM↔wearable temporal overlap for aux (≥24h vs ≥72h) — still open for Path B; clean default exists.
4. ~~Year-long-wear truncation~~ → clean window policy in `CLEANING.md` (best_coverage 14d for long wear).
5. Remaining `cmtrt_*` / `mhoccur_*` leakage audit beyond the confirmed hard-exclude list.
6. `mssrffl`/`msslffl` foot monofilament: include or exclude?
7. Lab biomarkers upper-bound arm: run or skip?
8. Canonical time grid & window length for sequence/aux views.
9. CORN vs CORAL final pick (lean CORN — drops proportional-odds).
10. SSL augmentations appropriate for PPG HR + interval sleep/activity.

**Closed (were open; now locked by audit / Training.md):**
- ~~`clinical_site`→time-zone mapping feasibility~~ → feasible; UAB Central, UW/UCSD Pacific; mandatory.
- ~~SpO₂ coverage on full~~ → 71.8% (1,638); stays Tier 3; does not gate primary pool.
- ~~Sex/race availability~~ → not available; struck from onboarding.
- ~~Stress scale~~ → 0–100; high thresholds ≥51 / ≥76.
- ~~Headline architecture~~ → B4 + rep-distill under LUPI **ran and null for deployable raise**;
  B1/B2 ablations **frozen**; Path A first; run order **B1 → B2 → B4 → B3** (B3 last remaining).
- ~~condition_occurrence ICD leakage~~ → no ICDs; self-report dup of observation.
- ~~`via*` blanket drop~~ → per-field: keep via1–3, drop clinical/retinal via*.

## 13. References (DOIs)

Benichou 2018 (HRV meta) 10.1371/journal.pone.0185160 · Aune 2015 (RHR)
10.1016/j.numecd.2015.02.008 · Shan 2015 (sleep duration) 10.2337/dc14-2073 · Cappuccio 2010
(sleep) · Liu/Zhu 2025 10.1080/07853890.2024.2447422 · Boonpor 2023 (MVPA)
10.1186/s12916-023-02851-5 · Strain 2023 10.2337/dc22-1467 · Zhou 2025 (sedentary)
10.1186/s12889-025-24359-8 · Wu 2025 (cosinor RAR) 10.1038/s41387-025-00395-6 · Wu 2023
(non-param RAR) 10.1111/dom.15236 · Chaput 2024 (SRI) 10.2337/dc24-1208 · Phillips 2017 (SRI
formula) 10.1038/s41598-017-03171-4 · Rotterdam 2023 (HRV+MR) 10.1210/clinem/dgad003 · Hoshi 2019
(ELSA-Brasil HRV) · Jae 2016 (HRR) · Nguyen 2025 (SpO₂/ODI) 10.2147/NSS.S512262 · Zhang 2026
(hypoxic burden) · Jiang 2023 (watch SpO₂) 10.1371/journal.pdig.0000296 · Graversen 2023
10.4187/respcare.10760 · Windisch 2023 (Apple SpO₂) · Hassanloo 2024 / Mehran 2022 (TLGS BMI
trajectory — **infeasible here, single timepoint**) · Lam 2021 (UK Biobank) 10.2196/23364 ·
WEAR-ME 2026 10.1038/s41586-026-10179-2 · HypnoLaus 2015 10.5665/sleep.4496 · Garmin stress
validation: biorxiv 2025 10.1101/2025.01.06.630177, J Affect Disord 2025
10.1016/j.jadr.2025.100974 · Firstbeat HRV white-paper (science behind Firstbeat HRV analytics) ·
Vapnik 2015 LUPI (privileged information framing for Path B). AI-READI docs:
https://docs.aireadi.org/docs/3/dataset/overall-description · MoP (SpO₂ protocol):
https://pmc.ncbi.nlm.nih.gov/articles/PMC13060450/
