# Path B / B1 — data readiness & FE plan

**Date:** 2026-07-14  
**Status:** FE **implemented & accepted** 2026-07-14 (`cgm_daily` / `cgm_person` / `watch_daily`). See `DECISIONS.md`.  
**Scope:** Decide whether cleaning must change for B1; if not, plan the minimal FE additions so B1 can train.  
**Authority:** `PROCESSED.md`, `CLEANING.md`, `FEATURES.md` §7–8, `Training.md` §4 B1 / §7 step 2.

**Critique disposition (2026-07-14):** critiquer → **revise**. Accepted: site-tz re-derive in FE (blocker), sleep onset-session rule (high), collinearity note (low/doc), concurrent-day acceptance, open-choice locks. Rejected as false-positive / out-of-scope: re-clean for tz (instants correct; parquet display only), DST edge laundry list, Path A re-FE for circadian fairness (document confound only).

---

## 1. Exploration verdict (what is already on disk)

### Present and usable (no re-clean)

| Asset | Status | Notes |
|---|---|---|
| `clean/cgm.parquet` | **OK** | 5.06M EGV rows, 1924 pids, mg/dL, bounds [40,400], local + UTC ts, ~5-min median gap |
| `clean/heart_rate|stress|rr|sleep|activity` | **OK** | HR-windowed, local time, row-group per pid |
| `meta/pool_masks.parquet` | **OK** | `aux_eligible=1685`, all ⊆ `wearable_core`; overlap flags 0/24/72h all true on aux |
| `meta/shared_windows.parquet` | **OK** | HR-anchored best-coverage window (~14d) |
| `features/watch_green.parquet` | **OK** | Path A floor, n=1824, person-level only |
| Clinical blocks | **OK** | Not needed for B1 watch-only claim; available for later deployable stacks |

### Missing for B1 (FE only — not cleaning)

| Asset | Status | Why B1 needs it |
|---|---|---|
| Daily CGM summary matrix | **MISSING** | Multi-task glucose head target (“8 daily CGM summaries”) |
| Daily wearable feature matrix | **MISSING** | Sequence backbone input (day × features per person) |
| Person-level CGM summary matrix | **MISSING (nice-to-have for B1; useful for B2)** | Tabular glucose head / two-stage handoff later |
| Aligned multi-modal 5-min grid | **MISSING** | **Not required for B1**; required for B4 trajectory |

### Cleaning policy judgment

- **Do not re-run `run_clean` for B1.** Clean series + pools already match the Path A freeze and aux gates.
- Overlap is **span intersection**, not minute-level concurrent wear (`CLEANING.md` Stage 5 note). Adequate for B1 daily summaries inside the shared window; **tighten only when building B4 View-B grids**.
- `aux_eligible` already requires: wearable_core ∧ RR ∧ CGM≥8d ∧ overlap≥24h. Empirically all 1685 aux also pass ≥72h span overlap — lock stays at 24h for now (matches config).
- Residual known limitation (document, don’t fix now): span overlap ≠ concurrent wear; year-long wearers already truncated by shared window.

**Conclusion:** pipeline *cleaning* is ready; pipeline *FE* needs a Path-B daily block.

---

## 2. What B1 actually consumes

From `Training.md` B1 (controlled ablation):

> Same **64-hidden attention backbone**, ± **glucose head**, **summary-CGM** target.  
> Teammate used daily wearable sequences + 8 daily CGM scalars; their fail was confounded by backbone size.

So B1 is **not** “tabular watch_green multi-task.” It is:

1. **Input sequence** \(X_{1:T}\): daily wearable feature vectors (length \(T\) days, masked).
2. **T2D head** on pooled backbone embedding → 4-class (primary metrics).
3. **Glucose head** on same embedding (or per-day decoder) → regress **daily CGM summary vector** (only on days/pids with CGM).
4. Compare **identical backbone** with λ=0 (class-only) vs λ>0 (multi-task).

**Bars to beat (Path A freeze):**

| Reference | 4-AUC | Binary |
|---|---:|---:|
| Watch-only floor (paper claim) | **0.666** | 0.689 |
| Deployable C1 (secondary) | **0.738** | 0.831 |

B1 is watch-side LUPI ablation → primary comparison is **vs watch floor 0.666**. C1 is a separate deployable ceiling, not the B1 claim bar.

---

## 3. Proposed FE additions (minimal, additive)

### 3.1 New module layout

```
pipeline/fe/
  watch_green.py          # unchanged (Path A)
  cgm_daily.py            # NEW — daily + optional person CGM summaries
  watch_daily.py          # NEW — daily wearable features for sequences
pipeline/run_fe.py        # wire blocks: watch, cgm_daily, watch_daily, …
```

Outputs (features only — no label/split/site):

| File | Grain | Pool default | Contents |
|---|---|---|---|
| `features/cgm_daily.parquet` | person_id × day_local | pids with cleaned CGM (or `aux_eligible`) | 8 glucose daily stats + `n_readings`, quality flags |
| `features/cgm_person.parquet` | person_id | same | aggregates over valid CGM days (B2-ready; cheap side product) |
| `features/watch_daily.parquet` | person_id × day_local | `wearable_core` | daily wearable vector + `n_hr_minutes`, quality flags |

Join labels/splits/aux flags at train time from `pool_masks` (same contract as Path A).

### 3.2 Day definition (lock) — **site-correct wall clock**

**Bug (confirmed):** cleaned `timestamp_local` / `*_local` columns are stored as
`datetime64[*, America/Los_Angeles]` for **all** pids (parquet single-tz column).
UTC `timestamp` instants are correct; wall-clock for UAB is **shifted −2h** vs Chicago
(~8% of UAB CGM readings mis-bucketed by calendar day; nocturnal means biased).
Path A `watch_green` inherited this; **do not re-clean** for B1 — fix at FE read time.

**Lock:**
1. Prefer **UTC `timestamp` / `start_time` / `end_time`** + per-pid `zone` from
   `meta/shared_windows.parquet` (`America/Chicago` UAB, `America/Los_Angeles` UW/UCSD).
2. Wall-clock: `ts.dt.tz_convert(zone)` then civil day as **string** `YYYY-MM-DD`
   via `strftime("%Y-%m-%d")` (avoid `floor("D")` DST traps; match `watch_green` style).
3. **Do not** use `timestamp_local.dt.date` / `.dt.hour` for day or nocturnal features.
4. Window membership: cleaned series are **already** HR-windowed; do not re-filter by
   Pacific-displayed `window_*_local` dates. Optional: intersect using UTC window bounds
   if ever re-windowing (not needed for v1 FE on `clean/*`).
5. Document confound: B1 daily circadian features are more UAB-correct than frozen Path A
   person-level GREEN; λ=0 vs λ>0 ablation remains controlled; B1-vs-A comparison is not
   purely architecture-matched on circadian encoding.

### 3.3 Daily CGM summaries (8-vector)

Computed on cleaned EGV `blood_glucose` for that **site-local** civil day.  
**Valid day gate (lock):** `n_readings ≥ 72` (~25% of 288 expected 5-min samples), config key
`features.cgm_daily.min_readings_per_day`. Spot-check: ~93% of days pass; median ~288.

| # | Name | Definition |
|---|---|---|
| 1 | `cgm_mean` | mean glucose (mg/dL) |
| 2 | `cgm_sd` | population sd (ddof=0) |
| 3 | `cgm_cv` | sd/mean (0 if mean==0) |
| 4 | `cgm_min` | min |
| 5 | `cgm_max` | max |
| 6 | `cgm_tir_70_180` | fraction in [70, 180] |
| 7 | `cgm_tbr_70` | fraction < 70 |
| 8 | `cgm_tar_180` | fraction > 180 |

**Collinearity (doc only):** `cv = sd/mean`; `TIR+TBR+TAR = 1` → 6 dof in “8” targets.
Keep 8 for teammate-faithful multi-output head; training plan must not treat them as independent signals.

Also write (not part of the “8” head unless ablation expands):

- `cgm_n` — reading count  
- `cgm_day_valid` — bool after gate  
- optional later: GMI, nocturnal mean — **out of v1**

**Person-level `cgm_person` (ship in same pass):** mean of daily **valid** vectors + `n_valid_days` + overall mean/sd on all EGV in window. B2-ready; B1 may ignore.

### 3.4 Daily wearable features (sequence input)

Goal: compact day-vector analogous to teammate “~41 wearable/day,” but **derived from our cleaned modalities** and close to GREEN families (not a new literature sweep).

**Hour-of-day** for nocturnal/daytime splits must use **site-correct** local hours (§3.2), not Pacific-flattened `timestamp_local`.

**v1 columns** (config-listed; ~18 feats):

| Family | Features |
|---|---|
| HR | `hr_mean`, `hr_sd`, `hr_min`, `hr_max`, `hr_n` (minute count), `hr_nocturnal_mean` (00–06 local), `hr_day_mean` (08–20 local) |
| Stress | `stress_mean`, `stress_sd`, `stress_pct_medium_plus`, `stress_pct_high`, `stress_n` |
| RR | `rr_mean`, `rr_sd`, `rr_n` (**include** — aux gates RR; free) |
| Sleep | **Onset-date session rule (lock):** group stage bouts into sessions if gap from previous bout end → next start **≥ 30 min** (new session); assign session to **first bout’s site-local start date** (`YYYY-MM-DD`). Duration via **`.dt.total_seconds()`** (unit-safe; never int64/1e9 on ms stamps). Duration = sum of non-`awake` bout lengths in session (hours); all-awake session → duration NaN. Features: `sleep_duration_hours`, `sleep_n_bouts` = **session count** per onset day (**definition change 2026-07-15** vs earlier “stage bouts in session”; see DECISIONS / PLAN_B1_FIX). |
| Activity | `steps_sum`, `mvpa_min`, `light_min`, `sedentary_min` (same intensity map as GREEN); day = site-local start date of interval |

**Valid day gate:** `hr_n ≥ 60` (config). **Keep + mask:** emit one row per civil day that appears in any modality for the pid (or densify only days with HR — v1: **emit days with any HR sample**, set `watch_day_valid = hr_n ≥ 60`; missing stress/RR/sleep/activity → NaN). Trainer pads to per-pid length, hard-cap 16.

No SpO₂ in v1 (Tier-3 coverage).

### 3.5 Alignment contract for the trainer (not FE)

FE emits **long tables**. Training package will:

1. Build per-pid day index from observed `day_local` keys (variable length; **pad to max ≤16**).
2. Left-join `watch_daily` and `cgm_daily` on `(person_id, day_local)`.
3. Masks: `watch_mask` ← `watch_day_valid`, `cgm_mask` ← `cgm_day_valid`.
4. **T2D loss:** all `wearable_core` with ≥ \(T_{\min}\) valid watch days (e.g. 7) — **includes non-aux**.
5. **Glucose loss:** only days with `cgm_day_valid` (typically aux); **never** drop non-CGM pids from T2D head.
6. Split strictly by `recommended_split`.

**Concurrent wear note:** span-overlap aux gate ≠ per-day concurrent HR∩CGM. Spot-check (30 aux pids, site-correct days): median **11** days with both valid (min 5). FE acceptance will report distribution for full aux.

LUPI: CGM never at inference; glucose head train-only.

---

## 4. Pipeline / config changes

### `pipeline/config.yaml` (new section sketch)

```yaml
features:
  watch_green: { enabled: true, require_pool: wearable_core }
  cgm_daily:
    enabled: true
    min_readings_per_day: 72
    require_pool: null          # all pids with clean CGM; trainer filters aux
  watch_daily:
    enabled: true
    min_hr_minutes: 60
    require_pool: wearable_core
```

### `run_fe` CLI

```bash
# Path A (unchanged)
.venv/bin/python -m pipeline.run_fe --blocks watch

# Path B daily (B1 inputs)
.venv/bin/python -m pipeline.run_fe --blocks cgm_daily,watch_daily

# Smoke
.venv/bin/python -m pipeline.run_fe --blocks cgm_daily,watch_daily --max-participants 20
```

### Explicit non-goals for this FE pass

- No re-clean / no window policy change  
- No 5-min multi-modal grid (B4)  
- No diet / clinical block changes  
- No model training code in this pass (training plan is next)  
- No minute-level concurrent HR∩CGM overlap redefinition  

---

## 5. Docs to update when implementing

| Doc | Update |
|---|---|
| `PROCESSED.md` | New feature files + Path B train join sketch |
| `CLEANING.md` | FE stage note: daily Path B blocks; still “clean once” |
| `FEATURES.md` | View (b) daily partial; list daily columns; B1 consumes daily not 5-min |
| `AGENTS.md` | Point to `training/path_b/` when package exists |
| `training/path_b/DECISIONS.md` | Lock day def, 8 CGM stats, gates, no re-clean |
| `training/path_b/PLAN_B1_DATA.md` | This file (status → done after build) |

---

## 6. Acceptance checks (FE)

1. `cgm_daily`: rows for ~all cleaned CGM pids; aux_eligible **median ≥8** `cgm_day_valid` days.
2. `watch_daily`: 1824 wearable_core pids; median `watch_day_valid` days ≥7.
3. No forbidden columns (`label`, `recommended_split`, `clinical_site`, pool flags) in feature files.
4. Full-aux concurrent: report p10/median/min of days with both `watch_day_valid` and `cgm_day_valid`; **median ≥5** required.
5. **UAB tz check:** for a UAB aux pid, FE `day_local` matches `timestamp.dt.tz_convert("America/Chicago").strftime("%Y-%m-%d")` with **0 mismatches** on a sampled day; `hr_nocturnal_mean` in plausible rest band **[45, 85]** bpm when defined.
6. Runtime OK on local 16 GB (per-pid streaming).
7. Path A `watch_green` **not rewritten** when running only new blocks.

---

## 7. Implementation order (this pass only)

1. Critique this plan → address real issues.  
2. Implement `cgm_daily.py` + `watch_daily.py` + `run_fe` wiring + config keys.  
3. Run smoke (`--max-participants 20`) then full FE.  
4. Update docs + write `DECISIONS.md` entries.  
5. **Stop** — next session/step is B1 **training** package plan (backbone, loss, λ sweep, metrics), not B4.

---

## 8. Open choices — **locked after critique**

| # | Choice | Lock |
|---|---|---|
| 1 | Sleep → day | **Onset-date sessions** (gap < 30 min); not raw `end.date()` per bout |
| 2 | CGM valid day | **`n ≥ 72`** (config-tunable) |
| 3 | `cgm_person` | **Ship** in same FE pass |
| 4 | RR in watch_daily | **Yes** |
| 5 | Seq length | **Variable**, pad ≤ **16** at train time |
| 6 | Local time | **Re-derive from UTC + `zone`** in FE; never trust flattened `*_local` for civil day/hour |
