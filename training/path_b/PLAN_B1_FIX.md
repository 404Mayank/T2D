# PLAN_B1_FIX — Sleep FE + input scale → B1 retest

**Date:** 2026-07-15  
**Authority:** `AUDIT_B1_UNDERPERF.md`, `DECISIONS.md` underperf section, `prompt.md` locked order.  
**Scope:** C1 FE fix → rebuild `watch_daily` → C2 feature z-score → overfit gate → λ=0 → optional mask → optional GREEN late-fuse → λ∈{0,0.5} retest.  
**Not in scope:** Path A re-open, B4 5-min grid, re-clean, full λ{0.3,1.0} grid as claim.

---

## 1. Goals / success

| Gate | Pass rule |
|---|---|
| G0 FE | Non-null sleep days: mean `sleep_duration_hours` in **[~3, ~9]** and near GREEN ~5.45; sleep non-null on **≥35%** of watch-valid days; mean sleep days/pid (pids with any sleep) **> 1**; `sleep_n_bouts` p50/p90 small (1–few). **Joint** evaluation (bouts alone can false-pass if gap unit still broken). |
| G1 Overfit | **`--max-participants 50 --lambdas 0` without `--quick`** (full max_epochs/es). Train CE **drops hard** toward ≲1.0 (not flat near ln4≈1.386). `--quick` = code-smoke only. |
| G2 λ=0 backbone | Full core, `--lambdas 0`: **val macro-OVR AUC ≥ 0.60** **and** train CE ≲ **1.1** (broken baseline val peak 0.564 / CE~1.38 — bar must clear both). |
| G3 Multi-task | **One** run `--lambdas 0,0.5` (paired boot needs co-located preds); test paired ΔAUC (λ0.5−λ0) boot CI **lo > 0**. |
| G4a Mask (conditional) | If G2 fails: add `sleep_present` mask channel → re-run λ=0. |
| G4b GREEN (conditional) | Only if still fails after G4a: late-fuse person GREEN into `z` before class head. |

Pre-fix grid `b1_grid_20260714` stays as **broken-input baseline** — do not overwrite interpretation.

---

## 2. C1 — `_sleep_daily` (pipeline/fe/watch_daily.py)

### Bug
```python
start_ns = s.astype("int64")   # datetime64[ms] → milliseconds, not ns
dur_h = (end_ns - start_ns) / 1e9 / 3600.0   # 1e6× too small
gap_m = (...) / 1e9 / 60.0                   # gaps never ≥ 30 → 1 session
```

### Fix (mirror `watch_green` / activity path)
1. Duration: **`(e - s).dt.total_seconds() / 3600.0` only** — unit-invariant. **No** int64 epoch ÷ 1e3.
2. Gap: **`(start[i] - end[i-1]).total_seconds() / 60.0`** for session split (`gap_min=30`).
3. **`sleep_n_bouts` = sessions per onset day** — **deliberate definition change** (not a unit bug). Old PLAN_B1_DATA said “stage bouts in session”; stage-row counts (100–700) are useless after z-score. Bless in DECISIONS; rewrite PLAN_B1_DATA §3.4.  
   After session ids: once per session, onset day from first bout → `day_bouts[day] += 1`; duration = sum non-awake stage lengths. If session has zero asleep time → duration **NaN** (not true-0).
4. Keep onset-date session rule (locked in PLAN_B1_DATA).

### Bout counting rewrite (pseudocode)
```
sess ids from gap ≥ 30 min (total_seconds units)
for each session sid:
  day = onset_day[first_index(sid)]
  day_bouts[day] += 1                    # once per session
  asleep_sum = sum(dur_h of non-awake stages in sid)
  day_dur[day] += asleep_sum if asleep_sum > 0 else treat as NaN contribution
```

### Rebuild
```bash
.venv/bin/python -m pipeline.run_fe --blocks watch_daily --max-participants 20
.venv/bin/python -m pipeline.run_fe --blocks watch_daily
```

### FE acceptance checks (all required, jointly)
- Mean duration on non-null days in **[~3, ~9]** h, near GREEN ~5.45.
- Sleep non-null on **≥35%** of watch-valid days (was ~8%).
- Mean sleep days/pid with any sleep **> 1**.
- `sleep_n_bouts` p50/p90 small (1–few), not hundreds.
- Spot-check one pid vs raw clean sleep with `total_seconds`.

No change to `watch_green` (correct).

---

## 3. C2 — train-only feature z-score (`b1/data.py`)

### Current
- Glu targets z-scored (keep).
- Features: fill_zero then train median impute → raw scale into LSTM.

### Fix
After impute, **before** building person sequences:
1. Fit train ∩ `watch_day_valid`:
   - **`fill_zero` cols:** mean/std on post-fill rows.
   - **Other cols (incl. sleep_duration):** mean/std on **observed-only** (pre-impute `notna()`). Avoids median-mass compressing sleep std.
   - `feat_std[c] = max(std, eps)` with `eps` e.g. `1e-6`.
2. Apply to **all** splits: `(x - mean) / std` (order impute→scale is correct).
3. Persist `feat_mean` / `feat_std` on `SequenceBundle`, run meta, **and train checkpoint**.
4. **No min-max.** log1p on `steps_sum` deferred unless G1/G2 still fail.
5. Smoke assert: `impute_values['sleep_duration_hours'] > 0` after C1.

### C4 impute tweak (with C1)
- **Remove `sleep_duration_hours` from `fill_zero`.** Missing → NaN → train median on observed. All-NaN edge → 0 + assert fail.
- **`sleep_n_bouts`:** keep fill 0.
- Activity fill_zero keep.
- Missing-mask channels: first retest **without**; if G2 fails → G4a before GREEN.

### Config
- `b1/config.yaml`: drop `sleep_duration_hours` from `fill_zero`.

### Bundle / artifacts
`SequenceBundle.feat_mean`, `feat_std`; dump in run meta + ckpt.

---

## 4. Sanity → train ladder

| Step | Command / action | Gate |
|---|---|---|
| 4a | Rebuild watch_daily smoke 20 → full | G0 |
| 4b | Code C2 + config fill_zero | finite stats; sleep impute >0; post-scale std≈1 on observed train |
| 4c | `--run-id … --lambdas 0 --max-participants 50 --device cuda` (**no** `--quick`) | G1 |
| 4d | `--run-id b1_fix_YYYYMMDD --lambdas 0 --device cuda` | G2 |
| 4e | If G2 fail: `sleep_present` mask → re-run λ=0 | G4a |
| 4f | If still fail: GREEN late-fuse | G4b |
| 4g | **One** run `--run-id b1_grid_YYYYMMDD_fix --lambdas 0,0.5` (4d λ=0 is **not** G3 λ=0 ref) | G3 |
| 4h | REPORT addendum + DECISIONS; keep `b1_grid_20260714` as pre-fix | docs |

### Protocol locks (unchanged)
- Day-level glu head; glu only aux∧watch_valid∧cgm_valid  
- No coverage counts in primary X  
- Seed reset per λ; ES on val macro-OVR AUC  
- Paired bootstrap n=2000 vs λ=0  
- Class weights inverse-freq train CE only  
- ROCm: `cudnn.enabled=False`; `HIP_VISIBLE_DEVICES=0`

### Mask then GREEN (only if G2 fails)
**G4a:** `sleep_present` channel; re-run λ=0.  
**G4b:** load `watch_green`, train-only impute+z, `z_fused = concat(z_seq, green)` → class head; glu stays on h_t; flag `green_fusion: true`. Do not implement preemptively.

---

## 5. Docs to update as we go

| Doc | Update |
|---|---|
| This plan | Critique disposition (below) |
| `DECISIONS.md` | Fix applied + bout definition blessing + run ids |
| `REPORT_B1.md` | Addendum: pre-fix invalid; new grid |
| `PLAN_B1_DATA.md` | `sleep_n_bouts` = sessions |
| `PLAN_B1_IMPL.md` / TRAIN | Input z-score; sleep fill policy |
| `AGENTS.md` | Index PLAN_B1_FIX if useful |

---

## 6. Explicit non-goals / false-fix avoid list

- Do not “fix” Path A GREEN sleep.  
- Do not re-clean raw pipeline for C1.  
- Do not claim multi-task win from pre-fix grid.  
- Do not expand λ grid beyond {0,0.5} for retest claim.  
- Do not add survey features to “watch-only” path.  
- Do not change class-weight scheme until floor is learnable.  
- Cap activity minutes (C7) deferred.

---

## 7. Implementation file touch list

| File | Change |
|---|---|
| `pipeline/fe/watch_daily.py` | C1 `_sleep_daily` units + session bout count |
| `training/path_b/b1/data.py` | C2 feat z-score (observed-only for non-fill_zero); bundle fields |
| `training/path_b/b1/config.yaml` | fill_zero without sleep_duration_hours |
| `training/path_b/b1/run.py` / `train.py` | persist feat scale stats (minimal) |
| docs listed above | after runs |

Model architecture: **no change** for C1–C2 unless G4 later.

---

## Critique disposition

Source: critiquer `opencode-go/glm-5.2:high` on this plan (2026-07-15).

| Finding | Sev | Disposition |
|---|---|---|
| G2 bar (0.55) ≤ pre-fix val 0.564 → false-pass | High | **Accept** → val AUC ≥ 0.60 **and** train CE ≲ 1.1 |
| G1 via `--quick` (3 ep) cannot overfit | High | **Accept** → G1 forbids `--quick`; 50-pid full epochs λ=0 |
| `sleep_n_bouts` = sessions is definition change vs PLAN_B1_DATA | Med | **Accept** → bless in DECISIONS; rewrite PLAN_B1_DATA |
| z-fit on imputed sleep compresses variance | Med | **Accept** → observed-only mean/std for non-fill_zero |
| Mask before GREEN | Med | **Accept** → G4a then G4b |
| 4d/4f omit `--lambdas` → default 4-λ grid | Med | **Accept** → explicit flags; 4d ≠ G3 ref |
| int64÷1e3 fallback fragile | Med | **Accept** → Timedelta only |
| G0 “≫8%” hand-wave | Low/Med | **Accept** → ≥35% + joint checks |
| Order impute→scale wrong | — | **Reject (FP)** |
| Sleep median all-zero risk | — | **Reject (FP)** at full n; keep smoke assert |
| Train z-fit leakage | — | **Reject (FP)** — train-only like glu |

**Verdict applied:** revise-before-implement edits folded into §§1–4.

---

## Execution log (2026-07-15)

| Step | Run / artifact | Result |
|---|---|---|
| 4a G0 | full `watch_daily` rebuild | **PASS** (mean 6.64 h; 77% coverage; 9 days/pid) |
| 4b C2 | `b1/data.py` + config | **PASS** |
| 4c G1 | `b1_overfit50_fix` | **PASS** CE→0.79 |
| 4d G2 | `b1_fix_20260715` λ=0 | **PASS** val 0.680 / test 0.652 |
| 4e G4a mask | skipped | G2 AUC cleared |
| 4f G4b GREEN | `b1_green_20260715` | **no raise** test 0.638 |
| 4g G3 | `b1_grid_20260715_fix` | multi-task **null** |
| 4h docs | `REPORT_B1.md` final freeze | **B1 CONCLUDED** |

G4b run after G3 (confirmation of “missing person summaries?”): **negative**. B1 frozen; next **B2**.
