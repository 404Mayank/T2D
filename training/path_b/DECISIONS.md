# Path B — decisions log

Living log. Methodology authority: repo-root `Training.md`.  
Data contract: `PROCESSED.md`. FE plan: `PLAN_B1_DATA.md`.

---

## 2026-07-14 — B1 data readiness + daily FE

### Verdict
- **No re-clean.** `clean/*` + `pool_masks` (`aux_eligible=1685` ⊆ `wearable_core=1824`) are sufficient.
- **FE-only** Path B daily matrices added for B1.

### Locks
| Topic | Decision |
|---|---|
| Local civil day / hour | Re-derive from **UTC `timestamp` + `zone`** (`shared_windows`). Do **not** trust parquet-flattened `*_local` (all LA) for wall clock. |
| Sleep → day | **Onset-date sessions** (bout gap < 30 min); duration = non-awake bout sum |
| CGM valid day | `n_readings ≥ 72` (config) |
| Watch valid day | `hr_n ≥ 60`; keep days with any HR; mask via `watch_day_valid` |
| CGM 8-vector | mean, sd, cv, min, max, TIR[70–180], TBR&lt;70, TAR&gt;180 (6 dof; collinear OK) |
| `cgm_person` | Ship with `cgm_daily` (B2-ready) |
| RR in watch_daily | Yes |
| Seq length (train later) | Variable; pad ≤ 16 |
| Overlap policy | Keep span-overlap aux gate; concurrent day stats reported at FE acceptance |

### Critique disposition (`PLAN_B1_DATA`)
- **Accepted:** site-tz FE fix (blocker), sleep session rule (high), concurrent acceptance, open locks.
- **Rejected / deferred:** re-clean for tz; Path A GREEN re-FE for circadian fairness (document only: B1 daily is more UAB-correct than frozen Path A person features; λ=0 vs λ>0 still controlled).

### Artifacts built
| File | Shape | Notes |
|---|---|---|
| `features/cgm_daily.parquet` | 19805 × 12 | 1924 pids |
| `features/cgm_person.parquet` | 1924 × 12 | daymean of valid days |
| `features/watch_daily.parquet` | 22844 × 24 | 1824 wearable_core |

### Acceptance (full)
- Aux valid CGM days: median **11** (min 7)
- Core valid watch days: median **12** (min 7)
- Aux both-valid days: median **11** (min 4, p10 9; none zero)
- UAB pid 7025: FE days match Chicago UTC convert (0 symdiff); nocturnal mean ~58 bpm
- No forbidden meta/label columns
- `watch_green` untouched (1824 × 31)

### Code
- `pipeline/fe/local_time.py`, `cgm_daily.py`, `watch_daily.py`
- `pipeline/run_fe.py` blocks: `cgm_daily`, `watch_daily`
- Config: `features.cgm_daily`, `features.watch_daily`, `runtime.fe_workers`

### Next
Implement B1 training package per `PLAN_B1_TRAIN.md` (critiqued/revised) — not B4 grids yet.

---

## 2026-07-14 — B1 training plan locks (post-critique)

Source: `PLAN_B1_TRAIN.md` (critiquer → revise; disposition applied).

| Topic | Lock |
|---|---|
| Scientific claim | λ>0 vs λ=0 on **identical** backbone (not “beat Path A” as sole claim) |
| Glu head primary | **Day-level** masked MSE on 8 daily CGM stats (v1b); person-level secondary |
| Glu pool | `wearable_core ∧ aux_eligible` valid days only |
| Backbone | `attn_lstm_64` (BiLSTM + attention); h=128 optional sensitivity |
| Features | 18 daily dims; **no** `hr_n/stress_n/rr_n` in primary |
| Class weights | Inverse-freq on train core, sum-normalize |
| λ grid | {0, 0.3, 0.5, 1.0}; science table = all λ; not best-of-3 alone |
| Multi-task win | Per-λ test paired ΔAUC boot CI **lo > 0** (n=2000, seed 42) |
| Path A floor | Informational (LSTM≠CatBoost; site-tz FE differs) |
| Calibration | Val per-class isotonic; raw ranking is claim; Brier diagnostic |
| Proceed to B4 | Always after B1 report |

---

## 2026-07-14 — B1 package implemented + smoke

### Package
`training/path_b/b1/` — data/model/train/evaluate/run + config.

### Impl locks applied
- Truncate: prefer `cgm_day_valid`, earliest tie-break (not last-16)
- `glu_mask` on concurrent watch-valid ∧ cgm-valid ∧ aux only
- Sequence rows = watch-valid only (pack_padded-safe)
- BiLSTM `2h→h` before glu head + attention
- Seed reset before each λ model init
- Class weights: train CE only
- Post-grid paired bootstrap vs λ=0
- Smoke stratified by split + label

### Torch
- Installed **CPU** `torch==2.13.0+cpu` (Py3.14). ROCm wheel index had no match for this Python.
- Local AMD 5600M: no CUDA; ROCm torch not installed this session. Full grid → Lightning CUDA or re-try ROCm wheels later.

### Smoke `smoke_b1` (CPU, 64 pids, 3 ep, λ∈{0,0.5})
- Pipeline OK end-to-end (~23s after stratify fix)
- Metrics **not claimable** (tiny n / 3 epochs): test 4-AUC ~0.33–0.36
- Δ λ0.5−λ0 ≈ +0.02, CI includes 0
- Artifacts: `training/path_b/b1/artifacts/smoke_b1/`

### Full grid `b1_grid_20260714` (RX 5600 ROCm torch 2.12.1)

| λ | val 4-AUC | test 4-AUC | test bin | Δ vs λ0 test CI lo>0 |
|---:|---:|---:|---:|---|
| 0.0 | 0.564 | 0.510 | 0.523 | — |
| 0.3 | 0.521 | 0.540 | 0.558 | No (+0.030 [−0.022,+0.085]) |
| 0.5 | 0.518 | 0.544 | 0.578 | No (+0.034 [−0.016,+0.084]) |
| 1.0 | 0.510 | 0.504 | 0.566 | No |

**Outcomes:** multi-task **does not** clear pre-registered win (CI lo>0). Sequence B1 **below** Path A floor 0.666 (informational).  
**ROCm note:** `torch.backends.cudnn.enabled=False` required (MIOpen reduction fail on gfx1010).  
**Report:** `REPORT_B1.md`.

### Next
B2 two-stage ablation and/or B4 trajectory plan; optional B1 h=128 / richer daily FE sensitivity — not blocking B4.

---

## 2026-07-14 — B1 underperformance audit (why ~0.51)

**Full write-up:** `AUDIT_B1_UNDERPERF.md` (dual agent audit + parent verify + web research).

### Root-cause locks (confirmed)

| ID | Severity | Finding | Decision |
|---|---|---|---|
| C1 | **Critical** | `_sleep_daily` treats `datetime64[ms]` int64 as ns (`/1e9`) → duration ~1e6× small; gaps never open sessions → 1 “sleep day”/pid, values ~1e-5 h | **Must fix FE** before any B1 re-claim; rebuild `watch_daily`. Use `.dt.total_seconds()` like `watch_green`. Redefine `sleep_n_bouts` = sessions not stage rows. |
| C2 | **Critical** | Watch inputs **not** z-scored for LSTM (only glu targets); steps/sedentary dominate (std ratio ~1e8) | **Must add** train-only feature StandardScaler (or robust) in `b1/data.py` before tensorize. |
| C3 | Critical (emergent) | λ=0 CE flat 1.40→1.38; val AUC 0.564 | Expected under C1+C2; require **tiny-subset overfit** gate after fix. |
| C4 | High | sleep fill_zero + median 0 → near-constant channel | Revisit impute after C1; prefer missing mask. |
| C5 | Medium (design) | Daily 18-d set omits GREEN SRI/RAR/… | After C1–C2, if still weak → **late-fuse GREEN** into `z` (research-supported). |
| C6 | Low/med | inverse-freq class weights heavy on insulin | Keep for now; revisit post-floor. |

### Explicit non-fixes / false alarms
- Do **not** re-FE Path A GREEN for this (its sleep path is correct).
- Glu target z-score, aux glu_mask, pack_padded/attention: **OK**.
- “Worse than Path A” alone is **not** proof of a bug (literature: trees often win at n~1–2k short T); **flat CE + dead sleep + unscaled X** are the bugs.
- B1 multi-task null is **not** a B4 kill; primary task was broken.

### Research-backed stance (condensed)
- At this n/T, cold BiLSTM rarely beats GBDT without clean inputs + static fusion (Liao 2022; TabZilla; Kuznetsova 2022).
- Z-score + missingness masks are load-bearing for RNNs (GRU-D / Lipton).
- Fix primary (λ=0) before trusting λ>0 or LUPI claims (Simpson MTL; Nobari LUPI skepticism).
- Hybrid static⊕sequence or sequence-latents→GBM is the pragmatic raise path if pure seq still lags.

### Implementation order (locked)
1. Fix `_sleep_daily` units/sessions/bouts → rebuild `watch_daily` (smoke + full).  
2. Input z-score in B1 data pipeline.  
3. Overfit smoke (50 pids) → full λ=0.  
4. Optional GREEN fusion if val still ≪ ~0.60.  
5. Re-run λ∈{0, 0.5} only; update `REPORT_B1.md` with new run id (do not overwrite old grid as claim).  
6. Then resume B2/B4 ladder.

**Why this order:** C1/C2 invalidate all prior B1 test numbers for scientific claim; multi-task and architecture search on broken inputs would waste runs.

---

## 2026-07-15 — B1 C1/C2 fix + retest

**Plan:** `PLAN_B1_FIX.md` (critiquer → revise; disposition applied).  
**Code:** `pipeline/fe/watch_daily.py` `_sleep_daily`; `b1/data.py` feat z-score; `b1/config.yaml` fill_zero; ckpt/meta persist feat scale.

### Definition lock — `sleep_n_bouts`
**Sessions per onset day**, not stage rows. Deliberate change vs earlier PLAN_B1_DATA wording (“stage bouts in session”); stage counts 100–700 are useless after z-score. Duration still non-awake bout sum; all-awake session → duration NaN. Units: **Timedelta / `total_seconds` only** (no int64÷1e3).

### Impute / scale locks
| Topic | Lock |
|---|---|
| `sleep_duration_hours` | **not** fill_zero; train median on observed |
| `sleep_n_bouts` + activity mins/steps | fill_zero OK |
| Feature scale | train-only z-score after impute; **observed-only** mean/std for non-fill_zero cols; fill_zero cols post-fill |
| Glu target z-score | unchanged |
| Missing-mask / GREEN fuse | deferred; not needed for G2 AUC pass |

### Gates
| Gate | Result |
|---|---|
| G0 FE rebuild | **PASS** — mean sleep 6.64 h; 77% watch_valid coverage; 8.98 sleep days/pid; bouts p90=2 |
| G1 `b1_overfit50_fix` | **PASS** — CE 1.38→0.79; val 0.775 |
| G2 `b1_fix_20260715` λ=0 | **PASS AUC** val **0.680** / test **0.652** (pre-fix 0.564 / 0.510); CE dynamics 1.35→1.11 (best ckpt ep1) |
| G3 `b1_grid_20260715_fix` λ∈{0,0.5} | multi-task **null**: ΔAUC −0.0003 CI [−0.0022,+0.0018] lo>0=False |
| G4a/G4b | **skipped** (G2 AUC ≥ 0.60) |

### Scientific claims (post-fix)
1. Pre-fix grid `b1_grid_20260714` remains **broken-input baseline only** — not a multi-task ceiling.
2. After C1+C2, **class-only daily sequence is learnable** (~0.65–0.68 4-AUC) — still informational vs Path A floor 0.666 (not arch-matched).
3. **Day-level multi-task CGM still does not clear** pre-registered win (paired CI lo>0) at λ=0.5.
4. Ladder next: **B2 → B4 → B3** (user order). Optional later: GREEN late-fuse / mask / lr schedule if raising pure-seq λ=0 is needed — not blocking multi-task close.

### Artifacts
- `b1/artifacts/b1_overfit50_fix/`
- `b1/artifacts/b1_fix_20260715/`
- `b1/artifacts/b1_grid_20260715_fix/`
- Report: `REPORT_B1.md` (final; superseded pre-fix narrative)

---

## 2026-07-15 — GREEN late-fuse confirmation + B1 freeze

### Implementation
- `green_fusion.enabled` + CLI `--green-fusion`
- Person `watch_green` (all numeric cols, train impute+z) concat to attention `z` → class head only; glu on `h_t` unchanged
- Run: `b1_green_20260715` λ=0 only

### Result (vs post-fix pure seq)
| | val 4-AUC | test 4-AUC | test bin |
|---|---:|---:|---:|
| Pure seq λ=0 | 0.680 | **0.652** | 0.679 |
| + GREEN fuse | 0.686 | **0.638** | 0.660 |
| Path A W0 | — | 0.666 | 0.689 |

**No raise** vs pure seq; still below Path A watch floor. Best epoch still 1.

### Locks / freeze
| Topic | Decision |
|---|---|
| B1 multi-task claim | **Closed null** — `b1_grid_20260715_fix` |
| B1 pure-seq floor | test **0.652** (informational vs W0 0.666) |
| GREEN late-fuse on B1 spine | **Does not** close Path A gap; not a B1 raise path |
| Full C1 static stack on B1 | **Out of scope** for B1 freeze (would change claim mix; Path A already owns it) |
| Further B1 λ / h / loss churn | **Frozen** unless a new plan reopens with explicit gate |
| Ladder next | **B2 → B4 → B3** |

### Interpretation
Daily BiLSTM + optional person GREEN is a different inductive bias from CatBoost-on-GREEN. Missing columns were not the binding constraint after C1+C2; architecture/task (trajectory / two-stage / distill) is. Pre-fix grid remains broken-input history only.

**Report:** `REPORT_B1.md` rewritten as final freeze report.
