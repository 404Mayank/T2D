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

---

## 2026-07-15 — B2 data readiness + plan (pre-implement)

**Plan:** `PLAN_B2.md` (critiquer → revise; disposition applied). **Not implemented yet.**

### Data readiness
| Topic | Decision |
|---|---|
| Re-clean for B2? | **No** |
| New FE for B2 v1? | **No** — `cgm_person` + GREEN + C1 blocks sufficient |
| B4 5-min grid | Still missing; **not** a B2 blocker |
| Stage-1 R² probe (HistGB) | ~0.07 on `cgm_mean` — modeling risk, not missing data |
| Oracle headroom probe | C1+true CGM ~0.79 vs C1~0.71 (rough) — privilege real if Stage-1 were perfect |

### B2 design locks (summary; full in PLAN_B2)
| Topic | Lock |
|---|---|
| Role | Two-stage point-estimate handoff ablation; not headline |
| User ambition bar | Beat frozen C1 (0.7378 / 0.8309); **primary fair compare = T1 vs D1** if D1 drifts |
| Stage-1 X / Y | W0 GREEN → 8 `cgm_*_daymean`; fit train∩aux only |
| Stage-1 model | **8× LightGBM** primary; CatBoost multi = sensitivity |
| Stage-2 | Path A family (CatBoost+LGBM); HPO spaces + `balanced`/`Balanced` from `path_a_blocks` |
| Train Ŷ leakage | K=5 OOF on aux-train; **non-aux train Ŷ = mean of K fold models** |
| Arms | D0/D1/T0/T1 (full core); **D1a + O1** (aux-matched oracle); O0/D0a optional |
| Median-fill true CGM | **Forbidden** as primary oracle |
| Metrics | 4-AUC + macro AUPRC + binary + Brier; **calibration required** (sigmoid; isotonic if min_pos≥30) |
| Ablation pass | T1−D1 CI lo>0; oracle headroom/kill on **O1−D1a** |
| B4 blocked by B2? | **No** |

### Critique disposition highlights
- **Accepted:** matched aux oracle baseline (D1a); non-aux Ŷ rule; AUPRC/cal; Stage-1 family pin; user-bar fallback; C1 manifest snapshot.
- **Rejected:** mandatory daily Stage-1 for v1; Path A +1pp block bar as ablation rule; re-clean.

### Next
User go → implement `training/path_b/b2/` per PLAN_B2 run ladder (smoke → s1 → grid → REPORT_B2).

---

## 2026-07-15 — B2 implemented + full grid + freeze

**Plan:** `PLAN_B2.md`. **Package:** `training/path_b/b2/`. **Report:** `REPORT_B2.md`.

### Runs
| Run | Role |
|---|---|
| `b2_smoke_20260715` | pipeline smoke (non-claim) |
| **`b2_grid_20260715`** | claim grid (50×2 Stage-2 trials; Stage-1 30/target) |

### Stage-1
- Val mean R² **0.047**; gate mean/sd/tar R²>0 **pass**
- Test mean R² **0.015** — weak emulator from W0 GREEN

### Stage-2 test (claim)
| Arm | 4-AUC | Binary | Notes |
|---|---:|---:|---|
| D0 | **0.6662** | 0.6889 | **exact** Path A W0 match |
| D1 | **0.7378** | 0.8309 | **exact** Path A C1 match |
| T0 | 0.6706 | 0.6812 | +Ŷ on W0 |
| T1 | **0.7345** | 0.8141 | +Ŷ on C1 |
| D1a | 0.7289 | 0.8298 | aux-matched direct |
| O1 | **0.8227** | 0.8768 | true CGM oracle |

### Decision bars
| Bar | Result |
|---|---|
| T1−D1 ΔAUC | **−0.003** CI [−0.019,+0.011] lo>0 **False** → ablation **fail** |
| T0−D0 ΔAUC | **+0.004** CI [−0.015,+0.024] lo>0 **False** → fail |
| User beat C1 | **Fail** (T1 < freeze) |
| O1−D1a headroom | **+0.094** CI [+0.058,+0.130] → **pass** (Stage-1 bottleneck) |
| Kill pivot | not triggered |

### Locks / freeze
| Topic | Decision |
|---|---|
| B2 predicted two-stage | **Closed null** (and slightly harms binary vs D1) |
| **Any deployable B2 arm beat C1?** | **No.** Best deployable = D1 ≡ C1. T1 loses. |
| Person-CGM oracle ceiling | **Real** (~+9 pp 4-AUC on aux-matched C1); O1 is **not** a deployable C1-beater claim |
| D1/D0 protocol parity | Confirmed exact vs Path A freeze |
| Further B2 HPO / daily Stage-1 / C1→glu Stage-1 | **Frozen** — optional footnotes only; not required to close B2 (`REPORT_B2` §7) |
| Ladder next | **B4 → B3** |

### Impl critique disposition (pre-run)
- Blockers: none. Wired `assert_no_leakage`; smoke min-class vs K-fold guard; Stage-1 GPU fallback error context.

---

## 2026-07-15 — B4 data readiness + plan (pre-critique)

**Plan:** `PLAN_B4.md` (later critiqued → implemented → A+B+hard concluded).

### Data readiness verdict
| Topic | Decision |
|---|---|
| Re-clean for B4? | **No** — `clean/*` + pools + shared windows sufficient |
| Pool / window policy change? | **No** — keep `wearable_core` 1824 / `aux_eligible` 1685 |
| Existing FE enough to train B4? | **No** — need **5-min multi-modal aligned grid** (View B) |
| Grid buildable from current clean? | **Yes** — additive `run_fe` only |
| Concurrent wear | **Dense enough** — probe n=30 aux: median CGM∩HR **~220 h** at 5-min bins; frac CGM bins with HR **~0.95**; all aux span-overlap ≥72h |
| SpO₂ | Optional only (∩aux ~1380); not a clean gap |

### Why FE is required (not optional)
B1/B2 used **daily / person scalar CGM summaries**. B4 headline is **full CGM trajectory** supervision — needs person × 5-min wearable + CGM + masks. Documented missing since `PROCESSED.md` / `FEATURES.md` §8 (b-grid) / B1 data plan.

### Design direction (post-critique locks — see `PLAN_B4.md` §8)
- **B4-A first:** seq2seq multi-task (encoder + masked traj decoder + class head); class on full core; traj on aux concurrent bins only.
- **B4-B second:** rep-distill (same package); not B3 logit-KD.
- **User bar:** deployable **Sλ+C1** vs matched **D1/C1** (0.7378 / 0.8309).
- **FE knobs:** 5-min UTC bins; site-local ToD via `zone`; **T=7d/2016**; channels hr/stress/rr/steps/intensity/asleep + cgm target + tod; no re-clean.

### Critique disposition (accepted blockers)
| ID | Fix |
|---|---|
| B1 subwindow CGM∩HR | **Wear/HR density only** train+infer; CGM only in traj mask |
| B2 C1 fusion | Ambition primary = **GBM Stage-2 (z ∥ C1)**; neural concat diagnostic only |
| B3 D1 pin | Same core test pids; prefer re-fit D1 in `b4/` |
| Pad / short support | Right-pad; `T_min=1008`; report pad frac + drops |
| Traj gate | Pearson **and** beat global train-aux mean RMSE |
| B4-B z_T | Student trains on train `z_T` only |
| CNN vs B1 BiLSTM | **Not** mandatory claim arm (partial reject) |

### Next
**User go** → implement FE `grid_5min` then `training/path_b/b4/` per `PLAN_B4.md` run ladder. **Do not implement until go.**

---

## 2026-07-15 — B4 implementation (pre-smoke; critique tooling failed)

**User go received.** Code landed; **critiquer subagent timed out twice** (5min + 3min). Parent performed plan-lock audit + fixes before any data run.

### Code landed
| Path | Role |
|---|---|
| `pipeline/fe/grid_5min.py` | View-B 5-min grid + `choose_subwindow_start` (wear-only) |
| `pipeline/run_fe.py` / `config.yaml` | block `grid_5min` |
| `training/path_b/b4/` | data/model/train/hybrid/run + config |

### Parent audit (approve smoke)
| Check | Result |
|---|---|
| CGM not in encoder `feature_cols` | **pass** |
| Subwindow API wear-only | **pass** (+ unit density tests) |
| traj_mask = wear ∧ cgm ∧ aux | **pass** |
| Feature leakage scan (grid/person cols) | **pass** (no pool flags) |
| D1 `pid_allow=bundle.pids` for pairing | **pass** (added) |
| Ambition head = GBM(z∥C1) | **pass** (`hybrid.py`) |
| Operator bug stress/rr masks | **fixed** |
| ToD zeroed on non-wear | **fixed** — keep tod_sin/cos always |

### Explicit non-blockers for smoke
- B4-B / O-traj / neural-C1 diagnostic deferred (plan).
- Full FE acceptance (UAB, full-aux concurrent) runs with FE full, not required to start 20-pid smoke.
- Critiquer timeout ≠ plan reject; re-critique after smoke if needed.

### Dual critiquer (smoke-readiness)
| Agent | Model | Result |
|---|---|---|
| critiquer | `grok-cli/grok-4.5:high` | **approve** smoke; no blockers; incomplete-grid footgun (med) |
| critiquer | `opencode-go/glm-5.2:high` | **timeout** (3rd GLM fail this session) |

**Parent disposition:** accept Grok approve + parent lock audit. Fixes before smoke: assert full-core grid coverage; soft shrink only when `max_participants` set; empty-val hard error.

### Smoke results (2026-07-15)

**FE** `run_fe --blocks grid_5min --max-participants 20` (~6s):
- grid **75924 × 18**, 20 pids; person concurrent_hours median **~216 h** (min ~186)
- wear_bin_valid mean ~0.75; cgm ~0.74; subwindow CGM-free unit OK
- Smoke pids all UW/LA (early person_id prefer) — UAB tz not covered in n=20

**Train** `b4_smoke_20260715 --quick --mode neural --max-participants 20` (~13s CPU):
- splits 14/2/3; 0 T_min drops; pad_frac 0; traj_bins train 24k
- λ=0/1.0 pipeline OK; metrics **not claimable** (n tiny; val AUC noise)
- Artifacts: `training/path_b/b4/artifacts/b4_smoke_20260715/`

### FE full acceptance (2026-07-15)

**Command:** `python -m pipeline.run_fe --blocks grid_5min` (~381s, 6 workers).

| Gate | Result |
|---|---|
| Shape | grid **6,883,255 × 18**, person **1824 × 11** (= wearable_core) |
| Leakage scan | **pass** (grid + person) |
| Aux concurrent hours | n=1685; median **210 h**; p10 **172 h**; **min 68.7 h** (median ≥72 **pass**; ~few aux below 72h — mask handles) |
| Wear in 7d subwindow (n=100 sample) | median **1846** bins; min 1389; **100% ≥ T_min 1008** |
| UAB 7025 | zone Chicago; ToD at 17:00 UTC → local **12:00** (correct CDT); concurrent ~207 h |
| `watch_green` | untouched 1824×31 |

### Smoke (prior)
- FE 20 pids + train `b4_smoke_20260715` neural quick — pipeline OK, metrics non-claimable.

### Claim grid `b4_grid_20260715` (ROCm 5600) — B4-A concluded

**Report:** `REPORT_B4.md`.

| Arm | test 4-AUC | test bin | vs bar |
|---|---:|---:|---|
| S0 (λ=0 neural) | 0.646 | 0.642 | floor OK |
| S0.3 / S1.0 | 0.637 / 0.639 | — | Sλ−S0 **null** |
| D1 re-fit | **0.736** | 0.809 | Δ freeze 4-AUC −0.0019 (fair) |
| S0+C1 hybrid | 0.713 | 0.819 | **< D1** |
| S0.3+C1 / S1+C1 | 0.714 / 0.703 | — | **no raise** |

**Bars:** ambition fail; multi-task fail; traj Pearson OK at λ>0 (~0.22–0.26). One pid dropped T_min (7189). Hybrid emb pickle bug fixed mid-run.

**Locks**
| Topic | Decision |
|---|---|
| B4-A deployable beat C1? | **No** |
| B4-A traj multi-task | **Closed null** |
| B4-A hybrid z∥C1 | **Harms vs D1** under this recipe |
| B4-B | **Done** (easy + hard; null) — see later entries |
| Ladder next | **B3 last** |

### Next (historical at B4-A close)
B4-B then B3 — both B4-B easy and hard now concluded.

---

## 2026-07-15 — B4-B rep-distill claim (`b4b_distill_20260715`)

**Report:** `REPORT_B4_B.md`. **Package:** `b4/distill.py`; modes `distill` / `distill_hybrid`.

### Design locks applied
| Topic | Lock |
|---|---|
| Teacher | CGM-AE: X∥cgm → traj MSE only; **no class head in loss** |
| Student | X only; CE + μ MSE(z_S, sg z_T); distill on **train∩aux** only |
| μ | {0, 0.3, 1.0}; μ=0 control |
| Hybrid | z∥C1 GBM vs matched D1 |

### Results (test)
| Arm | 4-AUC | Notes |
|---|---:|---|
| Teacher val traj | RMSE **0.175**, Pearson **~0.99** | privilege works |
| Student μ=0 | **0.646** | = B4-A S0 control |
| Student μ=0.3 | **0.625** | **hurts** vs μ0 CI entirely &lt;0 |
| Student μ=1.0 | **0.636** | null vs μ0 |
| D1 | **0.736** | matched |
| Dμ1+C1 best hybrid | **0.735** | ΔD1 **−0.001** lo≯0 |

**Bars:** distill raise **fail**; ambition **fail**. Cosine alignment **pass** (0.09→0.64).

### Locks / freeze
| Topic | Decision |
|---|---|
| B4-B rep-distill | **Closed null** (can hurt neural AUC) |
| B4 overall (A+B) | **No deployable C1 raise** |
| Further B4 churn | **Frozen** unless new `PLAN_*` |
| Ladder next | **B3 last** |

---

## 2026-07-15 — B4-B hard-teacher sensitivity (H1/H2)

**Plan:** `PLAN_B4_B_HARD.md`. **Report:** `REPORT_B4_B_HARD.md`.  
**Runs:** `b4b_hard_h1_20260715` (`cgm_only`), `b4b_hard_h2_20260715` (`wear_cgm`).

### Why
Easy B4-B teacher (X∥cgm) Pearson ~0.99 may have been “copy CGM.” Hard modes test whether that artifact caused the null.

### Results (test 4-AUC)
| | μ=0 | μ=0.3 | μ=1 | best hybrid | vs D1 0.736 |
|---|---:|---:|---:|---:|---|
| H1 cgm_only | 0.646 | **0.626 hurt** | 0.634 | 0.723 | **fail** |
| H2 wear_cgm | 0.646 | **0.620 hurt** | 0.638 | 0.713 | **fail** |
| Easy (ref) | 0.646 | 0.625 hurt | 0.636 | 0.735 | fail |

H2 teacher val Pearson **~0.30** (hard map real). Distill cosine still ↑ with μ.

### Locks
| Topic | Decision |
|---|---|
| Easy-teacher artifact? | **Not binding** — hard teachers also null |
| B4 LUPI (A+B easy+hard) | **Closed null** for deployable C1 raise |
| CLI | `--teacher-mode {easy,cgm_only,wear_cgm}` |
| Next | **B3** |

