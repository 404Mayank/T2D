# Path B4 — seq2seq CGM-trajectory teacher + rep-distill (headline)

**Date:** 2026-07-15  
**Status:** **B4 A+B CONCLUDED** 2026-07-15 — A: `b4_grid_20260715` (`REPORT_B4.md`); B: `b4b_distill_20260715` (`REPORT_B4_B.md`); hard teachers `REPORT_B4_B_HARD.md`. Sibling **B4-V2 also null** 2026-07-16 (`REPORT_B4_V2.md`). B3 separately concluded.  
**Role:** Path B **headline candidate** — full CGM trajectory supervision / representation LUPI.  
**Authority:** `Training.md` §4 B4 / §7; Path A freeze `REPORT_A_WRAP.md`; B1/B2 freezes `REPORT_B1.md` / `REPORT_B2.md` + `DECISIONS.md`; data `PROCESSED.md` / `CLEANING.md` / `FEATURES.md` §8 view (b-grid).  
**User ambition bar:** beat Path A **C1** (test 4-AUC **0.7378** / binary **0.8309** / macro AUPRC **0.4687**).  
**Ladder:** B1 frozen → B2 frozen → **B4 (this)** → B3 last. Do not reopen B1/B2 claim grids.

---

## 0. Data readiness verdict

### Verdict
| Question | Answer |
|---|---|
| Re-run `run_clean`? | **No** |
| Change cleaning policy / pools / shared windows? | **No** |
| Existing FE enough to *train* B4? | **No** — daily matrices (B1/B2) are scalar summaries; B4 needs **5-min multi-modal aligned grid** |
| Can grid be built from current `clean/*`? | **Yes** — additive FE only (`run_fe` View B) |
| Concurrent wear dense enough? | **Yes** (session probe below) |

### Present and usable (no re-clean)

| Asset | Status | B4 use |
|---|---|---|
| `clean/cgm.parquet` | OK — 5.06M EGV, 1924 pids, median gap **5 min**, bg∈[40,400] | trajectory **target** (train/aux only) |
| `clean/heart_rate.parquet` | OK — ~20.7M, 1-min, 1999 pids | primary wearable channel |
| `clean/stress.parquet` | OK — ~20.6M, 1-min | channel |
| `clean/respiratory_rate.parquet` | OK — ~22.8M, denser than HR | channel (aux already requires RR) |
| `clean/physical_activity.parquet` | OK — interval steps/intensity | expand → 5-min steps / intensity |
| `clean/sleep.parquet` | OK — stage intervals | expand → 5-min asleep / stage code (optional) |
| `clean/oxygen_saturation.parquet` | OK but sparse — 1617 pids, median ~1879 pts; ∩aux **1380** | **optional** channel; not required for v1 |
| `meta/shared_windows.parquet` | OK — HR-anchored window + **site `zone`** | window bounds + local-time re-derive |
| `meta/pool_masks.parquet` | OK — core 1824 / aux 1685; splits fixed | pools, labels, split |
| `features/watch_green`, onboarding, mood | OK | static late-fuse for deployable arms (C1 stack) |
| `features/watch_daily`, `cgm_daily`, `cgm_person` | OK (post sleep-unit fix) | diagnostics / optional hybrid; **not** B4 primary X/Y |

### Missing (FE only)

| Asset | Why |
|---|---|
| `features/grid_5min.parquet` (or sharded equivalent) | person × 5-min bin multi-channel wearable + CGM + masks |
| Optional: window index / quality table | concurrent density, chosen subwindow, pad length |

### Empirical concurrent density (session probe, n=30 aux, seed 0)

Floor timestamps to **UTC 5-min bins**; join CGM mean ∩ HR mean:

| Stat | median | p10-ish / range |
|---|---:|---|
| both_hours (CGM∩HR 5-min bins) | **~220 h** | all sampled ≥ **~153 h** |
| frac CGM bins with HR | **~0.95** | min ~0.64 |
| frac HR bins with CGM | **~0.91** | — |
| span-overlap `cgm_hr_overlap_hours` (pool_masks, all aux) | median **238 h**; min **152 h** | all aux pass 24h **and** 72h flags |

**Implication:** span-overlap gate already used for `aux_eligible` is **not** lying about concurrency for B4 — minute-level concurrent hours are large. Still: emit **concurrent** quality columns on the grid (do not trust span alone for masking).

### Cleaning policy judgment (locked for B4 plan)

1. **Do not re-clean.** UTC instants + HR shared window are correct; residual `*_local` parquet single-tz (LA) is already handled in Path B FE via `zone` (`local_time.py`).
2. **Do not change** `wearable_core` / `aux_eligible` definitions for B4 claim pools (same as B1/B2).
3. **Do not** re-FE Path A `watch_green` (Path A freeze).
4. Tighten **at FE time** only: concurrent masks, valid-bin gates, subwindow selection for fixed-length tensors.
5. SpO₂ stays optional; missingness is structural (~15–18% of core lack SpO₂) — not a clean bug.

### Conclusion
**Pipeline cleaning is ready. B4 is blocked only on View-B FE + training package** — same pattern as B1’s daily FE, not a Stage-2/3 clean reopen.

---

## 1. Goal

### Scientific question
Does supervising a wearable encoder with the **full CGM trajectory** (seq2seq / reconstruction of the glucose curve), and/or distilling a **glucose-shaped representation**, improve 4-class T2D discrimination over matched Path A baselines — especially over **C1** for the deployable stack?

### Why this cell (after B1/B2)
| Stage | Cell | Result |
|---|---|---|
| B1 | day-level multi-task on **8 CGM summaries** | pure-seq ~0.65; multi-task **null** |
| B2 | person **point-estimate** Ŷ CGM two-stage | T1 **null / slightly harmful** vs D1; **oracle +0.09** |
| **B4** | **trajectory / rep** glucose structure | untested; motivated by B2 oracle headroom + B1/B2 summary-handoff nulls |

B1+B2 converge: **scalar CGM summaries as handoff/multi-task targets do not raise deployable AUC**. Oracle shows privilege is real → representation must be richer than daymean 8-vector.

### Claims B4 may make
1. **Primary (ambition):** deployable student (wearable encoder ± C1 static, **no CGM at infer**) beats **matched C1 / D1** on test 4-AUC with paired boot CI lo > 0.  
2. **Scientific watch-only:** trajectory-supervised encoder (λ>0 or distill) beats **matched watch-only λ=0 / W0-family** control.  
3. **Trajectory quality:** CGM reconstruction metrics on val/test aux (not the T2D claim alone).  
4. **Ablation:** trajectory multi-task vs rep-distill vs class-only encoder (controlled backbone).

### Claims B4 is not
- B3 logit-KD (Diasense) — run **last**, separate package.  
- Reopen of B1 λ grids or B2 Stage-1 HPO.  
- Oracle (true CGM at Stage-2 / teacher logits at infer) as a deployable raise.  
- Path A number changes without re-run.

---

## 2. Design locks

### 2.1 Headline formulation (two complementary arms; one package)

**B4-A — Seq2seq multi-task (primary first pass)**  
```
Wearable 5-min grid X → Encoder → z
                      ├─ Decoder → ŷ_cgm(t)   (aux bins only; masked MSE / Huber)
                      └─ Class head → p(y)    (all wearable_core)
L = L_CE(y) + λ L_traj(cgm, ŷ_cgm; mask)
```
Inference: encoder + class head (+ optional static fuse); **drop decoder**.

**B4-B — Representation distillation (novelty delta; run if B4-A λ=0 is learnable)**  
```
Teacher (train only): sees wearable X **and/or** true CGM grid → z_T (glucose-shaped)
Student (deployable): X only → z_S
L = L_CE_student + μ || stopgrad(z_T) − z_S ||²  (+ optional small traj term on student)
```
Teacher **never** used at inference. Prefer **OOF or split-safe teacher** if teacher has a T2D head (avoid student fitting teacher’s in-sample class leakage). For pure CGM-autoencoder teacher (no class head), full train-aux fit is OK.

**v1 order:** implement **B4-A** end-to-end (FE → smoke → λ grid) first; **B4-B** is the same package’s second experiment, not a separate ladder stage. Do not skip to B3.

### 2.2 Pools (verified; same as B1/B2)

| Pool | train | val | test | train insulin |
|---|---:|---:|---:|---:|
| wearable_core | 1277 | 270 | 277 | 80 |
| aux_eligible (⊆ core) | 1184 | 247 | 254 | 69 |

| Head | Fit / loss mask |
|---|---|
| T2D class | all `wearable_core` (do **not** restrict T2D to CGM-haves) |
| Trajectory | bins with `cgm_bin_valid` on `aux_eligible` only |
| Rep-distill teacher | train ∩ aux (and val only for teacher ES if teacher trained) |

### 2.3 View-B FE locks (new; additive)

#### Grid definition
| Knob | Lock |
|---|---|
| Bin size | **5 min** (CGM native; HR/stress/RR aggregate into bin) |
| Time base | **UTC `timestamp` instants** floored to 5-min; **site-local civil features** via `zone` (`to_site_local`) — same rule as B1 daily FE. Do **not** use parquet `*_local` wall clock for UAB. |
| Window | Clean series already HR-windowed. Grid spans each pid’s cleaned support ∩ shared window. |
| Subwindow for train **and** infer tensors | **CGM-free only** (LUPI / train-serve parity). Prefer contiguous block of length `T_bins` maximizing **`wear_bin_valid` density** (HR proxy); tie-break: earliest start. CGM must **never** enter window positioning — only `traj_sup_valid` masking for L_traj. Same algorithm for aux and non-aux, train and test. |
| `T_bins` default | **7d = 2016** bins |
| Short support | If wear-valid support &lt; `T_bins`: take **all** wear-spanned bins, **right-pad** to `T_bins` with `pad_mask=1`. |
| Min effective length | Drop pid from sequence train if wear-valid bins in chosen window **&lt; `T_min`** (default **1008 = 3.5d**). Report n dropped per split; do not silently shrink claim pool without logging. |
| Pad / mask | **Right-pad** fixed; `pad_mask` bool; never treat pad as observed; CNN must respect mask (no loss / no attention on pad). Report pad-fraction distribution (p50/p90) in FE/train acceptance. |

#### Channels (v1 primary)
| Channel | Source | Bin aggregate |
|---|---|---|
| `hr` | heart_rate | mean of 1-min samples in bin |
| `stress` | stress | mean valid stress |
| `rr` | respiratory_rate | mean |
| `steps` | physical_activity | sum steps overlapping bin |
| `intensity` | physical_activity | max or time-weighted intensity tier |
| `asleep` | sleep | fraction of bin in non-awake stages (0–1) |
| `cgm` | cgm | mean EGV (target / teacher only) |
| `tod_sin`, `tod_cos` | local hour | cyclic time-of-day (always observed when bin exists) |

**Deferred v1:** SpO₂ (coverage hole), sleep stage one-hots, calorie.

#### Validity masks (required columns)
| Mask | Rule (defaults; config) |
|---|---|
| `hr_bin_valid` | ≥ `min_hr_samples_per_bin` (default **2** of ~5 expected 1-min) |
| `cgm_bin_valid` | ≥ 1 EGV in bin (CGM is ~1/bin when present) |
| `wear_bin_valid` | `hr_bin_valid` (primary wear proxy) |
| `traj_sup_valid` | `cgm_bin_valid ∧ wear_bin_valid ∧ aux_eligible` — **only** these bins enter L_traj |

#### Output artifacts
| File | Grain | Notes |
|---|---|---|
| `features/grid_5min.parquet` **or** `features/grid_5min/{pid}.parquet` + `features/grid_5min_index.parquet` | person × bin | Prefer **sharded** if single-file write is painful; consumer contract: index has `person_id`, `n_bins`, `t0`, quality stats |
| `features/grid_5min_person.parquet` | person | concurrent hours, frac, chosen subwindow start/end, n_valid bins — **no labels** |

**Storage estimate:** dense float32 core × 2016 × ~8 ch ≈ **~0.1 GB** (+ masks); fine for local disk. Build streaming per pid (row-group).

#### FE acceptance gates (must pass before train claim)
1. UAB pid smoke: site-local `tod` / day keys match Chicago convert (reuse B1 UAB check style; pid **7025** if still present).  
2. **Full aux** (not n=30 probe): median concurrent wear∩cgm hours on grid **≥ 72**; report min/p10/p50.  
3. Core: after CGM-free subwindow, median wear-valid bins **≥ 0.5 × 2016**; report **pad-fraction** p50/p90 and n with wear-valid &lt; `T_min`.  
4. Subwindow algorithm unit test: aux vs non-aux selection **identical** given same wear mask (CGM column zeroed must not change chosen start).  
5. No `label` / `recommended_split` / `clinical_site` in feature files.  
6. `watch_green` / Path A artifacts **untouched**.

### 2.4 Model locks (B4-A v1)

| Topic | Lock |
|---|---|
| Encoder | **1D CNN patch encoder** primary (kernel/stride patches over 5-min grid) → optional BiLSTM/Transformer **layer on patches** if CNN-only underfits. Cold full-length BiLSTM on 2k steps is **not** default (B1 short-T lessons + hourly-fail prior). |
| Hidden | start **h=64** (match B1 scale); h=128 sensitivity only after smoke. |
| Decoder | lightweight CNN/MLP **per-bin** head from encoder states (aligned) or small causal/non-causal conv decoder; predict **z-scored CGM** (train-aux stats). |
| Class head | attention or mean-pool over non-pad encoder states → Linear(4). |
| Static fuse (deployable) — **LOCKED primary** | **Hybrid Stage-2 GBM (Path A family):** freeze encoder → person embedding `z` (pooled, train-only scale if needed) **concat** Path A C1 static features → **CatBoost + LightGBM** multiclass, same HPO/class-weight/val-select protocol as B2 Stage-2 / `path_a_blocks`. This is the **ambition-bar combiner** (fair shot vs D1, which is the same family). |
| Static fuse — neural concat | **Secondary diagnostic only:** concat `z`+C1 → MLP class head (end-to-end or frozen-z). Report if run; **not** the user-bar primary (B1 GREEN neural fuse was null; GBM is the Path A-matched head). |
| Watch-only neural arms (S0/Sλ) | Neural class head on `z` only (no C1); controlled multi-task contrast. |
| Class weights | inverse-freq on **train core**, sum-normalize (B1 lock). |
| Feature scale | train-only per-channel z-score on **observed** wear bins; CGM target z-score on traj-valid bins. |
| λ grid (B4-A) | `{0, 0.3, 1.0}` primary; optional 0.5 if compute allows. Science table = all λ, not best-of alone. |
| Seeds | 42; ES on val macro 4-AUC; plateau LR optional. |
| Device | CUDA host for full grid (`COMPUTE.md`); local ROCm/CPU smoke only. |

### 2.5 Arms (evaluation matrix)

| Arm | Pool | Inputs at infer | Train supervision | Head | Role |
|---|---|---|---|---|---|
| **S0** | core | grid wear only | class only (λ=0) | neural on `z` | sequence floor |
| **Sλ** | core | grid wear only | class + traj (λ>0) | neural on `z` | multi-task raise? |
| **S0+C1** | core | `z` + C1 static | class only (λ=0 encoder) | **GBM Stage-2** | deployable hybrid, no traj |
| **Sλ+C1** | core | `z` + C1 static | class + traj encoder | **GBM Stage-2** | **user ambition primary** |
| **D1** | **same core 1824 / same split** | Path A C1 tabular only | — | **GBM re-fit in B4 package** (or B2 D1 scores if bit-identical protocol) | matched baseline |
| **W0** | core | Path A GREEN | — | GBM | informational watch tabular floor |
| **O-traj** (optional) | aux | true CGM grid | privileged | neural / teacher | ceiling only; **not deployable** |

**D1 pin (blocker fix):** D1 is always evaluated on the **same test person_ids** as S\*+C1 (wearable_core test n=277). Prefer **re-fit D1 inside `b4/`** with Path A C1 feature manifest + B2/path_a_blocks Stage-2 protocol so pairing is trivial; if reusing `b2_grid_20260715` D1 probabilities, assert pid set equality and sort-join on `person_id` (not row order alone). Freeze match: D1 test 4-AUC within **0.01** of 0.7378 else B2-style fallback (fair bar = vs re-fit D1 only).

**Primary ambition contrast:** **Sλ+C1 − D1** (GBM heads; paired boot on person).  
**Scientific multi-task contrast:** **Sλ − S0** (neural heads; paired boot).  
**Hybrid without traj:** **S0+C1 − D1** — if this wins and Sλ doesn't add, headline is **encoder embedding + GBM**, not traj LUPI.  
**B1 comparator:** frozen B1 pure-seq test **0.652** is **informational** only (daily BiLSTM ≠ 5-min CNN). No mandatory B1 re-run; if S0 val ≪ 0.60 after overfit gate, first debug FE/scale/CNN — optional daily-BiLSTM sanity is **footnote**, not a claim arm.

### 2.6 Metrics

| Metric | Use |
|---|---|
| Macro-OVR **4-AUC** | primary rank |
| Macro **AUPRC** | co-required |
| Binary AUC (`1−P0`) + binary AUPRC | screening |
| Per-class OVR AUC | diagnostic (insulin) |
| Brier + val **sigmoid** cal (isotonic if min_pos≥30) | required co-report; **ranking claim = raw** |
| Traj: masked RMSE / MAE / Pearson on z-CGM and mg/dL | aux only |
| Paired person bootstrap Δ (n=2000, seed 42) | Sλ−S0, Sλ+C1−D1, S0+C1−D1 |

### 2.7 Decision bars

| Question | Pass rule |
|---|---|
| **User ambition: beat C1** | **Sλ+C1** test 4-AUC **> D1** and paired boot CI **lo > 0**. External anchor frozen C1 0.7378. **Fallback (B2-style):** if D1 &lt; freeze − 0.01, fair bar is Sλ+C1 vs D1 only; do not claim “beats unreproduced freeze” alone. |
| Trajectory multi-task helps (watch) | Sλ−S0 CI lo > 0 on test 4-AUC. |
| Trajectory multi-task helps (deployable) | Sλ+C1 − S0+C1 CI lo > 0. |
| Sequence + C1 without traj beats C1 | S0+C1 − D1 CI lo > 0 — if yes but Sλ doesn’t add, headline is **encoder+static**, not traj LUPI; report honestly. |
| Traj head non-degenerate | On val **traj_sup_valid** bins: (1) masked Pearson(cgm, ŷ) **> 0.15**, **and** (2) masked RMSE **&lt;** RMSE of **global train-aux mean predictor** (constant = mean of train traj-valid CGM in **z-score space**). Both required. If fail → fix FE/scale/decoder before reading λ science. |
| Kill / pivot | If S0 val 4-AUC ≲ 0.55 after FE accept + overfit gate → debug encoder/data before λ grid (budget: fix scale/mask/T_bins/patch before architecture thrash). If traj quality OK but all Δ CI include 0 and Sλ+C1 ≯ D1 → **negative headline** allowed; still write REPORT; proceed to B4-B once, then B3. |
| B4-B gate | Run rep-distill if (a) S0 learnable (val ≳ 0.60) **and** (b) either traj multi-task weak or we need novelty arm for paper — **not** blocked on Sλ win. |
| B4-B z_T scope | Even for CGM-AE teacher (no class head): teacher may fit on **train∩aux** CGM; distillation targets `z_T` for student training use **train** persons only. Val/test `z_T` only for **metrics / optional analysis**, never for student weight updates or ES on distill loss. |

**B4 does not block B3** regardless of pass/fail; B3 remains last baseline.

### 2.8 Leakage rules
- Never CGM / teacher z / traj decoder outputs as **inference** features for deployable arms.  
- No site, label, split, pool flags in X.  
- No `n_valid_days`-style duration leakage as **predictor** features (quality columns may exist in `grid_5min_person` for QA; train loaders must not feed them as X).  
- **Subwindow selection is CGM-free** (wear density only) — CGM must not choose the tensor window.  
- Teacher with class head → **OOF teacher z on train** (K-fold on train-aux). CGM-AE teacher → train∩aux fit OK; student only trains on train `z_T` (§2.7).  
- Val/test person isolation: fixed `recommended_split` only.  
- Deployable S\*+C1: at infer, encoder(X_wear) → `z` → GBM(C1, `z`); no CGM path.

---

## 3. Package & run ladder

### 3.1 FE (pipeline)
```
pipeline/fe/grid_5min.py      # NEW
pipeline/run_fe.py            # wire block: grid_5min
pipeline/config.yaml          # features.grid_5min.* knobs
```
No changes to clean stages. Update `PROCESSED.md` consumer contract when FE lands.

### 3.2 Train package
```
training/path_b/b4/
  config.yaml
  data.py          # load grid, subwindow, scale, static C1, masks
  model.py         # encoder / decoder / class / optional distill
  train.py
  evaluate.py      # metrics + bootstrap Δ + traj metrics
  run.py / __main__.py
  artifacts/<run_id>/
```

### 3.3 Run ladder
1. **FE smoke** — 20 pids; schema + UAB tz check.  
2. **FE full** — `run_fe --blocks grid_5min`; acceptance §2.3.  
3. **Train smoke** `b4_smoke_<date>` — ~64–128 pids, 3–5 ep, λ∈{0,1}; pipeline only.  
4. **Overfit gate** `b4_overfit50` — 50 train pids, class-only; CE must drop; val AUC ≫ chance.  
5. **S0 full** `b4_s0_<date>` — class-only core (neural); establish sequence floor.  
6. **λ grid (watch)** `b4_grid_<date>` — λ∈{0,0.3,1.0} neural S0/Sλ; traj metrics + Sλ−S0 boot.  
7. **Ambition hybrid** `b4_c1_<date>` — extract `z` from best-val S0 and Sλ encoders → GBM Stage-2 on (z ∥ C1) = S0+C1 / Sλ+C1; **re-fit D1** same package; paired boots Sλ+C1−D1, S0+C1−D1, Sλ+C1−S0+C1.  
8. Optional **B4-B** `b4_distill_<date>` after §2.7 gate.  
9. `REPORT_B4.md` + `DECISIONS.md` + `path_b/README.md` + `AGENTS.md` status lines.

### 3.4 Compute placement
| Step | Where |
|---|---|
| FE grid build | local or Lightning CPU |
| Smoke / overfit | local ROCm/CPU or L4 |
| Full B4 grid | Lightning **L4/L40S/A100** (`COMPUTE.md`) |

---

## 4. Out of scope (B4 v1)

- Re-clean / pool redefinition / Path A GREEN re-FE  
- B1/B2 claim re-runs  
- B3 logit-KD implementation (next stage)  
- SSL / MOMENT (gated later; orthogonal)  
- SpO₂/ODI as primary channels  
- Cold 1-min full-resolution CNN (hourly negative prior; 5-min is the v1 resolution)  
- Claiming O-traj / teacher as deployable  
- Diet block / CORN (Path A leftovers)

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Long T (2k) RNN fails / OOMs | Patch CNN default; T=7d; gradient ckpt; batch size sweep |
| Traj loss dominates class | λ grid + loss scaling by mask count; optional uncertainty weighting later |
| S0 weak (seq ≪ tabular) | Debug FE/scale first; ambition path is already **z→GBM + C1** (primary); neural-only is not required to beat C1 |
| Concurrent holes | masks; do not impute CGM into L_traj; wear impute = train channel median on observed only |
| Teacher class leakage (B4-B) | OOF teacher or CGM-only teacher |
| Ambition bar too hard | Pre-register null + traj quality as publishable; oracle/O-traj shows ceiling |
| Scope creep into B3 | Logit-KD explicitly out; rep-distill ≠ soft-logit KD |

---

## 6. Success criteria (definition of done)

1. FE accepted (§2.3) and documented in `PROCESSED.md` / `DECISIONS.md`.  
2. Smoke + overfit gates pass.  
3. Full grid run id(s) with arm table, traj metrics, bootstrap Δs.  
4. Explicit **pass/fail** on user ambition (vs D1/C1) and multi-task (Sλ−S0).  
5. `REPORT_B4.md` written; ladder status updated; B3 still marked last.  
6. No Path A number rewritten.

---

## 7. Open choices (post-critique locks)

| # | Choice | Lock |
|---|---|---|
| O1 | Tensor length | **7d / 2016 bins**; 10d sensitivity later only if S0 solid |
| O2 | Encoder | **CNN patch** first; recurrent/transformer only if S0 val &lt; ~0.60 after overfit + FE debug |
| O3 | C1 ambition head | **GBM Stage-2 on (z ∥ C1)** primary; neural concat diagnostic only |
| O4 | B4-B teacher | **CGM-AE / encoder on (X, cgm)** without class head first |
| O5 | Hybrid z→GBM | **Promoted from footnote to primary ambition path** (was optional) |
| O6 | Grid storage | Implementer choice; **index + shards OK** if documented |
| O7 | Subwindow | **Wear/HR density only** (never CGM∩HR) |
| O8 | D1 | Re-fit in B4 package preferred; pid-aligned if reuse B2 |
| O9 | B1 CNN vs BiLSTM | Informational B1 freeze only; no mandatory re-run |
| O10 | Compute fallback | If OOM: cut batch → patch stride → `T_bins=1440` (5d) before dropping channels |

---

## 8. Critique disposition

**Source:** fresh `critiquer` (`opencode-go/glm-5.2:high`) on pre-critique `PLAN_B4.md`. Verdict was **revise**. Parent merge below.

| Finding | Severity | Disposition |
|---|---|---|
| **B1** Subwindow max CGM∩HR density → train/serve skew + privileged window choice | **Blocker** | **Accept** → subwindow **wear/HR density only** for all pids train+infer; CGM only in `traj_sup_valid` |
| **B2** C1 fusion undefined (neural concat vs GBM) under-defines ambition bar | **Blocker** | **Accept** → primary **GBM Stage-2 (z ∥ C1)**; neural concat secondary diagnostic only |
| **B3** D1 pool/split / score reuse not pinned for paired boot | **High** | **Accept** → same core test pids; prefer re-fit D1 in `b4/`; assert pid join if reuse B2 |
| **B4** No CNN vs B1 BiLSTM architecture ablation | **Med** | **Partial reject** — Sλ−S0 isolates traj on **fixed** B4 encoder (correct cell). B1 0.652 stays informational. Mandatory BiLSTM re-run is scope creep; optional footnote only if S0 fails debug |
| Short blocks &lt; T_bins / pad policy missing | **High** | **Accept** → right-pad; `T_min=1008`; report pad fraction + n dropped |
| Traj non-degeneracy ambiguous (predict-mean / Pearson alone) | **Med** | **Accept** → Pearson **and** beat global train-aux mean RMSE in z-space |
| B4-B z_T on val/test scope | **Med** | **Accept** → student trains on train `z_T` only; val/test z_T metrics-only |
| n=30 concurrent probe vs full aux min &lt; 7d | **Med** | **Accept** via pad + T_min; FE acceptance still reports concurrent hours on **full** aux |
| S0≥0.60 assumption / no iteration budget | **Med** | **Accept note** → debug FE/scale/mask/T before arch thrash; kill branch explicit |
| Neural late-fuse may not beat GBM D1 | **Med** | **Accept** via B2 disposition (GBM primary) |
| Compute budget unstated | **Low** | **Accept** → O10 OOM fallback ladder |
| Solid: no-re-clean, LUPI pools, D1 drift fallback, S0+C1 honesty check, B3 last | — | **Keep** |

---

## 9. Doc updates when implementing (checklist)

- [ ] `PROCESSED.md` — grid layout + consumer sketch  
- [ ] `CLEANING.md` — note View B FE exists (still out of clean scope)  
- [ ] `FEATURES.md` §8 — (b-grid) status → built  
- [ ] `training/path_b/DECISIONS.md` — readiness + locks  
- [ ] `training/path_b/README.md` — B4 status  
- [ ] `AGENTS.md` / root `README.md` — ladder line  
- [ ] `REPORT_B4.md` after runs  

---

## 10. One-page summary

**No re-clean.** Build **5-min multi-modal grid FE** from existing `clean/*` (concurrent density is strong). **Subwindows are wear/HR-density only** (never CGM-positioned). Train **B4-A** encoder + masked traj decoder; watch multi-task = neural **Sλ vs S0**. Ambition arm = frozen **`z` + C1 → Path A–family GBM** (**Sλ+C1**) vs **matched re-fit D1**. Bar: bootstrap CI lo>0. If sequence floor is solid, add **B4-B rep-distill**. Then B3. Headline only if ambition or clear LUPI Δ lands; otherwise honest negative + trajectory diagnostics still close the cell.
