# PLAN_B4_V2 — CRD/RKD distill + gradient-balanced traj MTL + fusion retry

**Date:** 2026-07-15  
**Status:** **CLAIM CONCLUDED 2026-07-16** — honest null; see **`REPORT_B4_V2.md`**. Post-claim critique **approve-with-caveats** (null authentic; freeze stands; F3 residual disclosed). Plan locks remain the protocol of record for those runs.  
**Role:** Sibling retry of **B4 under different formulations**. Does **not** reopen frozen B4-A/B L2/plain-λ recipes.  
**Authority:** `Training.md` §4–5 / §9; frozen B4 reports `REPORT_B4.md`, `REPORT_B4_B.md`, `REPORT_B4_B_HARD.md`; claim report `REPORT_B4_V2.md`; `PLAN_B4.md` (v1 locks for data/FE); `REVIEW_PHASES.md`; `DECISIONS.md`; data `PROCESSED.md` (`grid_5min`).  
**Baseline to beat:** matched **D1** re-fit on surviving `pid_allow` (fair bar). Freeze C1 = test 4-AUC **0.7378** / binary **0.831** informational. S0 ~0.65 informational floor.  
**Outcome:** gates pass; PCGrad / RKD / CRD / hybrid all **fail** vs bars (valid accept-state). Do **not** reopen without a new `PLAN_*`.  
**B3 boundary:** **logit-KD stays B3** (already concluded separately).

---

## 0. Why V2 (not a reopen)

Frozen nulls are nulls of **one recipe per cell**:

| Cell | Frozen recipe | Authentic null of… | V2 formulation |
|---|---|---|---|
| B4-A MTL | plain λ CE + traj MSE | λ-weighted imbalance | **PCGrad / uncertainty weighting** |
| B4-B distill | L2-MSE on raw z | z-reconstruction KD | **RKD primary (+ CRD sensitivity)** |
| Hybrid | frozen z ∥ C1 → GBM | collinear dilution | **OOF-z→GBM first, then learned residual / FiLM** |

Do **not** re-run λ/μ grids at L2 + plain-λ. New run ids only. Frozen `REPORT_B4*.md` stay authoritative for v1.

---

## 1. Data readiness verdict

### Verdict
| Question | Answer |
|---|---|
| Re-run `run_clean`? | **No** |
| Change pools / shared windows / Path A GREEN? | **No** |
| Rebuild `grid_5min` FE? | **No** — full core already accepted (6.88M × 1824; aux median concurrent ~210 h) |
| FE changes needed for V2? | **No for claim.** Augment / fusion masks are **train-time** only |
| Data already sufficient? | **Yes** — proceed to B4-V2 plan |

### Present assets (reuse)
| Asset | Use in V2 |
|---|---|
| `features/grid_5min.parquet` + `_person` | X / CGM / masks (unchanged contract) |
| `pool_masks`, C1 static (watch_green + onboarding + mood) | pools, D1, hybrid |
| `training/path_b/b4/` package | extend (do not fork a parallel package unless code size forces it) |
| Frozen B4-A S0 / teacher H2 quality numbers | control anchors (μ=0 / λ=0 must reproduce ~0.646 test 4-AUC family) |

### Data inspection caveats (surface; do not silently work around)

| ID | Finding | Implication for V2 |
|---|---|---|
| D1 | Aux concurrent min **68.7 h** (1 pid &lt; 72 h); median **210 h**, p10 **172 h** | Keep `traj_sup_valid` mask; no pool redefinition |
| D2 | Train **non-aux** n=**93** only (core non-aux total 139) | CRD memory-bank / negatives are **small** — do not pretend large-n InfoNCE; prefer RKD + in-batch / multi-view pairs |
| D3 | Train insulin **n=80** (train∩aux insulin **69**) | Bundle cheap TS aug with contrastive views; class weights stay inverse-freq |
| D4 | Wear→curve SNR wall: B4-A Pearson ~0.14–0.26; H2 teacher ~**0.30** | **Teacher non-degeneracy + probe gate before reading class Δ** |
| D5 | 47 pids with raw `n_bins` &gt; 8000 (max 16.8k) | Subwindow still CGM-free wear-density (v1 lock); pad/mask unchanged |
| D6 | 77 core pids have total `n_wear_valid` &lt; 2016 (shorter than full 7d span); **zero** pids have total wear count &lt; T_min 1008 — but **subwindow contiguity** can still drop ~1 pid (v1 dropped 7189) | Assert V2 `pid_allow` vs v1 drop set; re-fit D1 on **same** survivors |
| D7 | Per-modality masks already on grid (`hr/stress/rr_bin_valid`) | Available for attention fusion without FE rebuild |
| D8 | Non-aux concurrent hours median **~52 h** (many zero CGM) | Non-aux usable as **wear views / CRD negatives**, never as traj targets |

**Cleaning/FE decision:** no pipeline change. Optional later `PLAN_SSL.md` may add FE-side multi-views; not in V2 claim path.

---

## 2. Goal & claims

### Scientific questions
1. Does **relational / contrastive** representation distill (not L2-z) convert teacher structure into 4-AUC under leakage-safe LUPI scope?  
2. Does **gradient-balanced** traj multi-task (not plain λ) raise S over S0 when recon SNR ≪ class SNR?  
3. Does **OOF embedding → GBM** or **learned residual fusion** avoid the frozen-z∥C1 dilution vs matched D1?

### Claims V2 may make
| Claim | Pass rule |
|---|---|
| **Deployable ambition** | Best V2 hybrid (prefer F1 OOF-z+C1) test 4-AUC **> matched D1** and paired boot CI **lo > 0** |
| **Distill raise** | Student μ>0 (RKD/CRD) test 4-AUC **> μ=0** control, CI lo > 0 |
| **MTL raise** | Gradient-balanced arm test 4-AUC **> S0**, CI lo > 0 |
| **Mechanism** | Distill relational metrics improve with μ (diagnostic, not sufficient) |
| **Teacher honesty** | H2 wear→cgm Pearson + probe suite reported; easy teacher not sole claim |

### Explicit non-claims
- Logit-KD / soft targets (B3)  
- Reopen of L2-μ or plain-λ grids as “V2”  
- Path A number changes  
- SSL as required for V2 pass (deferred; see §6)  
- CORN (out of V2)  
- F4 end-to-end fusion (out of V2)

**Null accept:** if all science bars fail while plumbing/teacher gates pass → **valid V2 close**, recommend B3 and/or `PLAN_SSL.md`.

---

## 3. Design locks

### 3.1 Shared (inherit B4 v1 unless noted)

| Topic | Lock |
|---|---|
| Grid / subwindow | 5-min; T=2016; T_min=1008; **CGM-free** wear-density subwindow; right-pad |
| Encoder | PatchCNN h=64, patch 12 (same as v1) — controlled backbone |
| Class pool | wearable_core (log T_min drops; assert vs v1 `pid_allow`) |
| Traj / distill pool | aux concurrent bins / train∩aux only for privileged loss |
| Class weights | inverse-freq train core, sum-normalize |
| Seed / boot | 42 / n=2000 person paired bootstrap |
| D1 | re-fit C1 features on **same `pid_allow`**; drift vs freeze ≤0.01 or fair-bar = re-fit only |
| Leakage | no CGM / z_T / decoder at deployable infer; val/test z_T never in student weight update or ES on distill loss |
| Device | smoke/probe: local ROCm/CPU OK; full claim: **Lightning L4/L40S preferred** (`COMPUTE.md`) |

### 3.2 Cell A — gradient-balanced traj MTL (B4-A-V2)

**Keep:** same X, traj target (z-scored CGM on `traj_sup_valid`), CE + masked traj MSE, S0 control.

**Change:** replace scalar λ mixing with **task-gradient balancing**:

| Arm | Loss / method | Role |
|---|---|---|
| **S0** | CE only | control (must match v1 S0 family ~0.65 test) |
| **S_pc** | PCGrad on (CE, traj MSE) each step; equal task init | primary MTL retry |
| **S_uw** | Kendall uncertainty weighting (learn σ_ce, σ_traj) | secondary **only if** PCGrad shows conflict |
| **S_λ1** | plain λ=1.0 | **plumbing smoke only** — run id prefix `b4v2_plumbing_`; **never** enter `REPORT_B4_V2` science table |

**Implementation notes:**
- Project conflicting grads (PCGrad) on shared encoder params only; heads keep native grads.  
- Log per-step / per-epoch median cos(g_ce, g_traj) on encoder.  
- ES on val macro 4-AUC (same as v1).  
- Traj non-degeneracy: val Pearson > 0.15 **and** beats mean-predictor RMSE.

**Conflict stop-rule (pre-registered):**  
After S_pc early epochs (1–5), if **median encoder cos(g_ce, g_traj) > 0** (co-aligned, no conflict), **do not run full S_uw claim grid** — log “no PCGrad headroom / no conflict” and treat balancing mechanism as **null** (PCGrad ≈ co-aligned multi-task). Still finish **one** full S_pc train for the controlled contrast vs S0.  
If median cos **&lt; 0**, run **S_uw** as secondary.

**Out:** GradNorm / CAGrad require a **new `PLAN_*`** (do not expand this claim grid).

### 3.3 Cell B — RKD / CRD distill (B4-B-V2)

**Keep:** teacher modes `{easy, cgm_only, wear_cgm}`; student X-only; distill scope **train∩aux**; hybrid after student fit.

**Change:** distill objective:

| Objective | Formula (sketch) | Role |
|---|---|---|
| **RKD** (primary) | μ ( α L_distance + β L_angle ) on **projected** z; mean-normalized pairwise distances + triplet angles; ε-guard | **claim primary** |
| **CRD** (sensitivity) | InfoNCE on projection head; memory bank ≈ train core (~1.2–1.3k); τ=0.07 | secondary if RKD nulls |
| **L2-anchor** | **off by default** | if enabled, also report **anchor-only** μ∈{0.05,0.1} transparency arm (no RKD) |
| **μ=0** | CE only | control |

**Projection heads:** independent student/teacher 2-layer MLPs → dim **128**. RKD/CRD/optional L2-anchor act in this **same** projection space. Distance-only RKD (β=0) = **relational** pairwise L2 on the projection — **not** frozen pointwise raw-z MSE (different objective; no reopen).

**RKD defaults:** α:β = **1:2** (Park); μ ∈ {0, 0.3, 1.0}. Distance-only sensitivity if full RKD null.

**Teacher priority:**
1. **H2 `wear_cgm`** — claim teacher (hard map; prior Pearson ~0.30).  
2. **H1 `cgm_only`** — second.  
3. **Easy `X∥cgm`** — sensitivity only (not sole claim).

**Pre-distill teacher probe (required before student μ grid):**
- Fit H2; report val∩aux traj Pearson / RMSE / beats-mean.  
- **Probes on frozen teacher z** (fit **train∩aux** only; evaluate **val∩aux** only — non-aux have no teacher CGM path):
  1. **Linear** multinomial logistic → macro-OVR 4-AUC  
  2. **Nonlinear** 2-layer MLP (h=64, dropout 0.3) → macro-OVR 4-AUC  
  3. **Relational:** 5-NN accuracy / macro-AUC on teacher z (val∩aux)  
- **STOP distill claim if:**
  - H2 val traj Pearson &lt; **0.15** or fails mean-predictor, **or**
  - **both** linear **and** nonlinear probes val∩aux 4-AUC ≤ **0.55** (chance 0.5 + 0.05) **and** 5-NN ≤ 0.55  
- **GO** if traj non-deg **and** (nonlinear probe **or** 5-NN) **&gt; 0.55**. Linear-only failure with nonlinear/5-NN pass → still GO (RKD may use relational structure a linear probe misses).  
- On STOP: REPORT **teacher bottleneck** (concern #1); no μ×mode student grid; pointer to `PLAN_SSL.md`.

**CRD bank locks (if CRD runs):** bank stores **student** projected features only (never teacher z, never val/test); FIFO; non-aux as negatives only.

**Views / negatives:** same-person + mask-aware TS aug as multi-view; other persons / non-aux as negatives only (no traj). Do not require large CRD bank given D2.

**Augmentation lock:** mask-aware jitter/scale for CRD views; insulin oversample ≤2× in-batch rate. No mixup across labels. No CGM in student aug path.

### 3.4 Cell C — fusion (hybrid ambition)

Frozen v1 hybrid = concat frozen z to C1 → GBM → dilution.

| Stage | Method | Role |
|---|---|---|
| **F0** | D1 re-fit on `pid_allow` | fair bar |
| **F0b** | **z-only GBM** (no C1) on best-val student z vs matched D0/W0-family | **orthogonal-signal probe** |
| **F1** | **OOF student z** (protocol below) ∥ C1 → Path A Stage-2 GBM | primary cheap hybrid |
| **F2** | Single full-train frozen z ∥ C1 GBM (v1 recipe) | dilution control |
| **F3** | Learned residual / FiLM (dropout ≥0.3, rank≤16, ES) | **only if borderline** (trigger below) |
| **F4** | **Dropped from V2** | new plan if ever needed |

#### OOF-z protocol (F1) — leakage locks
| Split | z source |
|---|---|
| **train** | K=5 person-grouped OOF: each train pid’s z from student fit on other 4/5 train pids; distill loss still **train-fold∩aux only** inside each fold |
| **val / test** | z from **one** student fit on **full train** (standard stacking; never fit encoder on val/test labels) |

**Cost control:** full K=5 OOF only for (a) μ=0 control and (b) **one** best-val RKD μ on H2 (val 4-AUC select among μ∈{0.3,1.0}). Other μ get F2-only hybrid footnote unless within 0.005 of selected μ.

**Order:** F0 + F0b → F2 → F1.  
**F3 trigger (tightened):** F1 test 4-AUC **> D1 − 0.01** (borderline/near-tie) **and** F1 ≥ F2.  
Do **not** run F3 when F1 is a clear null far below D1. If F1 ≪ F2, debug OOF plumbing first.

**Overfit kill:** F3 param count ≪ GBM; kill if val Δ&lt;0.005 while train jumps.

### 3.5 Explicitly deferred / out

| Item | Decision |
|---|---|
| **SSL MAE / SimCLR** | **Defer entirely** to `PLAN_SSL.md` (no in-V2 MAE smoke) |
| **CORN class head** | **Out of V2** |
| **Logit-KD** | **B3 only** |
| **Deeper encoder / T / SpO₂ / F4** | Out of V2 |

---

## 4. Decision bars & gates

### 4.1 Plumbing gates
| Gate | Rule |
|---|---|
| G0 overfit50 | CE drops; val 4-AUC ≫ chance on 50 pids (class-only) |
| G1 S0 parity | Full S0 test 4-AUC within ~0.02 of v1 **0.646**; `pid_allow` equals v1 drop set (or log intentional diff) |
| G2 traj non-deg | val Pearson > 0.15 and beats mean |
| G3 teacher stop/go | §3.3 probe suite on **val∩aux** |
| G3b conflict | median cos logged; UW skipped if cos>0 |
| G4 leakage | distill = train∩aux; OOF protocol unit-tested; CRD bank never val/test |
| G5 D1 parity | re-fit D1 vs freeze \|Δ\| ≤ 0.01 or fair-bar note |

### 4.2 Science bars (pre-registered)
| Bar | Pass | Notes |
|---|---|---|
| RKD/CRD μ>0 vs μ=0 | CI lo > 0 on test 4-AUC | |
| PCGrad (UW if run) vs S0 | CI lo > 0 | mechanism-null if no conflict |
| Best hybrid (F1 preferred) vs D1 | CI lo > 0 | ambition; **null expected a priori** |
| F0b z-only vs D0 | report | if z-only ≲ chance+0.05, fusion cannot invent signal |
| F1 vs F2 | report | OOF vs frozen dilution diagnostic |
| Kill distill | G3 STOP | teacher bottleneck REPORT |
| Kill MTL | G2 fail | fix scale/mask first |
| **Null accept** | bars fail, gates pass | **valid V2 close** → B3 and/or `PLAN_SSL.md` |

---

## 5. Run order (smoke → full)

1. **Plumbing smoke** `b4v2_smoke_<date>` — 64 pids; RKD fwd/bwd; PCGrad step; OOF-z shape; no claim metrics. Optional `b4v2_plumbing_λ1` never enters science table.  
2. **Overfit50** `b4v2_overfit50` — class-only.  
3. **Teacher probe** `b4v2_teacher_h2` — H2 + linear/MLP/5-NN on val∩aux + Pearson. **Stop/go**.  
4. **S0 refresh** `b4v2_s0` — control floor + pid_allow assert.  
5. **MTL** `b4v2_mtl` — S0 / S_pc + conflict cos; S_uw only if cos&lt;0.  
6. **Distill** `b4v2_rkd` — μ∈{0,0.3,1.0} on H2 if G3 GO; H1 if H2 GO; distance-only if full RKD null.  
7. **CRD** `b4v2_crd` — only if RKD null **and** G3 GO.  
8. **Hybrid** `b4v2_hybrid` — F0, F0b, F2, F1 (cost-controlled OOF); F3 only on borderline trigger.  
9. **REPORT_B4_V2.md** + `DECISIONS.md` (sibling; never overwrite `REPORT_B4*.md`).

**Compute:** full claim on Lightning L4/L40S preferred; local ROCm 5600 for smoke/probe/overfit. Budget ~**1.5–3×** v1 B4-B wall-clock after cost controls. OOM fallback: batch 4, grad accum.

**Full claim runs require user go.** Smoke + overfit + teacher probe may run to validate plumbing.

---

## 6. SSL decision (concern #3)

SSL on ~2,280 unlabeled grids is the coherent unbuilt lever for cold-start + Stage-1 SNR + wear→curve Pearson. **V2 does not bundle SSL.** If V2 nulls with healthy teacher **and** gradient-balanced MTL also null, REPORT points to `PLAN_SSL.md` — not another KD objective tweak.

---

## 7. Package shape (implementation sketch; not yet coded)

Extend `training/path_b/b4/` surgically:

| Module | Add |
|---|---|
| `losses_rkd.py` / `losses_crd.py` | RKD distance+angle; CRD InfoNCE + bank |
| `pcgrad.py` | PCGrad hook |
| `augment.py` | mask-aware jitter/scale/warp |
| `distill.py` | objective switch `l2|rkd|crd`; keep teacher modes |
| `train.py` | uncertainty weights + PCGrad path |
| `hybrid.py` | OOF-z; z-only GBM; optional FiLM residual |
| `config.yaml` | `distill.objective`, `mtl.balancer`, `fusion.mode`, aug knobs |
| `run.py` | modes: `mtl_bal`, `distill_rkd`, `distill_crd`, `hybrid_oof`, `teacher_probe`, … |

No clean/FE code changes for V2 claim.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Teacher recon geometry orthogonal to severity | Dual probe + 5-NN gate; stop if fail |
| RKD NaNs on near-duplicate pairs | ε on distances; skip zero-norm pairs |
| CRD weak with tiny negative bank | RKD primary; CRD sensitivity only |
| Learned fusion overfit n≈1.8k | F0b + F1 first; tight F3 trigger; F4 out |
| OOF 5× cost explosion | OOF only μ=0 + best RKD μ |
| PCGrad with no conflict | cos stop-rule; skip UW |
| Accidental logit-KD | B3 boundary; representation objectives only |
| S0 drift after refactor | G1 parity + pid_allow assert |

---

## 9. Success criteria (definition of done)

1. Plan critiqued; disposition logged (§11 + `DECISIONS.md`).  
2. Smoke + G0–G1 pass (and G3 if distill run).  
3. Full run ids with tables: MTL, RKD/CRD, hybrid F0/F0b/F1/F2 (F3 if run).  
4. Explicit pass/fail on bars §4.2; teacher probe reported.  
5. `REPORT_B4_V2.md` written; `DECISIONS.md` + `AGENTS.md` status lines updated **without** rewriting frozen B4 reports or Path A numbers.  
6. Null with healthy teacher → recommend B3 and/or `PLAN_SSL.md` — not silent L2 re-tweak. **Null REPORT is a valid accept-state.**

---

## 10. Locks (post-critique)

| # | Choice | Lock |
|---|---|---|
| O1 | Primary distill | **RKD** (distance+angle 1:2); CRD sensitivity |
| O2 | Primary MTL balancer | **PCGrad**; UW only if cos&lt;0 |
| O3 | Hybrid primary | **OOF-z ∥ C1 → GBM (F1)**; F0b first; F3 borderline only; F4 out |
| O4 | Claim teacher | **H2 first**; H1 second; easy last |
| O5 | SSL in V2 | **No** |
| O6 | CORN | **out of V2** |
| O7 | FE/clean | **no change** |
| O8 | Null REPORT | **valid accept-state** |

---

## 11. Critique disposition (2026-07-15)

Critiquer: `opencode-go/glm-5.2:high`, fresh. Verdict: **approve-with-changes**. Parent disposition:

| ID | Severity | Disposition |
|---|---|---|
| BLK-1 probe threshold 0.52 too loose | Blocker | **Accepted** — threshold → **0.55**; dual linear+MLP + 5-NN |
| BLK-2 linear-only understates RKD | Blocker | **Accepted** — nonlinear + 5-NN; GO if relational probe passes |
| BLK-3 OOF protocol / 5× cost underspec | Blocker | **Accepted** — full OOF table; val/test = full-train student; K=5 only μ=0 + best RKD μ |
| BLK-4 no PCGrad conflict stop | Blocker | **Accepted** — cos>0 → skip UW; log mechanism-null |
| H-1 z-only GBM probe | High | **Accepted** — F0b |
| H-2 null is expected accept-state | High | **Accepted** — locked in status + §2 + §4.2 + §9 |
| H-3 F3/F4 too eager | High | **Accepted** — F3 only if F1 &gt; D1−0.01 and F1≥F2; **F4 dropped** |
| H-4 probe eval split | High | **Accepted** — val∩aux only |
| H-5 `REVIEW_PHASES.md` missing | High | **Rejected (false positive)** — file exists at repo root (375 lines); keep as authority |
| M-1 CRD bank rules | Medium | **Accepted** |
| M-2 compute host/budget | Medium | **Accepted** |
| M-3 projection head | Medium | **Accepted** — dim 128 independent MLPs |
| M-4 S_λ1 reopen optics | Medium | **Accepted** — `b4v2_plumbing_` prefix; not in REPORT table |
| M-5 D6 wording | Medium | **Accepted** |
| M-6 distance-only ≠ raw L2 | Medium | **Accepted** |
| N-2 in-V2 MAE smoke | Nit | **Accepted** — removed |
| N-4 CORN in V2 | Nit | **Accepted** — dropped |
| N-3 L2-anchor confound | Nit | **Accepted** — off by default; anchor-only transparency if on |
| N-1/N-5/N-6 | Nit | partially applied (pid_allow assert; Pearson 0.15 kept with stronger probe suite) |

**Do not implement full claim until user go.** Smoke / overfit / teacher probe OK to validate plumbing.

---

## 12. Implementation notes (2026-07-15 post-code-critique)

Code critique (`opencode-go/glm-5.2:high`) → **approve-with-changes**. Parent applied:

| ID | Fix |
|---|---|
| BLK-1 | Teacher GO requires Pearson≥0.15 **and** beats mean-predictor at **best-RMSE epoch** |
| BLK-2 | D1 freeze check fires when n ≈ full core (even if `pid_allow` set); always writes fair-bar note |
| BLK-3 | **OOF F1 is class-only (μ=0)** — arm name `F1_oof_z_C1_mu0`; distill fusion ambition stays **F2** until per-fold RKD OOF |
| H-1 | Best-epoch pearson/beats_mean, not last epoch |
| H-2 | `mtl_bal` defaults balancer to **pcgrad** if none |
| H-3 | `--plumbing` rewrites run_id to `b4v2_plumbing_*`; skips Sλ−S0 compare |
| H-4 | `mtl_bal` default λ grid **{0, 1}** only |
| H-5 | PCGrad combines projected grads by **mean** (LR scale) |
| H-7 | CRD bank enqueues non-aux student projs as negatives |

**Deferred (not blocking smoke):** per-fold RKD OOF; insulin 2× oversample; CRD true multi-view pairs; F3 FiLM.

---

## 13. Post-claim critique (2026-07-16) — freeze hygiene

External critiquer verified artifacts vs `REPORT_B4_V2.md` → **approve-with-caveats**.

| Finding | Disposition |
|---|---|
| Null authentic per cell (PCGrad / RKD / CRD / fusion) | **Accepted** — freeze stands |
| No leakage / OOF / D1 blockers | **Accepted** |
| F3 trigger fired (F1 &gt; D1−0.01 ∧ F1≥F2) but F3 not run | **Disclose residual** — not a silent win; same-budget still null |
| Per-fold RKD-μ OOF deferred | **Already labeled** in report; does not un-freeze |
| Reopen V2? | **No** — residual → `PLAN_SSL.md` |
