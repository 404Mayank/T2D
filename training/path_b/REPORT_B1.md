# B1 final report — controlled multi-task + post-fix freeze

**Status:** **B1 CONCLUDED** (2026-07-15).  
**Protocol:** `PLAN_B1_TRAIN.md` / `PLAN_B1_IMPL.md` / `PLAN_B1_FIX.md`  
**Authority for claims:** this report + `DECISIONS.md` (2026-07-14 audit + 2026-07-15 fix/fuse/freeze).

Path A numbers are **frozen** and **unchanged** by B1. Watch floor W0 **0.666** / bin **0.689**; deployable C1 **0.738** / **0.831** (`training/path_a_blocks/REPORT_A_WRAP.md`).

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Did day-level multi-task CGM help class AUC? | **No** (clean null after bugfix) |
| Is the daily sequence backbone learnable? | **Yes** after C1+C2 FE/scale fixes |
| Does pure seq beat Path A watch floor? | **No** (test 0.652 vs 0.666) |
| Does GREEN late-fuse close the gap? | **No** (test 0.638; no raise vs pure seq) |
| Proceed Path B ladder? | **Yes → B2 → B4 → B3** |

**Scientific takeaway:** Pre-fix near-chance B1 was **broken inputs**, not proof multi-task fails. After repair, class-only daily BiLSTM is a **real ~0.65 floor**, multi-task still **null**, and person GREEN late-fusion **does not** turn the spine into a CatBoost-beater. Path B value moves to **B2/B4** (privileged CGM structure), not more B1 λ grids.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Backbone | `attn_lstm_64` (BiLSTM + mask attention), h=64 |
| Glu head | day-level Linear → 8 CGM stats (masked MSE) |
| Features | 18 daily watch dims (no coverage counts) |
| Optional fuse | person `watch_green` → concat to `z` before class head only |
| Pool | wearable_core n=1824; glu only aux_eligible ∧ watch_valid ∧ cgm_valid |
| Split | train/val/test **1277 / 270 / 277** |
| ES | val macro-OVR AUC, patience 15; seed 42 per λ |
| Class weights | inverse-freq train, sum-normalize (CE only) |
| Device | AMD RX 5600 ROCm; `cudnn.enabled=False` |

---

## 2. Timeline of runs

| Run id | Role | Valid as claim? |
|---|---|---|
| `b1_grid_20260714` | Full λ pre-audit | **No** — broken sleep FE + no input scale |
| `b1_overfit50_fix` | G1 overfit gate post C1+C2 | diagnostic |
| `b1_fix_20260715` | G2 λ=0 full post-fix | yes (class-only floor) |
| `b1_grid_20260715_fix` | G3 λ∈{0,0.5} multi-task | **yes — multi-task claim** |
| `b1_green_20260715` | G4b GREEN late-fuse λ=0 | yes (fusion confirmation) |

---

## 3. Root-cause fix (required before any claim)

Full audit: `AUDIT_B1_UNDERPERF.md`.

| ID | Fix | Evidence |
|---|---|---|
| **C1** | `_sleep_daily` duration/gaps via `.dt.total_seconds()`; `sleep_n_bouts` = **sessions**/onset day | sleep mean **6.64 h** (was ~1e-5); coverage **77%** watch_valid (was ~8%); days/pid **9** (was 1) |
| **C2** | train-only feature z-score; observed-only stats for non-fill_zero; sleep_duration not fill-zero | post-scale train std ratio ~1.1 (was ~1e8) |
| **C3** | emergent: backbone learns | G1 CE 1.38→0.79 on 50 pids |

`sleep_n_bouts` = sessions is a **definition change** vs early PLAN_B1_DATA text (stage-row counts); blessed in DECISIONS 2026-07-15.

---

## 4. Results

### 4.1 Pre-fix grid (historical only) — `b1_grid_20260714`

| λ | val 4-AUC | test 4-AUC | test bin | Δ vs λ0 CI lo>0 |
|---:|---:|---:|---:|---|
| 0.0 | 0.564 | 0.510 | 0.523 | — |
| 0.3 | 0.521 | 0.540 | 0.558 | No |
| 0.5 | 0.518 | 0.544 | 0.578 | No |
| 1.0 | 0.510 | 0.504 | 0.566 | No |

CE flat ~1.40→1.38. **Do not cite as multi-task ceiling.**

### 4.2 Post-fix class-only + multi-task — `b1_grid_20260715_fix`

| λ | best ep | val 4-AUC | test 4-AUC | test binary | test glu MSE (z) |
|---:|---:|---:|---:|---:|---:|
| **0.0** | 1 | **0.680** | **0.652** | **0.679** | ~1.51 |
| **0.5** | 1 | 0.678 | **0.652** | 0.680 | ~1.43 |

Paired bootstrap ΔAUC (λ0.5 − λ0), n=2000, seed 42:  
**Δ = −0.0003**, 95% CI **[−0.0022, +0.0018]**, **lo > 0? No**.

Train CE dynamics (λ=0): **1.35 → 1.11** (min); best val still **ep1** then mild train-fit / val drift.

### 4.3 GREEN late-fusion confirmation — `b1_green_20260715`

| Model | val 4-AUC | test 4-AUC | test bin | vs W0 0.666 |
|---|---:|---:|---:|---|
| Pure seq λ=0 (post-fix) | 0.680 | **0.652** | 0.679 | −0.014 |
| Seq + GREEN late-fuse λ=0 | 0.686 | **0.638** | 0.660 | −0.028 |
| Path A W0 (CatBoost GREEN) | — | **0.666** | 0.689 | ref |

Fusion: all numeric `watch_green` cols (30), train-only impute+z, concat to attention `z` → class head; glu head unchanged. Best still **ep1**.

**Read:** person GREEN does **not** lift the LSTM over pure daily seq on this recipe; does **not** match Path A W0. Gap to GBM is **architecture / how statics are used**, not “missing columns in the seq model.”

### 4.4 Overfit gate — `b1_overfit50_fix`

50 pids, λ=0, full epochs: CE **1.38→0.79**, best val **0.775** @ ep19 → backbone capacity OK after C1+C2.

---

## 5. Decision rules applied

| Question | Rule | Outcome |
|---|---|---|
| Backbone learnable? | Overfit CE drop; full val ≥ 0.60 | **Pass** |
| Multi-task helps? | test paired ΔAUC CI **lo > 0** | **Fail** (null) |
| vs Path A floor | informational ≥ 0.666 | **Fail** pure seq & fusion |
| GREEN fuse raises λ=0? | test lift vs pure seq / ≥ W0 | **Fail** |
| Proceed to B2/B4? | always after B1 report | **Yes** |

---

## 6. What B1 is / is not

**Is:**
- Controlled ablation: same backbone ± day-level glu multi-task  
- Proof that FE+scaling bugs can fake “DL fails”  
- A ~0.65 daily-sequence **watch-only** floor under this protocol  

**Is not:**
- Path A replacement or deployable C1 competitor  
- Evidence that privileged CGM (B4 trajectory) is dead  
- A license to keep tuning λ or full C1 survey stacks on this spine  

---

## 7. Implications for Path B ladder

| Next | Implication |
|---|---|
| **B2** two-stage | Still useful; `cgm_person` ready. Expect limited lift if stage-1 glucose from daily watch is hard. |
| **B4** trajectory teacher | **Headline** — B1 null on *scalar daily* multi-task does **not** kill full-curve / rep-distill. |
| **B3** logit-KD | Last baseline to beat. |
| B1 further | **Frozen.** Optional hybrid (seq `z` → GBM) is a separate experiment, not B1 reopen. |

---

## 8. Reproduce

```bash
export HIP_VISIBLE_DEVICES=0

# FE (after C1 sleep fix)
.venv/bin/python -m pipeline.run_fe --blocks watch_daily

# Post-fix multi-task claim run
.venv/bin/python -m training.path_b.b1 --run-id b1_grid_YYYYMMDD_fix \
  --lambdas 0,0.5 --device cuda

# GREEN late-fuse confirmation (λ=0)
.venv/bin/python -m training.path_b.b1 --run-id b1_green_YYYYMMDD \
  --lambdas 0 --device cuda --green-fusion
```

Smoke: `--quick --device cuda`. Overfit gate: `--max-participants 50 --lambdas 0` (**no** `--quick`).

---

## 9. Artifacts

| Path | Role |
|---|---|
| `artifacts/b1_grid_20260714/` | pre-fix baseline (invalid ceiling) |
| `artifacts/b1_overfit50_fix/` | G1 |
| `artifacts/b1_fix_20260715/` | G2 λ=0 |
| `artifacts/b1_grid_20260715_fix/` | **G3 multi-task claim** |
| `artifacts/b1_green_20260715/` | **G4b fusion confirmation** |
| `AUDIT_B1_UNDERPERF.md` | root cause |
| `PLAN_B1_FIX.md` | fix + gate plan |

---

## 10. Freeze statement

**B1 is closed.** Canonical multi-task result: `b1_grid_20260715_fix`. Canonical pure-seq floor: same run λ=0 (test **0.652**). Fusion confirmation: `b1_green_20260715` (no raise). Pre-fix `b1_grid_20260714` retained only as broken-input history.

Next package work: **B2** (then **B4**, **B3** last).
