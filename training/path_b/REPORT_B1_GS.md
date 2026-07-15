# REPORT_B1_GS — gradient-balanced multi-task retry

**Status:** **CONCLUDED** (2026-07-16).  
**Plan:** `PLAN_B1_GS.md` (critiqued → revised → implemented).  
**Sibling of freeze:** does **not** overwrite `REPORT_B1.md` / plain-λ null.  
**Authority:** this report + `DECISIONS.md` (2026-07-16 entries).

Path A numbers are **frozen and unchanged**. W0 **0.666** / bin **0.689**; C1 **0.738** / **0.831**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does PCGrad multi-task beat A0 (λ=0)? | **No** — Δ +0.0006, CI lo ≯ 0 |
| Does uncertainty weighting beat A0? | **No** — Δ −0.0002, CI lo ≯ 0 |
| Does plain λ=0.5 still null (repro)? | **Yes** — Δ −0.0003, matches freeze |
| Did task-gradient conflict exist? | **Yes, moderate** (~15–32% of glu-active steps) |
| Was glu head “alive”? | **No** — best val z-MSE ~1.32 ≫ constant-predictor 1.0 |
| Null type | **Balancing-tried + weak-aux**: conflict present and partially addressable, but watch→day-CGM signal too weak for class transfer on this backbone |
| Proceed? | GS family on day-level B1 spine **closed**. Path B claim ladder complete (B2/B2-V2/B4/B4-V2/B3 frozen); residual only via new `PLAN_*` (e.g. SSL, CORN) |

**One-liner:** Gradient surgery and uncertainty weighting close the “naive-λ only” gap from `REVIEW_PHASES`. They **do not** raise 4-AUC over class-only. The glu head remains sub-constant; privilege transfer fails at the aux SNR wall, not only at loss weighting.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Backbone | `attn_lstm_64` (same as freeze) |
| Features / scale | 18 daily dims; C1+C2 sleep + train-only z-score |
| Pool / split | core 1824; train/val/test **1277 / 270 / 277** |
| Class weights | inverse-freq train, sum-normalize (pinned) |
| Arms | A0, plain (λ=0.5), PCGrad, UW |
| PCGrad shared | `input` / `lstm` / `proj` only (**not** `attn`) |
| PCGrad \|T\|=2 | project each task onto **original** peer grad |
| UW | CE-primary prior; `lr_s=0.1×`; clamp ±5 |
| Claim rule | paired person boot ΔAUC vs A0, n=2000 seed 42, CI lo > 0 |
| Device | ROCm CUDA; `cudnn.enabled=False` |

---

## 2. Runs

| Run id | Role | Claim? |
|---|---|---|
| `b1gs_smoke` | 64 pids, 3 ep, arms default | no |
| `b1gs_overfit50_a0` | G1 class-only overfit | diagnostic |
| `b1gs_overfit50_pcg` | G1 PCGrad overfit | diagnostic |
| **`b1gs_grid_20260716`** | full claim arms | **yes** |

Unit: `python -m training.path_b.b1.test_balance_unit` → **10/10**.

---

## 3. Gates

| Gate | Result |
|---|---|
| R3a overfit50 A0 | **PASS** — CE 1.38 → **0.79** min; best val **0.775** @ ep19 |
| R3b overfit50 A_pcg | **PASS** — CE → **0.71** min; best val **0.850** @ ep23; conflict rates up to **1.0** (real surgery) |
| A0 full val ≥ 0.60 | **PASS** — val **0.680** (matches freeze λ=0 bit-exact on test **0.65227**) |

---

## 4. Claim results — `b1gs_grid_20260716`

### 4.1 Primary table (test raw)

| Arm | best ep | val 4-AUC | test 4-AUC | test bin | test glu z-MSE | Δ vs A0 | 95% CI | lo>0? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **A0** | 1 | **0.680** | **0.6523** | 0.679 | 1.507 | — | — | — |
| plain λ=0.5 | 1 | 0.678 | 0.6520 | 0.680 | 1.429 | −0.0003 | [−0.0022, +0.0018] | **No** |
| **PCGrad** | 1 | 0.677 | **0.6529** | 0.679 | 1.424 | **+0.0006** | [−0.0030, +0.0043] | **No** |
| **UW** | 1 | 0.678 | 0.6521 | 0.680 | 1.429 | −0.0002 | [−0.0022, +0.0018] | **No** |

Path A floor (info): W0 0.666 / C1 0.738 — none of the arms clear the floor; not the claim.

### 4.2 Per-class test OVR (A0 vs best GS)

| Class | A0 | PCGrad | plain | UW |
|---|---:|---:|---:|---:|
| 0 | 0.679 | 0.679 | 0.680 | 0.680 |
| 1 | 0.616 | 0.613 | 0.613 | 0.613 |
| 2 | 0.598 | 0.598 | 0.596 | 0.596 |
| 3 | 0.716 | 0.722 | 0.719 | 0.719 |

Class-2 remains the bottleneck; GS does not fix label geometry.

### 4.3 Conflict diagnostics (full train; glu-active only)

| Arm | conflict_grad_source | mean conflict rate | min–max | mean cos |
|---|---|---:|---|---:|
| plain | `plain_lambda_scaled_probe` | **0.220** | 0.15–0.33 | ~0.20 |
| PCGrad | `unweighted_pcgrad` | **0.208** | 0.13–0.33 | ~0.20 |
| UW | `unweighted_probe` | **0.219** | 0.15–0.33 | ~0.20 |

- Denominator: **40** glu-active steps / epoch (batch_size 32; train n=1277 → 40 batches; all batches had some glu mass in practice).
- Band: **moderate (5–20%]** edge into **high** on some epochs — plan language: *intermittent conflict; transfer still null*.
- Mean cos **positive** (~0.20): majority of steps are non-conflicting; PCGrad is a no-op on those steps (combined = g_CE + g_glu unweighted ≡ plain λ=1.0 on those steps).

### 4.4 Glu-head quality

| Arm | val z-MSE | test z-MSE | glu-alive? (≤0.95 or Pearson>0.10) |
|---|---:|---:|---|
| A0 (untrained glu) | 1.407 | 1.507 | no |
| plain | 1.322 | 1.429 | **no** |
| PCGrad | **1.315** | **1.424** | **no** |
| UW | 1.322 | 1.429 | no |

Constant predictor after z-score ≈ **1.0**. All multi-task arms beat random A0 slightly but stay **worse than constant** — head is not learning useful day-level CGM structure from 18 daily watch dims.

### 4.5 UW trajectory

- `s_ce`: 0.004 → 0.029 (slight down-weight of CE scale)
- `s_glu`: −0.000 → −0.007 (slight **up**-weight of glu)
- Clamp hits: **0** (never pegged)
- With best_epoch=**1**, selected ckpt barely sees UW move — same ES pattern as freeze

### 4.6 best_epoch

All claim arms: **best_epoch = 1** (val peaks immediately, then mild train-fit / val drift). Matches freeze. GS had little opportunity to reshape the *selected* representation even though train ran 16 epochs.

---

## 5. Decision rules applied

| Question | Rule | Outcome |
|---|---|---|
| Backbone learnable? | overfit CE drop | **Pass** (A0 + PCGrad) |
| A0 floor intact? | val ≥ 0.60 | **Pass** (0.680; test matches freeze) |
| Glu-alive? | z-MSE ≤ 0.95 | **Fail** all arms |
| GS multi-task helps? | CI lo > 0 vs A0 | **Fail** PCGrad, UW, plain |
| Data null vs balancing-tried? | §5 bands | **Moderate conflict + dead glu** → not pure data-null; **balancing tried and class transfer still null** |
| Open C5 from this run? | GS null + glu not alive + no high-only story | **Optional separate plan only** — freeze already tried full GREEN late-fuse; SNR wall more binding than missing SRI/RAR alone |

---

## 6. Interpretation

1. **Freeze plain-λ null is reproduced** (A0 test 0.65227; plain Δ −0.0003 identical to `b1_grid_20260715_fix`).
2. **Conflict is real but intermittent** (~20% steps). PCGrad is active often enough that overfit50 shows different dynamics (and higher conf rates), yet on the full cohort the selected epoch-1 checkpoint is indistinguishable from plain multi-task on class AUC.
3. **Aux SNR wall:** day-level glu z-MSE stays ~1.32–1.43. B2 Stage-1 R²≈0.05 and B4 Pearson~0.25 are consistent external priors. No balancing method can mint class signal from a near-unlearnable aux head.
4. **REVIEW_PHASES gap closed for this recipe family** on the day-level B1 spine: plain-λ, PCGrad, and UW all null. Residual Path B novelty is not “another B1 λ/GS grid.”
5. **Informational:** pure-seq still 0.652 < W0 0.666; trees-at-n≈1.8k prior holds.

---

## 7. What this is / is not

**Is:**
- Controlled ablation: same backbone, C1+C2 data, gradient-balanced vs class-only
- Proof that moderate task conflict exists and was measured
- Closure of the “only tried naive λ” objection for day-level multi-task

**Is not:**
- A Path A or C1 beater
- Evidence that trajectory LUPI / logit-KD (B3) / SSL are dead
- License to reopen plain-λ grids or more GS on this spine without a new plan

---

## 8. Artifacts

| Path | Role |
|---|---|
| `b1/artifacts/b1gs_grid_20260716/` | **claim** |
| `b1/artifacts/b1gs_overfit50_a0/` | G1 A0 |
| `b1/artifacts/b1gs_overfit50_pcg/` | G1 PCGrad |
| `b1/artifacts/b1gs_smoke/` | plumbing only |
| `b1/balance.py` | PCGrad + UW |
| `PLAN_B1_GS.md` | protocol |

Reproduce:

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b1.test_balance_unit
.venv/bin/python -m training.path_b.b1 --run-id b1gs_grid_YYYYMMDD --arms default --device cuda
```

---

## 9. Freeze statement (GS)

**B1 gradient-balanced multi-task is closed.** Canonical claim run: `b1gs_grid_20260716`.  
Canonical class-only floor unchanged: test **0.652**.  
No GS arm clears pre-registered CI lo > 0.  
Frozen plain-λ `REPORT_B1.md` remains valid history; this report is the sibling GS claim.
