# B4-B hard-teacher sensitivity

**Status:** **CONCLUDED** (2026-07-15).  
**Plan:** `PLAN_B4_B_HARD.md`.  
**Runs:** `b4b_hard_h1_20260715` (CGM-only teacher), `b4b_hard_h2_20260715` (wear→CGM teacher).  
**Parent audit gap closed:** original B4-B teacher was easy `X∥cgm` AE (Pearson ~0.99).

Path A freeze unchanged. Matched D1 in both runs: **0.7359**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Was easy-teacher the reason B4-B failed? | **No.** Hard teachers also **null / hurt**. |
| H1 CGM-only distill raise vs μ=0? | **Fail** — μ=0.3 **hurts** (CI entirely &lt;0); μ=1 null. |
| H1 hybrid beat D1? | **Fail** — best **0.723** &lt; D1 **0.736**. |
| H2 wear→CGM distill raise? | **Fail** — μ=0.3 hurts; μ=1 null. |
| H2 hybrid beat D1? | **Fail** — all hybrids **&lt;** D1 (μ=1 CI entirely ≤0). |
| H2 teacher hard map real? | **Yes** — val Pearson **~0.30** (not 0.99 copy). |

**Takeaway:** Closing the easy-teacher loophole **does not** produce a deployable C1 raise. Privilege still exists (prior probes); **wear student + z-MSE distill + hybrid GBM** remains a null family under H1 and H2.

---

## 1. Setup

| Mode | Teacher input | Teacher loss | Student |
|---|---|---|---|
| **H1 `cgm_only`** | cgm + tod only (**no wear**) | traj recon | CE + μ distill |
| **H2 `wear_cgm`** | wear X only | traj MSE (hard map) | same |
| μ | {0, 0.3, 1.0} | | |
| Hybrid | z∥C1 → Path A GBM vs D1 | | |

---

## 2. Teacher quality

| | H1 cgm_only | H2 wear_cgm | Easy (prior B4-B) |
|---|---:|---:|---:|
| Best val traj RMSE | **0.167** | **1.137** | 0.175 |
| Val traj Pearson | **~0.990** | **~0.30** | ~0.99 |
| Interpretation | privileged CGM encoder (still “sees” glucose) | real wear→curve skill | copy channel |

H1 Pearson still ~0.99 is expected (teacher **is** a CGM sequence model with tod). It is **not** wear-conditioned recon. H2 is the true hard map and matches B4-A wear→cgm skill (~0.25–0.30).

---

## 3. Student neural (test 4-AUC)

| μ | H1 | H2 | Easy B4-B (ref) |
|---:|---:|---:|---:|
| 0 | **0.646** | **0.646** | 0.646 |
| 0.3 | **0.626** | **0.620** | 0.625 |
| 1.0 | 0.634 | 0.638 | 0.636 |

### μ − μ0 boot

| Run | μ | Δ | CI | lo>0? |
|---|---:|---:|---|---|
| H1 | 0.3 | −0.021 | [−0.040, −0.002] | **No (hurts)** |
| H1 | 1.0 | −0.013 | [−0.032, +0.007] | No |
| H2 | 0.3 | −0.026 | [−0.047, −0.006] | **No (hurts)** |
| H2 | 1.0 | −0.008 | [−0.022, +0.005] | No |

Train-aux cosine still rises with μ (H1: 0.14→0.49→0.66; H2: 0.21→0.23→0.75) — distill **aligns** z without helping ranking.

---

## 4. Hybrid vs D1 (test 4-AUC)

| Arm | H1 | H2 |
|---|---:|---:|
| D1 | 0.736 | 0.736 |
| Dμ0+C1 | 0.713 | 0.713 |
| Dμ0.3+C1 | 0.705 | 0.712 |
| Dμ1+C1 | **0.723** | 0.708 |

All Δ vs D1: CI lo ≯ 0. H2 μ=1 hybrid CI **entirely &lt; 0**.

---

## 5. Bars

| Bar | H1 | H2 |
|---|---|---|
| Distill μ>0 vs μ0 | **Fail** | **Fail** |
| Hybrid vs D1 | **Fail** | **Fail** |
| Teacher non-deg | Pass (privileged) | Pass (Pearson 0.30) |
| Beat C1 | **Fail** | **Fail** |

---

## 6. Interpretation

1. **Easy-teacher was not the binding failure.** H1/H2 reproduce the null.  
2. **Matching a privileged (H1) or hard wear→glu (H2) z via MSE does not raise 4-AUC** and often **hurts** at μ=0.3.  
3. **Hybrid still dilutes C1** — best hard-teacher hybrid (H1 μ=1: 0.723) still &lt; D1; worse than easy-teacher’s near-tie 0.735.  
4. **B4 LUPI cell (traj multi-task + rep-distill, easy+hard teachers) is closed null** for deployable raise under current encoder/distill recipe.  
5. Residual science value: H2 teacher confirms wear→curve SNR ~0.3 Pearson — multi-task ceiling is low; C1 tabular remains dominant.

---

## 7. Ladder

**B4 fully closed** (A + B easy + B hard). **B3 next.**  
Do not reopen without a new `PLAN_*` (e.g. different distill objective, CGM-only **no tod** ablations, or logit-KD B3).

## 8. Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h1_YYYYMMDD \
  --mode distill_hybrid --teacher-mode cgm_only --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h2_YYYYMMDD \
  --mode distill_hybrid --teacher-mode wear_cgm --device cuda
```
