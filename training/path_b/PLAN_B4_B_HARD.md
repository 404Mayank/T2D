# B4-B hard-teacher sensitivity (post B4-B null)

**Date:** 2026-07-15  
**Status:** **CONCLUDED** 2026-07-15 — `REPORT_B4_B_HARD.md` (H1+H2 both null).  
**Why:** B4-B teacher was `X∥cgm` AE (Pearson ~0.99 = copy channel). Null may be **easy-teacher artifact**.  
**Authority:** `PLAN_B4.md` B4-B locks; `REPORT_B4_B.md` gap; no reopen of B4-A claim.

## Goal
Does **hard** privileged representation distillation beat μ=0 student and/or matched D1?

## Design (one package, two teachers)

| ID | Teacher input | Teacher loss | Student | Distill |
|---|---|---|---|---|
| **H1 CGM-only** (primary) | `cgm` (+ `tod_sin/cos` for phase) only; **no wear** | masked traj recon / identity on cgm channel | X wear only | μ MSE(z_S, sg z_T) train∩aux |
| **H2 wear→CGM** (secondary) | wear X only | masked traj MSE (hard map) | X wear only | same |

**Shared locks (from B4-B):**
- No class head on teacher loss (no class leakage via teacher).
- Student: CE + μ distill; μ∈{0, 0.3, 1.0}; μ=0 control.
- z_T in student **loss** only for train∩aux; never CGM at student infer.
- Hybrid: z∥C1 GBM vs matched D1 (same pid_allow).
- Same grid FE / subwindow / pools as B4.

## Bars
| Bar | Pass |
|---|---|
| Distill helps neural | μ>0 − μ0 test 4-AUC CI lo>0 |
| Ambition | best hybrid − D1 CI lo>0 |
| Teacher non-deg | H1: z linear probe or traj quality; H2: val Pearson > 0.15 (wear→cgm) |

## Run
`b4b_hard_YYYYMMDD` — H1 full (teacher + μ grid + hybrid); H2 if time (same).

## Out of scope
B3, re-HPO B4-A, re-clean, Path A changes.
