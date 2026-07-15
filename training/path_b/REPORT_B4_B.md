# B4-B report — representation distillation (CGM-AE teacher → wear student)

**Status:** **B4-B CONCLUDED** (2026-07-15).  
**Protocol:** `PLAN_B4.md` §2.1 B4-B / §2.7 gate (S0 learnable after B4-A).  
**Claim run:** `b4b_distill_20260715` (`--mode distill_hybrid`)  
**Authority:** this report + `DECISIONS.md` + B4-A `REPORT_B4.md`.

Path A freeze unchanged: C1 **0.7378** / **0.8309**. Matched D1 in this run: **0.7359** / **0.809**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **rep-distill** raise neural 4-AUC vs μ=0 student? | **No** — μ=0.3 **hurts** (Δ **−0.021**, CI entirely &lt;0); μ=1.0 null (Δ **−0.010**, CI lo≯0). |
| Does distill student **z∥C1** beat matched D1? | **No** — best **Dμ1+C1 0.735** ≈ D1 **0.736** (Δ **−0.001**, CI lo≯0). |
| Does teacher learn glucose structure? | **Yes** — val traj Pearson **~0.99**, RMSE **0.175** (privileged CGM in teacher input). |
| Does student match teacher z? | **Yes** — train-aux cosine **0.09 → 0.48 → 0.64** as μ 0→0.3→1. |
| Deployable beat C1? | **No.** Best deployable still **D1 ≡ C1 family**. |
| Ladder? | **B4 closed** (A+B). **B3 next.** |

**Scientific takeaway:** A strong CGM-privileged teacher representation **can be distilled** into the wearable student (cosine↑), but that glucose-shaped `z` **does not improve** 4-class ranking under CE (and can **hurt**). Hybrid GBM(z∥C1) at high μ almost recovers pure C1 but **does not beat** it. B4-B is a **controlled null** for the novelty arm (rep-distill under LUPI), not a C1 raise.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Teacher | PatchCNN on **X ∥ cgm** (cgm zeroed off `traj_mask`); **no CE** — traj MSE only; fit train (aux bins); ES val traj RMSE |
| Student | PatchCNN on **X only**; `L = CE + μ ‖sg(z_T)−z_S‖²`; distill loss **only train∩aux** persons |
| μ grid | {0, 0.3, 1.0}; μ=0 = CE-only control |
| z_T scope | Teacher may see train/val/test CGM for forward; student **loss** uses train aux only; val/test z_T not in student training |
| Hybrid | freeze student z → GBM(z ∥ C1) vs matched D1 (`pid_allow` = sequence pool) |
| Pool | same as B4-A (1 T_min drop → 1823; test n=277) |
| Seed / boot | 42 / n=2000 |
| Device | ROCm RX 5600 |

---

## 2. Teacher quality

| Metric | Value |
|---|---|
| Best epoch | 57 |
| Best val traj RMSE | **0.175** |
| Val traj Pearson (late epochs) | **~0.988** |

Teacher is a near-perfect **privileged** CGM reconstructor when CGM is an input channel — expected and confirms FE + decoder path.

---

## 3. Student neural results

| μ | val 4-AUC | test 4-AUC | test bin | train-aux z cosine | train-aux z MSE |
|---:|---:|---:|---:|---:|---:|
| **0** (control) | 0.682 | **0.646** | 0.642 | 0.091 | (high) |
| **0.3** | 0.686 | **0.625** | 0.635 | 0.478 | ↓ |
| **1.0** | 0.679 | **0.636** | 0.618 | **0.644** | ↓ |

μ=0 test **0.646** matches B4-A S0 exactly (same backbone/seed family) — control is trustworthy.

### μ − μ0 (test, paired boot)

| μ | Δ4-AUC | 95% CI | lo>0? |
|---:|---:|---|---|
| 0.3 | **−0.021** | **[−0.041, −0.003]** | **No** (hurts) |
| 1.0 | −0.010 | [−0.025, +0.003] | **No** |

**Rep-distill multi-task / LUPI student raise: fail.**

---

## 4. Hybrid ambition (z ∥ C1 → GBM)

| Arm | test 4-AUC | test bin | Δ vs D1 | CI lo>0? |
|---|---:|---:|---:|---|
| **D1** | **0.736** | 0.809 | — | — |
| Dμ0+C1 | 0.711 | 0.809 | −0.025 | No |
| Dμ0.3+C1 | 0.720 | 0.821 | −0.016 | No |
| **Dμ1+C1** | **0.735** | 0.819 | **−0.001** | No |

Higher μ makes hybrid **closer** to D1 (less harmful dilution than B4-A S0+C1 **0.713**), but still **not a raise**. Ambition bar **fail**.

---

## 5. Decision bars

| Bar | Outcome |
|---|---|
| Student μ>0 vs μ=0 (CI lo>0) | **Fail** (0.3 hurts; 1.0 null) |
| Dμ*+C1 vs D1 (CI lo>0) | **Fail** |
| Teacher non-degenerate | **Pass** (Pearson ~0.99) |
| Distill alignment | **Pass** (cosine↑ with μ) |
| Beat frozen C1 | **Fail** |

---

## 6. Interpretation

1. **LUPI representation exists** in the teacher (near-perfect traj under privilege).  
2. **Student can copy that z** (cosine 0.64 at μ=1) without improving class ranking — glucose-shaped features are **not the bottleneck** for 4-AUC under this encoder+CE recipe (or are misaligned with the label geometry).  
3. **Forcing distill can harm** CE optimization (μ=0.3).  
4. **Hybrid almost ties C1** at μ=1 — distilled z is less harmful noise than B4-A multi-task z, but still not additive signal over C1.  
5. Together with B4-A: trajectory multi-task **null**, rep-distill **null** for deployable raise → Path B headline **does not beat C1** under B4 formulations run; **B3** remains the remaining KD baseline cell.

---

## 7. What B4-B is / is not

**Is:** Controlled rep-distill under LUPI with CGM-AE teacher (no class-head leakage); hybrid ambition arm; paired boots.  
**Is not:** Soft logit-KD (B3); deployable C1 raise; proof against all LUPI.

---

## 8. Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b4 --run-id b4b_distill_YYYYMMDD \
  --mode distill_hybrid --device cuda
# neural-only distill:
.venv/bin/python -m training.path_b.b4 --run-id b4b_distill_YYYYMMDD \
  --mode distill --device cuda
```

Artifacts: `training/path_b/b4/artifacts/b4b_distill_20260715/`.

---

## 9. Package

`training/path_b/b4/distill.py` + `run.py` modes `distill` / `distill_hybrid`; config `distill.mus`.
