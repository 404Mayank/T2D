# B4 report — seq2seq CGM trajectory + hybrid C1

**Status:** **B4-A claim grid CONCLUDED** (2026-07-15). B4-B also concluded — see **`REPORT_B4_B.md`** (`b4b_distill_20260715`).  
**Protocol:** `PLAN_B4.md` (post-critique).  
**Claim run:** `b4_grid_20260715`  
**Overfit (non-claim):** `b4_overfit50_20260715`  
**Smoke (non-claim):** `b4_smoke_20260715`  

Path A numbers **frozen / unchanged**. W0 **0.6662** / bin **0.6889**; C1 **0.7378** / **0.8309**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **any deployable B4 arm beat C1 / matched D1**? | **No.** Best hybrid = **S0+C1** test 4-AUC **0.713** ≪ D1 **0.736** (freeze C1 **0.738**). |
| Does trajectory multi-task help (Sλ − S0)? | **No** — λ=0.3/1.0 Δ4-AUC **−0.010 / −0.008**, CI lo≯0. |
| Does traj multi-task help hybrid (Sλ+C1 − S0+C1)? | **No** — null / slightly worse. |
| Is traj head non-degenerate? | **Mostly yes** — val Pearson **0.15–0.25** (λ>0); beats mean-predictor RMSE; λ=0 Pearson **0.145** (borderline). |
| Is sequence floor learnable? | **Yes** — S0 val **0.682** / test **0.646** (informational vs W0 0.666, B1 0.652). |
| D1 protocol parity vs freeze? | Re-fit D1 test **0.7359** (Δ freeze **−0.0019**, within 0.01) — fair bar OK. |
| Proceed ladder? | **Yes → B4-B done (null); B3 next.** B4-A headline ambition **failed**. |

**Scientific takeaway:** Full-curve CGM multi-task on a 5-min CNN encoder **does not** raise deployable T2D over matched C1. Trajectory supervision is **non-null as a reconstruction task** (Pearson↑ with λ) but **does not transfer** to 4-AUC (Sλ ≯ S0; hybrid z∥C1 **hurts** vs pure C1). Same pattern as B1/B2 scalar-CGM nulls with a richer target — LUPI headroom may still exist (B2 oracle), but **this B4-A recipe is not a C1-beater**.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| FE | `grid_5min` full core 6.88M bins × 1824; wear-density subwindow **CGM-free**; T=2016 (7d), T_min=1008 |
| Encoder | Patch CNN h=64, patch 12; attn pool → class; traj head per-bin |
| λ | {0, 0.3, 1.0} |
| Class pool | wearable_core (1 pid dropped T_min → **1276/270/277** train/val/test) |
| Traj pool | aux concurrent bins only |
| Ambition head | GBM Stage-2 on **z ∥ C1** (CatBoost/LGBM Path A family, 50 trials) |
| D1 | re-fit C1 on **same pid_allow** as sequence survivors |
| Seed | 42; bootstrap n=2000 |
| Device | ROCm RX 5600 (`HIP_VISIBLE_DEVICES=0`) |

---

## 2. Gates

| Gate | Result |
|---|---|
| FE full acceptance | **PASS** (aux median concurrent ~210 h; UAB ToD OK) |
| Overfit50 λ=0 | **PASS** — CE 1.40→0.89; best val 4-AUC **0.878** (ep5) |
| S0 val ≳ 0.55 | **PASS** (0.682) |
| Traj non-deg (Pearson>0.15 **and** beat mean) | **PASS** for λ∈{0.3,1.0}; λ=0 Pearson 0.145 borderline + beats mean |
| Empty-split / grid coverage asserts | **PASS** |

---

## 3. Neural arms (`b4_grid_20260715`)

| Arm | val 4-AUC | test 4-AUC | test bin | val traj Pearson | test traj Pearson |
|---|---:|---:|---:|---:|---:|
| **S0** (λ=0) | **0.682** | **0.646** | 0.642 | 0.145 | 0.144 |
| **S0.3** | 0.685 | 0.637 | 0.637 | 0.243 | 0.219 |
| **S1.0** | 0.687 | 0.639 | 0.650 | 0.255 | 0.256 |

### Sλ − S0 (test, paired boot)

| λ | Δ4-AUC | 95% CI | lo>0? |
|---:|---:|---|---|
| 0.3 | −0.010 | [−0.029, +0.010] | **No** |
| 1.0 | −0.008 | [−0.023, +0.007] | **No** |

**Multi-task ablation: fail.**

---

## 4. Hybrid ambition arms

| Arm | val 4-AUC | test 4-AUC | test bin | test AUPRC | family |
|---|---:|---:|---:|---:|---|
| **D1** (C1 only) | 0.740 | **0.736** | 0.809 | 0.487 | CatBoost |
| Frozen C1 anchor | — | **0.738** | 0.831 | 0.469 | — |
| **S0+C1** | 0.737 | **0.713** | 0.819 | 0.470 | CatBoost |
| **S0.3+C1** | 0.734 | 0.714 | 0.809 | 0.461 | CatBoost |
| **S1.0+C1** | 0.736 | 0.703 | 0.806 | 0.450 | CatBoost |

### vs D1 (test, paired boot)

| Contrast | Δ4-AUC | 95% CI | lo>0? |
|---|---:|---|---|
| S0+C1 − D1 | **−0.022** | [−0.049, +0.004] | **No** |
| S0.3+C1 − D1 | **−0.022** | [−0.047, +0.003] | **No** |
| S1.0+C1 − D1 | **−0.033** | [−0.062, −0.005] | **No** (CI entirely ≤0) |
| S0.3+C1 − S0+C1 | +0.000 | [−0.016, +0.017] | No |
| S1.0+C1 − S0+C1 | −0.011 | [−0.027, +0.005] | No |

**User ambition (beat C1/D1): fail.**  
**Encoder z + C1 without traj also fails** vs D1 — adding sequence embedding **dilutes** the strong tabular C1 stack under this recipe (same family of outcome as B1 GREEN neural fuse null / B2 weak Ŷ harm).

D1 vs freeze: **0.7359 − 0.7378 = −0.0019** → within 0.01; fair bar = re-fit D1 (no unreproduced-freeze fallback).

---

## 5. Decision bars (pre-registered)

| Bar | Outcome |
|---|---|
| Sλ+C1 > D1 (CI lo>0) | **Fail** |
| Sλ > S0 (CI lo>0) | **Fail** |
| Sλ+C1 > S0+C1 | **Fail** |
| S0+C1 > D1 | **Fail** (headline would have been encoder+static, not traj — also fail) |
| Traj quality | **Pass** (λ>0); λ=0 borderline Pearson |
| Kill S0 val ≲0.55 | **Not triggered** |

---

## 6. Interpretation

1. **B4-A multi-task is a controlled null** on 4-AUC despite a working traj head (Pearson rises with λ).  
2. **Deployable hybrid loses to pure C1** — frozen `z` is not additive under Path A GBM Stage-2; may be collinear noise / weak representation.  
3. **Convergent with B1/B2:** scalar multi-task null, point-estimate handoff null, trajectory multi-task null — privilege (oracle B2) still real; **handoff form** still wrong for deployable raise.  
4. **S0 ~0.65** is a real sequence floor (overfit gate clean) but does not beat W0 tabular or C1.  
5. One train pid dropped by T_min (1276 vs 1277) — logged; D1 used same `pid_allow`.

---

## 7. What B4-A is / is not

**Is:** Critiqued FE + CNN traj multi-task + hybrid ambition run with paired boots and matched D1.  
**Is not:** B3 logit-KD; a deployable C1 raise; proof that all LUPI is impossible. B4-B (easy + hard teachers) is reported separately and is also a deployable null.

---

## 8. Residual / next

| Option | Recommendation |
|---|---|
| **B4-B rep-distill** | **Done** — null / can hurt; see `REPORT_B4_B.md` |
| Deeper encoder / longer T / SSL | New plan only |
| z→GBM without C1 / different pool | Footnote |
| **B3** | Ladder **next** |

---

## 9. Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
# FE (done)
.venv/bin/python -m pipeline.run_fe --blocks grid_5min

# overfit
.venv/bin/python -m training.path_b.b4 --run-id b4_overfit50_YYYYMMDD \
  --mode neural --lambdas 0 --max-participants 50 --device cuda

# claim (neural + D1 + hybrid)
.venv/bin/python -m training.path_b.b4 --run-id b4_grid_YYYYMMDD --mode all --device cuda
```

Artifacts: `training/path_b/b4/artifacts/b4_grid_20260715/`.

---

## 10. Package

```
pipeline/fe/grid_5min.py
training/path_b/b4/
  config.yaml data.py model.py train.py hybrid.py run.py
```

Impl note: first hybrid pass failed on object-dtype `split` in `embeddings.npz` (pickle); fixed to U16 + legacy load; hybrid re-run with `--mode hybrid --resume`.
