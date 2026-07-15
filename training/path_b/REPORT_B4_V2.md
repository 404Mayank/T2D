# B4-V2 report — RKD/CRD distill + PCGrad MTL + OOF fusion

**Status:** **CLAIM LADDER CONCLUDED** (2026-07-16). Post-claim critique: **approve-with-caveats** (null authentic; freeze stands).  
**Protocol:** `PLAN_B4_V2.md` (critique addressed).  
**Sibling of** frozen `REPORT_B4.md` / `REPORT_B4_B.md` / `REPORT_B4_B_HARD.md` — does **not** overwrite them.  
**Accept-state:** **honest null** (gates pass; science bars fail). Path A freeze unchanged.

Path A freeze: C1 test 4-AUC **0.7378** / binary **0.8309**.  
Matched **D1** this run: **0.7359** (Δ freeze **−0.0019**, fair bar OK).

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **PCGrad** traj MTL beat S0? | **No** — Δ **−0.016**, CI entirely ≤0 (hurts). Early grad cos **+0.16** → **no conflict** (mechanism-null for balancing). |
| Does **RKD** distill beat μ=0? | **No** — μ=0.3 null; μ=1.0 **hurts** (CI entirely ≤0). |
| Does **CRD** beat μ=0? | **No** — both μ null (family check). |
| Does hybrid beat matched **D1**? | **No** — best F1 OOF-z∥C1 **0.726** &lt; D1 **0.736** (CI lo≯0). |
| Teacher H2 non-degenerate? | **Yes** — Pearson **0.301**, beats mean; probes GO (mlp 0.66, knn 0.57). |
| Deployable beat C1/D1? | **No.** |
| Ladder next? | **Do not reopen B4-V2.** Residual: **`PLAN_SSL.md`** (B3 already frozen null). Not another KD objective tweak. |

**Scientific takeaway:** Stronger formulations of the same B4 cells (relational/contrastive distill, PCGrad MTL, OOF fusion) still **do not** raise deployable 4-AUC over matched C1/D1. Teacher wear→curve SNR remains ~0.30; z-only ranking ~0.64; frozen/OOF z **dilutes** C1. Null is **authentic for V2 recipes**, with healthy teacher and clean gates — valid accept-state per plan.

---

## 1. Runs (ids)

| Stage | Run id | Role |
|---|---|---|
| Overfit50 | `b4v2_overfit50_20260715` | G0 |
| Teacher H2 + probe | `b4v2_teacher_h2_20260715` | G3 |
| MTL S0 + S_pc | `b4v2_mtl_20260715` | G1 + MTL bar |
| RKD distill + hybrid | `b4v2_rkd_20260715` | distill + F2 on student z |
| Hybrid F0b/F1/F2 | `b4v2_hybrid_20260715` | fusion (μ=0 OOF) |
| CRD sensitivity | `b4v2_crd_20260715` | objective family |
| Ladder log | `b4v2_ladder_20260715.log` | orchestration |

Device: ROCm RX 5600 (`HIP_VISIBLE_DEVICES=0`). Seed 42. Bootstrap n=2000.

Pool: wearable_core survivors **1823** (drop pid **7189** T_min); train/val/test **1276/270/277**.

---

## 2. Gates

| Gate | Result |
|---|---|
| G0 overfit50 | **PASS** — CE 1.40→0.89; best val 4-AUC **0.878** (ep5) |
| G1 S0 parity | **PASS** — test 4-AUC **0.6464** = v1 S0 **0.646** |
| G2 traj non-deg (S_pc) | **PASS** — test Pearson **0.246**, beats mean |
| G3 teacher H2 | **PASS / GO** — pearson **0.301**, beats mean; lin 0.678 / mlp **0.662** / knn **0.574** (val∩aux) |
| G3b conflict | **no conflict** — early cos median **+0.159** → UW **skipped** (plan lock) |
| G5 D1 | **PASS** — 0.7359 vs freeze 0.7378 (Δ −0.0019 ≤ 0.01) |

---

## 3. Cell A — PCGrad MTL (`b4v2_mtl_20260715`)

| Arm | balancer | val 4-AUC | test 4-AUC | test bin | traj Pearson |
|---|---|---:|---:|---:|---:|
| **S0** | none | 0.682 | **0.646** | 0.642 | 0.144 |
| **S_pc** λ=1 | pcgrad | 0.687 | **0.631** | 0.636 | **0.246** |

### S_pc − S0 (test, paired boot)

| Δ4-AUC | 95% CI | lo>0? |
|---:|---|---|
| **−0.016** | **[−0.033, −0.000]** | **No (hurts)** |

**Interpretation:** Traj head improves with multi-task (Pearson↑) but PCGrad does **not** convert that into class AUC. Gradients were **co-aligned** (cos&gt;0) — the textbook PCGrad premise (conflict) was **absent**, so the cell is a **mechanism-null** plus a controlled negative on 4-AUC.

UW not run (G3b).

---

## 4. Cell B — RKD distill H2 (`b4v2_rkd_20260715`)

Teacher H2 (wear→cgm): best val RMSE **1.137**, Pearson **0.301**, beats mean. Probe GO.

| μ | val 4-AUC | test 4-AUC | test bin | train-aux z cosine |
|---:|---:|---:|---:|---:|
| 0 | 0.685 | **0.645** | 0.642 | 0.151 |
| 0.3 | 0.679 | **0.647** | 0.652 | 0.050 |
| 1.0 | 0.681 | **0.627** | 0.616 | 0.153 |

### μ − μ0 boot

| μ | Δ | CI | lo>0? |
|---:|---:|---|---|
| 0.3 | +0.002 | [−0.019, +0.022] | **No** |
| 1.0 | **−0.019** | **[−0.035, −0.001]** | **No (hurts)** |

### RKD student z ∥ C1 vs D1

| Arm | test 4-AUC | Δ vs D1 | lo>0? |
|---|---:|---:|---|
| D1 | **0.736** | — | — |
| Dμ0+C1 | 0.719 | −0.017 | No |
| Dμ0.3+C1 | 0.715 | −0.021 | No |
| Dμ1+C1 | 0.715 | −0.021 | No |

**RKD bar: fail.** Alignment weak vs v1 L2 cosine (0.09→0.64); RKD does not force pointwise match and does not raise ranking.

---

## 5. Cell B′ — CRD sensitivity (`b4v2_crd_20260715`)

| μ | test 4-AUC | Δ vs μ0 | lo>0? |
|---:|---:|---:|---|
| 0 | 0.637 | — | — |
| 0.3 | 0.635 | −0.002 | No |
| 1.0 | 0.645 | +0.008 | No |

**CRD bar: fail** (objective family null).

---

## 6. Cell C — fusion (`b4v2_hybrid_20260715`)

Embeddings: S0 from `b4v2_mtl_20260715` (class-only). F1 = **μ=0 OOF only** (labeled; not RKD-μ OOF).

| Arm | test 4-AUC | test bin | Δ vs D1 | lo>0? |
|---|---:|---:|---:|---|
| **D1** | **0.736** | 0.809 | — | — |
| **F0b** z-only GBM | **0.635** | 0.659 | −0.101 | No (CI entirely &lt;0) |
| **F2** frozen z∥C1 | **0.717** | 0.812 | −0.019 | No |
| **F1** OOF z∥C1 (μ0) | **0.726** | 0.798 | −0.010 | No |

**F1 ≥ F2** (OOF reduces dilution vs frozen) but **still &lt; D1**.  
**F0b** shows sequence z alone is **not** orthogonal high-SNR ranking signal (~0.64, near pure neural S0).  
**F3 FiLM residual (post-claim critique):** pre-registered trigger **fired** (F1 **0.72615** &gt; D1−0.01 **0.72591** and F1 ≥ F2 **0.717**) but F3 was **not run**. Disclosure only — same-budget counterfactual still null (F1 ceiling 0.726 makes CI lo&gt;0 vs D1 0.736 unreachable in expectation; FiLM overfit-kill would likely trip). Not a silent win; freeze stands with this residual noted.

---

## 7. Decision bars (pre-registered)

| Bar | Outcome |
|---|---|
| PCGrad vs S0 CI lo&gt;0 | **Fail** (hurts) |
| RKD μ&gt;0 vs μ0 | **Fail** |
| CRD μ&gt;0 vs μ0 | **Fail** |
| Best hybrid vs D1 | **Fail** |
| Teacher non-deg + probe GO | **Pass** |
| Null accept (gates pass, bars fail) | **Yes — valid V2 close** |

---

## 8. What V2 is / is not

**Is:** Critiqued sibling retry of B4 with different formulations (RKD/CRD, PCGrad, OOF fusion); leakage-safe distill scope; matched D1; paired boots; honest null with healthy H2 teacher.

**Is not:** Reopen of L2-μ / plain-λ grids; logit-KD (B3); full SSL; per-fold RKD OOF ambition; deployable C1 raise; proof that all LUPI is impossible.

---

## 9. Interpretation (brief)

1. **Wear→curve SNR wall (~0.30)** is real and stable (matches B4-A / H2 hard). Teacher probes beat chance but do not imply student 4-AUC lift.  
2. **No task-gradient conflict** on this encoder → PCGrad had nothing to project; multi-task still fails transfer (same pattern as plain-λ).  
3. **Relational/contrastive KD** does not fix the ranking gap when the privileged structure is recon-shaped and weak.  
4. **OOF-z reduces but does not eliminate** C1 dilution; z is not additive enough to clear D1.  
5. Residual high-value path: **SSL backbone** (`PLAN_SSL.md`). B3 logit-KD already **frozen null** (`REPORT_B3.md`). Do **not** reopen B4-V2 RKD/CRD/PCGrad grids on cold PatchCNN.

---

## 10. Reproduce

```bash
export HIP_VISIBLE_DEVICES=0
# full ladder script (reference)
bash training/path_b/b4/artifacts/b4v2_ladder_20260715.sh

# or stages:
.venv/bin/python -m training.path_b.b4 --run-id b4v2_overfit50_YYYYMMDD \
  --mode neural --lambdas 0 --max-participants 50 --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4v2_teacher_h2_YYYYMMDD \
  --mode teacher_probe --teacher-mode wear_cgm --run-probe --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4v2_mtl_YYYYMMDD \
  --mode mtl_bal --balancer pcgrad --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4v2_rkd_YYYYMMDD \
  --mode distill_hybrid --distill-objective rkd --teacher-mode wear_cgm \
  --run-probe --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4v2_hybrid_YYYYMMDD \
  --mode hybrid_v2 --emb <S0 embeddings.npz> --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4v2_crd_YYYYMMDD \
  --mode distill --distill-objective crd --teacher-mode wear_cgm \
  --run-probe --device cuda
```

Artifacts: `training/path_b/b4/artifacts/b4v2_*/`.

---

## 11. Package / locks

- Plan: `PLAN_B4_V2.md`  
- Code: `training/path_b/b4/` (V2 modes + losses)  
- Decisions: `DECISIONS.md` (incl. post-claim critique 2026-07-16)  
- Frozen v1 reports remain authoritative for L2 / plain-λ nulls  

---

## 12. Post-claim critique disposition (2026-07-16)

Critiquer `opencode-go/glm-5.2:high` (fresh) re-verified artifacts vs this report.

| Item | Disposition |
|---|---|
| Overall | **approve-with-caveats** — null authentic for arms run |
| Blockers (leakage / OOF / D1 / teacher GO) | **None** |
| Same-budget counterfactual (F3 + H1 + per-fold RKD OOF) | **Would not flip** lo&gt;0 bars |
| Caveat 1 | F3 FiLM trigger fired, not run — residual disclosure |
| Caveat 2 | Per-fold RKD-μ OOF deferred; F1 is class-only as labeled |
| Caveat 3 | Insulin 2× / multi-view CRD deferred — CRD null is lower-bound |
| Reopen B4-V2? | **No** |
| Next | **`PLAN_SSL.md`** (or optional tiny F3 residual plan for hygiene only) |
