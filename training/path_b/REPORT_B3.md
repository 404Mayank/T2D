# B3 final report — logit knowledge distillation (Diasense-style baseline)

**Status:** **B3 CONCLUDED** (2026-07-15).  
**Protocol:** `PLAN_B3.md` (critiqued → revised → implemented).  
**Authority for claims:** this report + `DECISIONS.md` (2026-07-15 B3 entries).  
**Claim run:** `b3_grid_20260715`  
**Supporting:** `b3_smoke_20260715` (non-claim); `b3_tsens_t1_20260715` / `b3_tsens_t4_20260715` (T sens); `b3_nfix_20260715` (strict Hinton re-fit).

Path A numbers are **frozen** and **unchanged**. C1 **0.7378** / binary **0.8309** / AUPRC **0.4687**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **deployable logit-KD beat C1**? | **No.** Decision arm `G_α=0.3` (T=2) test 4-AUC **0.7469** vs D1 **0.7378**; Δ **+0.009** CI **[−0.0086, +0.0273]** lo≯0 → **ambition fail** |
| Does exact Hinton MLP KD help? | **No.** `N_α=0.3`−N0 Δ **−0.002** CI includes 0 (grid + post-fix) |
| Teacher privilege real? | **Yes.** Tch **0.8227** vs D1a **0.7463**; Δ **+0.076** CI **[+0.041, +0.114]** |
| OOF teacher usable? | **Yes** (mean fold val AUC **0.786** > D1a val **0.744** + 0.01) |
| G0 protocol? | **Pass** (\|G0 − D1_LGBM\| = **0**) |
| D1 vs frozen C1? | **Exact match** (0.7378 / 0.8309 / 0.4687) |
| T∈{1,4} at α=0.3? | Still null (point Δ ≈ +0.005, CI includes 0) |
| Path B deployable beat C1? | **Still no** across B1/B2/B4/B3 |

**Scientific takeaway:** Soft class-logit KD from a CGM-privileged teacher **does not** clear the pre-registered bar against matched C1 under this recipe. Privilege remains large (teacher ≈ B2 oracle O1). Point estimates at α∈{0.3,0.5} sit ~+0.9 pp above D1 but bootstrap CIs include 0; binary is **worse** than D1. Completes the LUPI-KD comparison: B4-B L2-z null **and** B3 logit-KD null.

**Narrative 2×2 (pre-registered):** G null + N null → **logit-KD cell closed null** under this recipe.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Teacher X | C1 (47) + 8 true `cgm_*_daymean` |
| Teacher pool | train∩aux (1184); eval aux val/test |
| Soft labels | K=5 OOF LGBM on train∩aux; T=2 primary |
| Student X | C1 only (full core 1824) |
| Non-aux train | hard CE only (no soft) |
| Ambition arm | LightGBM soft-row expansion **`G_α=0.3`, T=2** vs D1 |
| Science arm | MLP Hinton **`N_α=0.3` vs N0** (not beat-C1) |
| α grid | {0, 0.3, 0.5, 1.0} at T=2 (sensitivities non-claim for max) |
| Seed / boot | 42 / n=2000 |
| Package | `training/path_b/b3/` |

**No re-clean. No new FE.**

---

## 2. Runs

| Run id | Role | Claimable? |
|---|---|---|
| `b3_smoke_20260715` | pipeline smoke (400 train, 5 trials) | no |
| **`b3_grid_20260715`** | claim grid α∈{0,0.3,0.5,1}, T=2 | **yes** (GBM + teacher; Nα pre-fix) |
| `b3_tsens_t1_20260715` | T=1, α=0.3 | sensitivity |
| `b3_tsens_t4_20260715` | T=4, α=0.3 | sensitivity |
| `b3_nfix_20260715` | strict Hinton fix re-fit N0/`N_α=0.3` | **yes for N science** |

### Impl critique disposition (post-smoke / mid-grid)
| ID | Item | Disposition |
|---|---|---|
| OB-1 | MLP non-aux CE not scaled by (1−α) | **Accept** — fixed before `b3_nfix`; grid N* labeled pre-fix |
| OB-2 | OOF always LGBM vs family-selected Tch | **Accept as documented** — OOF = LGBM; Tch selected CatBoost; gate still pass vs D1a |
| OB-3 | OOF early-stop on fold holdout | **Accept caveat** — standard CV-predict; not re-run |
| OB-4 | G vs D1 confounds LGBM-KD vs CatBoost-hard | **Accept diagnostic** — report G vs D1_LGBM/G0 below |
| OB-5 | KL term unweighted at high α | **Accept report note** |

---

## 3. Teacher & OOF

| Arm | pool | family | test 4-AUC | test binary |
|---|---|---|---:|---:|
| **Tch** | aux | catboost | **0.8227** | **0.8768** |
| **D1a** | aux | lightgbm | 0.7463 | 0.8248 |

- Tch−D1a Δ4-AUC **+0.0764** CI **[+0.0409, +0.1135]** → headroom **pass** (aligns B2 O1 **0.8227**).
- OOF mean val AUC **0.7858 ± 0.0189** vs D1a val **0.7444** → gate **pass**.
- Teacher val Brier/ECE recorded in `teacher_metrics.json`.
- Soft labels: train∩aux n=1184; non-aux train receive **zero** teacher soft mass.

---

## 4. Claim results (test, raw ranking)

### 4.1 Hard baselines & GBM KD (`b3_grid_20260715`)

| Arm | 4-AUC | Binary | AUPRC | Notes |
|---|---:|---:|---:|---|
| **D1** | **0.7378** | **0.8309** | **0.4687** | ≡ frozen C1 |
| G0 | 0.7383 | 0.8275 | 0.4776 | pinned D1-LGBM; protocol **pass** |
| **G_α=0.3** | **0.7469** | 0.8169 | 0.4922 | **decision arm** |
| G_α=0.5 | 0.7460 | 0.8157 | 0.4884 | sensitivity |
| G_α=1.0 | 0.7351 | 0.8060 | 0.4877 | sensitivity |

### 4.2 Decision bars

| Bar | Result |
|---|---|
| **User ambition** `G_α=0.3`−D1 | Δ **+0.0091** CI **[−0.0086, +0.0273]** lo≯0 → **fail** (soft note: point > +0.005) |
| Binary Δ `G_α=0.3`−D1 | **−0.014** CI **[−0.036, +0.007]** (includes 0) — multi-AUC point up, screening binary point **down** |
| Beats frozen C1 4-AUC (point) | yes (+0.009) — **not** bar-pass without CI |
| Beats frozen C1 binary | **no** (0.817 < 0.831) |
| **KD science** `N_α=0.3`−N0 (grid pre-fix) | Δ **−0.0022** CI includes 0 → **fail** |
| **KD science** (`b3_nfix`, strict Hinton) | Δ **−0.0020** CI includes 0 → **fail** (same) |
| Teacher headroom | **pass** |
| OOF usable | **pass** |
| G0 protocol | **pass** |

### 4.3 Diagnostic (non-claim): family-matched KD

`G_α=0.3` vs **G0 / D1_LGBM** (same family):  
Δ4-AUC **+0.0086** CI **[−0.0054, +0.0231]** lo≯0 → still **null**.  
So the ambition fail is **not** only “LGBM-KD vs CatBoost-hard.”

### 4.4 Neural science (test 4-AUC)

| Arm | grid (pre-fix loss) | nfix (strict Hinton) |
|---|---:|---:|
| N0 | 0.7136 | 0.7136 |
| N_α=0.3 | 0.7114 | 0.7116 |
| N_α=0.5 / 1.0 | 0.708 / 0.706 | — |

MLP remains below trees; Hinton does not raise N0.

### 4.5 Temperature sensitivity (`G_α=0.3` only)

| T | G test 4-AUC | Δ vs D1 | CI lo>0 |
|---:|---:|---:|---|
| 1 | 0.7423 | +0.0045 | no |
| **2** (primary) | **0.7469** | **+0.0091** | no |
| 4 | 0.7427 | +0.0049 | no |

No T recovers the bar.

---

## 5. Decision bars (summary table)

| Question | Pass? |
|---|---|
| Deployable logit-KD beats C1 (`G_α=0.3` CI lo>0) | **No** |
| Exact Hinton helps (`N_α=0.3`−N0) | **No** |
| Teacher ceiling real (Tch−D1a) | **Yes** |
| OOF teacher usable | **Yes** |
| Protocol G0 / D1 freeze parity | **Yes** |

---

## 6. Interpretation

1. **Dark knowledge did not transfer into a deployable raise.** Soft targets from a strong CGM teacher (0.82 4-AUC) produce at best a **non-significant** ~1 pp 4-AUC bump and **hurt binary** vs C1 (binary Δ **−0.014**).
2. **Teacher headroom is feature-driven, not soft-logit-rich.** Tch ≡ B2 O1 (0.8227): the ~8 pp comes from *seeing* true CGM features. Soft labels are mild (raw max-prob mean ~0.57 → T=2 ~0.43) — a weak 4-dim channel vs C1 already at 0.74. Same bottleneck family as B2 (oracle real; handoff null).
3. **Val/test cross-flip on the decision arm:** `G_α=0.3` **val** 4-AUC **0.7348 < D1 val 0.7439**, while test is +0.009 the other way. Val-based selection would **not** pick G over D1; the test point sits inside CI noise, not a robust ranking gain.
4. **α=0.3/0.5 ≈ each other on test; α=1 collapses** — pure soft labels without enough hard CE are worse. α=0.5 is sensitivity only (not the ambition bar).
5. **Not a Diasense reproduction** — different n, split (`recommended_split` vs random k-fold), feature stack (C1 vs their 41+27), and models. Controlled baseline only.
6. **Limitations (pre-registered + audit):**
   - Gα vs G0 does not separate teacher dark knowledge from generic soft-label regularisation; shuffled-soft not run (no headline raise).
   - Soft arm also gets **fresh HPO on ~4× expanded rows** (4829 vs 1277) — confounds KD with HPO-on-augmented-data; zero-α row-duplication control was contingent on a raise and was not triggered.
   - OOF soft labels are **LGBM-fold** (mean val ~0.79) while headline Tch is **CatBoost** (val ~0.81); student sees a slightly weaker teacher than the ceiling number. Gate still passes; directionally biases toward (not against) a null.
   - B3 D1a (0.746) is not bit-identical to B2 D1a (~0.729) under separate HPO; Tch 0.8227 is shared. Cross-package D1a variance is expected at this n.

---

## 7. Path B ladder close-out

| Stage | Deployable beat C1? | Notes |
|---|---|---|
| B1 multi-task | **No** | pure-seq ~0.65; λ null |
| B2 two-stage Ŷ | **No** | T1 ≤ D1; O1 oracle +0.09 |
| B4-A traj MTL | **No** | hybrid ≤ D1 |
| B4-B rep-distill | **No** | easy+hard null/hurt |
| **B3 logit-KD** | **No** | this report |

**Honest frozen claim:** under AI-READI wearable_core + C1 stack, **naive LUPI handoffs** (scalar MTL, point-estimate Ŷ-glu, traj MTL, L2-z distill, logit-KD) **do not beat** tuned watch+onboarding+mood CatBoost; **privilege is real** (oracle/teacher). Stronger recipes (CRD/RKD, PCGrad MTL, SSL backbone) remain explicit future work — not silent reopen of closed cells.

---

## 8. Artifacts

```
training/path_b/b3/artifacts/b3_grid_20260715/
  decision_bars.json  arm_summaries.json  g0_protocol.json
  teacher_metrics.json  oof_soft_labels.parquet
  arms/{D1,D1a,Tch,G0,G_a=0.3,G_a=0.5,G_a=1,N0,N_a=*}
  compare_*.json  run_manifest.json  run.log
```

Package: `python -m training.path_b.b3 --run-id ...`

---

## 9. Next

Path B ladder **complete** for planned cells. Optional later (new `PLAN_*` only): CRD/RKD, SSL, gradient-surgery MTL, Path A leftovers (diet/CORN). **Do not** reopen B1–B4 claim grids without a plan.
