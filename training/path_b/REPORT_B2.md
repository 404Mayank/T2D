# B2 final report — two-stage glucose-emulator ablation

**Status:** **B2 CONCLUDED** (2026-07-15).  
**Protocol:** `PLAN_B2.md` (critiqued → revised → implemented).  
**Authority for claims:** this report + `DECISIONS.md` (2026-07-15 B2 entries).  
**Run id (claim):** `b2_grid_20260715`  
**Smoke (non-claim):** `b2_smoke_20260715`

Path A numbers are **frozen** and **unchanged**. W0 **0.6662** / bin **0.6889**; C1 **0.7378** / **0.8309** / AUPRC **0.4687**.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **any deployable B2 arm beat C1**? | **No.** Best deployable = **D1 ≡ C1** (0.7378 / 0.8309). T1 is **worse** (0.7345 / 0.814). |
| Does **predicted** person-CGM help vs matched direct? | **No** — T1−D1 Δ4-AUC **−0.003** CI **[−0.019, +0.011]**; binary Δ **−0.017** CI entirely &lt; 0 |
| Does predicted CGM help on watch-only? | **No** — T0−D0 Δ **+0.004** CI **[−0.015, +0.024]** |
| Does **oracle** (true CGM features) beat C1 numbers? | **Yes on raw score** — O1 **0.823** / **0.877** — but **not deployable** and **aux pool** (n_test=254), not a C1 replacement claim |
| Matched oracle headroom (fair) | O1−D1a **+0.094** CI **[+0.058, +0.130]** |
| Stage-1 bottleneck? | **Yes** — val mean R² **~0.047** |
| Protocol parity D1 vs frozen C1? | **Exact match** |
| Proceed Path B ladder? | **Yes → B4 → B3** |

**Scientific takeaway:** Modular two-stage handoff of **point-estimate** person CGM **does not** improve T2D over matched direct C1/W0 when Stage-1 is watch GREEN → daymean 8-vector (R²≈0.05). The **oracle** arm proves ~9 pp 4-AUC headroom if true CGM were available at Stage-2 — so LUPI is not empty; the **emulator** is the failure mode. This **motivates B4** (trajectory / rep-distill), not more scalar two-stage tuning.

### What is still worth noting (not claim raises)
1. **D0/D1 exact freeze match** — B2 Stage-2 plumbing is trustworthy; Δs are not HPO artifacts.
2. **Oracle ceiling is large** — person CGM 8-vector is highly T2D-informative when *true*; path is Stage-1 / representation, not “CGM doesn’t matter.”
3. **T1 slightly hurts binary** vs D1 — weak Ŷ is not neutral noise under this recipe.
4. **B1 + B2 both null on scalar CGM summaries** (multi-task day-level and two-stage person-level) — convergent negative on the *summary handoff* cell.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Stage-1 X | W0 GREEN (30) |
| Stage-1 Y | 8 `cgm_*_daymean` from `cgm_person` |
| Stage-1 model | 8× LightGBM regressors; OOF K=5; non-aux train Ŷ = mean of K folds |
| Stage-1 fit pool | train ∩ aux (1184) |
| Stage-2 family | CatBoost + LightGBM, Path A HPO spaces, val-select AUC→AUPRC |
| Class weights | LGBM `balanced` / Cat `Balanced` |
| Calibration | val sigmoid primary (+ isotonic secondary); ranking = raw |
| Claim pool D/T | wearable_core 1824 (1277/270/277) |
| Claim pool O/D1a | aux 1685 (1184/247/254) |
| Seed | 42; bootstrap n=2000 |
| HPO | Stage-1 30 trials/target; Stage-2 50 trials/family |

**No re-clean. No new FE.** Package: `training/path_b/b2/`.

---

## 2. Runs

| Run id | Role | Claimable? |
|---|---|---|
| `b2_smoke_20260715` | pipeline smoke (200 train, 5 trials) | no — sanity only |
| **`b2_grid_20260715`** | full arms D0,D1,T0,T1,D1a,O1 | **yes** |

---

## 3. Stage-1 emulator quality

Val (aux) mean R² **0.0466**; test mean R² **0.0148**. Gate (val R²>0 on mean/sd/tar): **PASS**.

| Target | val R² | val RMSE | test R² |
|---|---:|---:|---:|
| cgm_mean_daymean | 0.085 | 39.5 | (see artifact) |
| cgm_sd_daymean | 0.075 | 10.9 | |
| cgm_cv_daymean | −0.025 | 0.049 | |
| cgm_min_daymean | 0.057 | 27.9 | |
| cgm_max_daymean | 0.087 | 55.0 | |
| cgm_tir_70_180_daymean | 0.048 | 0.229 | |
| cgm_tbr_70_daymean | −0.003 | 0.016 | |
| cgm_tar_180_daymean | 0.049 | 0.232 | |

OOF fold insulin counts ~13–14 / fold (ok). Ŷ drift table: `artifacts/.../yhat_drift.json`.

**Read:** Stage-1 is non-degenerate but **near-null** as a glucose emulator from person GREEN — consistent with pre-plan HistGB probes (~0.07).

---

## 4. Stage-2 results (`b2_grid_20260715`)

### 4.1 Arm table (test, raw ranking)

| Arm | Pool | n_feat | Family | 4-AUC | Binary | Macro AUPRC | Brier |
|---|---|---:|---|---:|---:|---:|---:|
| **D0** | core | 30 | CatBoost | **0.6662** | **0.6889** | 0.3916 | 0.697 |
| **D1** | core | 47 | CatBoost | **0.7378** | **0.8309** | **0.4687** | 0.651 |
| **T0** | core | 38 | LightGBM | 0.6706 | 0.6812 | 0.3916 | — |
| **T1** | core | 55 | CatBoost | **0.7345** | **0.8141** | 0.4728 | — |
| **D1a** | aux | 47 | LightGBM | 0.7289 | 0.8298 | 0.4674 | 0.671 |
| **O1** | aux | 55 | CatBoost | **0.8227** | **0.8768** | **0.5946** | 0.575 |

### 4.2 Protocol parity (critical)

| Compare | 4-AUC | Binary | AUPRC |
|---|---:|---:|---:|
| Frozen Path A W0 | 0.6662 | 0.6889 | 0.3916 |
| B2 **D0** | 0.6662 | 0.6889 | 0.3916 |
| Frozen Path A C1 | 0.7378 | 0.8309 | 0.4687 |
| B2 **D1** | 0.7378 | 0.8309 | 0.4687 |

D1−freeze AUC ≈ **0** (within 1e-5). Matched baseline is **not** drifted → user-bar fallback not triggered.

### 4.3 Pre-registered comparisons (paired person bootstrap, n=2000, seed 42)

| Contrast | Δ4-AUC point | 95% CI | lo>0? | Verdict |
|---|---:|---|---|---|
| **T1 − D1** | −0.0033 | [−0.0187, +0.0110] | No | **ablation fail** |
| **T0 − D0** | +0.0044 | [−0.0148, +0.0242] | No | **ablation fail** |
| **O1 − D1a** | **+0.0938** | **[+0.0583, +0.1299]** | **Yes** | **oracle headroom pass** |

Binary side: T1−D1 Δbin point **−0.0168**, CI **[−0.035, −0.0003]** (excludes 0 on the **low** side) — predicted CGM **harms** binary ranking slightly under this recipe.

### 4.4 Decision bars applied

| Bar | Outcome |
|---|---|
| Stage-1 R² gate | **Pass** |
| T1 vs D1 ablation | **Fail** |
| T0 vs D0 ablation | **Fail** |
| User ambition beat C1 | **Fail** (T1 0.7345 < 0.7378; bin 0.814 < 0.831) |
| Oracle headroom ≥ +0.02 | **Pass** (+0.094) |
| Kill pivot (O1−D1a < +0.01) | **Not triggered** |
| B4 blocked? | **No** |

---

## 5. Interpretation

1. **Two-stage predicted CGM is a null (or slightly harmful) add-on** to an already strong C1 stack. Weak Stage-1 Ŷ adds noise/collinear proxies rather than glycemic signal.
2. **True person-level CGM is highly informative** on the matched aux pool (O1 ~0.82 4-AUC). Headroom is real → LUPI is not dead; **handoff representation** is wrong at person daymean resolution.
3. **D0/D1 exact freeze match** validates B2 Stage-2 plumbing against Path A — internal Δs are trustworthy.
4. Relative to **B1**: multi-task day-level glu was also null; two-stage scalar handoff is also null. Both point away from “add 8 CGM summaries somehow” and toward **B4 trajectory / rep-distill**.
5. Do **not** claim “CGM aux doesn’t help T2D” — oracle refutes that. Claim: **this modular point-estimate emulator does not help deployable AUC.**

---

## 6. What B2 is / is not

**Is:**
- Controlled ablation of LUPI point-estimate handoff vs matched direct  
- Proof of large oracle ceiling for person CGM on C1 stack  
- Protocol-matched re-fit of W0/C1 under B2 package  

**Is not:**
- A deployable raise over C1  
- Evidence against B4 trajectory teachers  
- Logit-KD (B3) or multi-task (B1)  
- A claim that O1 “beats C1 for deployment” (privileged + aux pool)  

---

## 7. Residual open cells — rule in/out?

**Primary B2 cell is closed.** No further run is required to claim: *person-level predicted daymean CGM two-stage does not beat C1 under Path A–matched Stage-2.*

| Optional sensitivity | Would it change the freeze? | Recommendation |
|---|---|---|
| Stage-1 X = C1 (age/BMI/…) | May raise R² without being a *watch* emulator; still unlikely to close +9 pp oracle gap | **Skip** unless paper needs “even with clinical Stage-1” footnote |
| Daily `watch_daily` → person glu Stage-1 | Different cell; might improve Ŷ modestly | **Done as B2-V2** (`REPORT_B2_V2.md`) — still null |
| Variance-propagated stacking / reduced Y | Address Ŷ noise / collinearity | **Done as B2-V2** — T1v/T1p still ≯ D1 |
| Freeze Path A C1 hyperparams for T1 | D1 already matches freeze exactly under re-HPO | **Skip** |
| Heavier Stage-1 (deeper nets, more trials) | Unlikely: signal from person GREEN to CGM is weak (R²&lt;0.1) | **Skip** as B2 claim work |
| B1 `z` → Stage-2 GBM hybrid | New experiment, not B2 | Separate plan if ever |

**Update 2026-07-16:** B2 residual knobs (daily grain, variance pack, reduced Y) were run as sibling **`b2v2_grid_20260716`** — deployable **still null** (T1v 0.727 ≯ C1); oracle +0.096. See `REPORT_B2_V2.md`. Do not reopen frozen B2 HPO.

**To fully “rule in” a deployable CGM-aux win over C1 you need a different method** (not more B2 tabular knobs; B4 traj/rep-distill also null under recipes run). Oracle already **rules in** that *true* person CGM features help; predicted handoff **rules out** modular summary recipes tested.

---

## 8. Implications for ladder

| Next | Implication |
|---|---|
| **B4** | Ran after B2 freeze — traj/rep-distill **also null** for deployable C1 raise |
| **B3** | Ran — logit-KD **null** vs C1 (`REPORT_B3.md`) |
| **B2-V2** | Residual knobs ran — **still null** (`REPORT_B2_V2.md`) |
| B2 further | **Frozen** — no silent reopen |

---

## 9. Reproduce

```bash
# smoke
.venv/bin/python -m training.path_b.b2 --run-id b2_smoke_YYYYMMDD --quick

# full claim grid
.venv/bin/python -m training.path_b.b2 --run-id b2_grid_YYYYMMDD \
  --n-trials 50 --stage1-n-trials 30
```

Artifacts: `training/path_b/b2/artifacts/b2_grid_20260715/`  
(`arm_summaries.json`, `decision_bars.json`, `stage1_metrics.json`, `compare_*.json`, `arms/*/`).

---

## 10. Package

```
training/path_b/b2/
  config.yaml  data.py  stage1.py  stage2.py  evaluate.py  run.py
```

Critique-before-run: no blockers; medium fixes applied (assert_no_leakage wired; smoke min-class guard).
