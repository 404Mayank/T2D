# Path B2 — two-stage glucose-emulator ablation

**Date:** 2026-07-15  
**Status:** **IMPLEMENTED & CONCLUDED** 2026-07-15 — claim run `b2_grid_20260715`; see `REPORT_B2.md`.  
**Role:** ablation — modular LUPI handoff (point-estimate CGM → T2D). Not paper headline (B4 is).  
**Authority:** `Training.md` §2 metrics / §4 B2 / §7; Path A freeze `REPORT_A_WRAP.md`; B1 freeze `REPORT_B1.md` + `DECISIONS.md`.  
**User ambition bar:** beat Path A **C1** (test 4-AUC **0.7378** / binary **0.8309** / macro AUPRC **0.4687**).

**Critique:** critiquer → **revise** (2026-07-15). Disposition in §8.

---

## 0. Data readiness verdict

### Verdict
**No re-clean. No new FE required for B2 v1.** Existing processed assets are sufficient.

| Asset | Status | B2 use |
|---|---|---|
| `features/cgm_person.parquet` | **ready** 1924×12; 0 nulls on daymeans; **aux_eligible 1685/1685** covered | Stage-1 targets (true CGM person 8-vector) |
| `features/watch_green.parquet` | ready 1824×31; frozen Path A GREEN | Stage-1 X (watch-only) + Stage-2 base watch block |
| `features/onboarding.parquet` / `mood.parquet` | ready; Path A C1 stack | Stage-2 deployable base (match C1) |
| `meta/pool_masks.parquet` | ready; splits 1277/270/277 core; aux 1184/247/254 | pools + labels + split |
| `features/cgm_daily.parquet` / `watch_daily.parquet` | ready (post sleep-unit fix) | **not primary for B2 v1**; optional later / B4 |

### Why no pipeline change
- Person CGM aggregates already shipped with B1 FE (`cgm_*_daymean` over valid days; aux `n_valid_days` ≥7).
- Sleep unit bug was **watch_daily only** (fixed 2026-07-15); `watch_green` path was already correct — Path A numbers stay frozen.
- B4’s 5-min multi-modal grid is **still missing** — B4 work, not B2.
- Weak Stage-1 R² (probe ~0.07 on `cgm_mean`) is a **modeling** risk, not missing columns. B2’s role is the *point-estimate handoff* ablation; daily FE does not change that scientific cell.

### Empirical anchors (session probes; **not claim numbers**)
| Probe (aux pool, HistGB, fixed split; rough) | test 4-AUC | notes |
|---|---:|---|
| W0 GREEN only | ~0.61 | below frozen CatBoost W0 0.666 |
| C1-ish tabular | ~0.71 | below frozen CatBoost C1 0.738 |
| True CGM 8-vector only | ~0.73 | privileged signal is real |
| C1 + true CGM (oracle) | ~0.79 | ~+8 pp headroom if Stage-1 perfect |
| Stage-1 C1→`cgm_mean` R² | ~0.07 | hard regression |

**Implication:** beating frozen C1 with **predicted** CGM is unlikely under person-level tabular Stage-1. Plan separates **ambition bar** (user: beat C1) from **ablation pass** (T vs matched D). A clean null + oracle headroom is an informative Path B result and motivates B4.

---

## 1. Goal

### Scientific question
Does a **two-stage** pipeline (wearable → **point-estimate** person CGM → T2D) improve T2D discrimination over the **same** Stage-2 base **without** predicted CGM?

### Claims B2 may make
1. **Primary ablation:** predicted-CGM two-stage vs **matched** direct Stage-2 base (ΔAUC + paired boot CI) on full `wearable_core`.
2. **Oracle ceiling (matched pool):** true-CGM Stage-2 vs direct on **aux-only** train/eval pools (same n) — headroom for the handoff idea.
3. **User ambition bar:** deployable T1 vs **matched D1** first; frozen Path A C1 as **external anchor** (see §2.8 fallback if D1 drifts).

### Claims B2 is not
- Multi-task (B1 frozen null), logit-KD (B3), trajectory teacher (B4).
- Re-open of Path A survey blocks / C1 sensitivities.
- License to change `watch_green` or Path A claim numbers.
- “LUPI works” from oracle alone.

---

## 2. Design locks

### 2.1 Pipeline shape

```
Stage-1 (glucose emulator; fit only on train ∩ aux_eligible):
  X1 = watch_green (30 GREEN numerics)   # primary; C1-X Stage-1 = sensitivity only
  Y1 = cgm_person 8 daymeans
  → ĝ → ŷ_glu ∈ R^8

Stage-2 (T2D; CatBoost + LightGBM, Path A protocol family):
  X2 = base_block ∪ {ŷ_glu | y_glu_true (oracle only)}
  Y2 = label ∈ {0,1,2,3}
```

**Deployable inference:** `ŷ_glu = ĝ(watch_green)` → Stage-2. **Never** true CGM at infer.

### 2.2 Stage-1 targets

```
cgm_mean_daymean, cgm_sd_daymean, cgm_cv_daymean,
cgm_min_daymean, cgm_max_daymean,
cgm_tir_70_180_daymean, cgm_tbr_70_daymean, cgm_tar_180_daymean
```

- Fit Stage-1 **only** on `wearable_core ∧ aux_eligible` with targets (all 1685 aux; train slice ~1184).
- Collinearity OK (same as B1).
- **Forbidden** as targets or Stage-2 features: `n_valid_days`, `cgm_n_total`, `n_days` (coverage / wear-duration leakage).

### 2.3 Feature blocks

| Block | Columns | Role |
|---|---|---|
| **W0** | 30 GREEN numeric (`watch_green` minus `person_id`) | Stage-1 primary X; Stage-2 watch base |
| **C1** | W0 + Path A `onboarding_keep` + `paidscore` + `cestl` | Stage-2 primary base (user bar stack) |
| **Ŷ_glu** | 8 predicted daymeans | deployable handoff |
| **Y_glu** | 8 true daymeans | oracle only |

**C1 manifest lock:** load onboarding keep-list and mood score cols from `training/path_a_blocks/config.yaml` at runtime **and** snapshot the resolved column list into `b2/artifacts/<run_id>/c1_feature_manifest.json`. Smoke asserts n_feat and names match frozen C1 (wrap: 47 features). Do not silently track future edits to Path A config without a B2 re-pin note in DECISIONS.

**Primary Stage-1 X = W0 only.** Stage-1 on C1 features = **sensitivity only** (age/BMI can inflate R² without being a watch emulator).

### 2.4 Leakage rules (hard)

1. **Stage-1 never fit on val/test persons.**
2. **Train Ŷ for Stage-2 (full core, including non-aux):**
   - Run **K=5** stratified (by label) OOF **on train ∩ aux** only for fitting Stage-1 folds.
   - For each fold *k*: fit Stage-1 on aux-train \ fold_k; predict **(a)** aux persons in fold_k **and (b) all non-aux train persons** with that same fold model; average the K predictions for each non-aux train person (so non-aux Ŷ uses models of the same train-size regime as OOF aux Ŷ, not one full-data Stage-1).
   - **Lock:** non-aux train Ŷ = mean of K fold-models’ predictions (each fold-model never saw that fold’s aux holdout; non-aux were never Stage-1 targets).
3. **Val/test Ŷ:** single Stage-1 fit on **full train ∩ aux**; apply to all core val/test pids (aux and non-aux).
4. Never put `label`, `recommended_split`, `clinical_site`, pool flags in X.
5. Oracle arms use **true** CGM only; no median-fill of true CGM into non-aux in any **primary** O* table (see §2.6).
6. Outer claim split = fixed `recommended_split` only.

### 2.5 Models

| Stage | Primary | Notes |
|---|---|---|
| **Stage-1** | **8× LightGBM regressors** (one model per glu dim) | Clean per-target OOF; independent heads. CatBoost multi-output = **sensitivity only** (document if run). |
| **Stage-2** | CatBoost + LightGBM multiclass, Path A family | Val-select by macro-OVR AUC; tie → macro AUPRC within `auc_tie_eps=0.005`. |

**Class weights (exact Path A spelling):**
- LightGBM: `class_weight="balanced"`
- CatBoost: `auto_class_weights="Balanced"`

**HPO:** reuse **identical search spaces** from `training/path_a_blocks/config.yaml` → `hpo.lightgbm` / `hpo.catboost`. Stage-2 primary = re-HPO under B2 (~50 trials/family, seed 42) so D1 is an internal matched baseline. If wall-time binds: freeze frozen-C1 hyperparams for D1/T1 as documented fallback.

Stage-1 HPO: ≤30 trials maximizing val mean R² over the 8 dims (or multi-RMSE); seed 42.

**Do not** use B1 LSTM as primary B2 backbone.

### 2.6 Arms (pre-registered)

| ID | Pool | Stage-2 features | CGM | Deployable? | Role |
|---|---|---|---|---|---|
| **D0** | full core 1824 | W0 | none | yes | matched watch direct |
| **D1** | full core 1824 | C1 | none | yes | **matched C1 direct** (primary ablation baseline) |
| **T0** | full core 1824 | W0 + Ŷ_glu | pred | yes | two-stage watch |
| **T1** | full core 1824 | C1 + Ŷ_glu | pred | yes | **primary deployable two-stage** |
| **D1a** | **aux-only** 1685 | C1 | none | yes* | **matched oracle baseline** (same train n as O1) |
| **O1** | **aux-only** 1685 | C1 + Y_glu | true | **no** | oracle ceiling (matched to D1a) |
| **O0** | **aux-only** 1685 | W0 + Y_glu | true | **no** | oracle on watch base (matched; optional vs D0a if needed) |
| **D0a** | **aux-only** 1685 | W0 | none | yes* | matched watch direct for O0 (run if reporting O0) |

\*D1a/D0a are deployable in the trivial sense (no true CGM) but are **protocol baselines for oracle matching**, not the user-bar deployable story (user-bar uses full-core T1).

**Dropped from primary:** median-fill of true CGM onto non-aux “full-core oracle.” Forbidden as a primary O* number. Optional sensitivity only if ever needed for narrative — label **non-claim**.

**Primary comparisons**
1. **T1 vs D1** (full core) — does predicted CGM help on C1 stack?  
2. **T0 vs D0** (full core) — watch-only handoff story.  
3. **O1 vs D1a** (aux-only, **matched train/eval n**) — privileged headroom / kill-pivot.  
4. **User ambition:** T1 vs D1; frozen Path A C1 as external anchor (§2.8).  
5. Report Stage-1 R²/RMSE table always.

### 2.7 Metrics (Training.md §2 aligned)

| Metric | Use |
|---|---|
| Macro-OVR **4-AUC** | primary rank |
| Macro **AUPRC** | co-required under imbalance |
| Binary AUC (`1−P0`) + binary AUPRC | screening report |
| Per-class OVR AUC | diagnostic |
| **Brier** (multiclass / binary) | required with AUC |
| **Calibration** | **required** val-fold step: default **sigmoid**; secondary **isotonic** if min_pos ≥ 30 (same as `path_a_blocks/config.yaml` calibration block). Report curves + Brier raw vs calibrated. Ranking metrics on **raw** scores remain the ablation claim (Path A practice); calibrated numbers co-reported. |
| Stage-1 R² / RMSE per dim + mean R² | emulator quality |
| Paired person bootstrap Δ (n=2000, seed 42) | T1−D1, T0−D0, O1−D1a |

### 2.8 Decision bars

| Question | Pass rule |
|---|---|
| Predicted CGM helps (ablation) | T1−D1 test Δ4-AUC **> 0** and paired boot CI **lo > 0**. Soft note if point > +0.005 but CI includes 0. |
| Watch-only handoff | same rule for T0−D0. |
| **User ambition: beat C1** | **Primary:** T1 vs **D1** (internal). **External anchor:** frozen C1 (0.7378 / 0.8309 / AUPRC 0.4687). **Fallback lock:** if `D1 < frozen_C1 − 0.01` on 4-AUC, **do not** treat “T1 > frozen C1” as the sole user-bar story — report T1 vs D1 as the fair bar and frozen C1 as unreproduced external reference; diagnose feature/HPO parity. If D1 matches freeze within 0.01, also report paired Δ vs frozen scores when available. |
| Oracle headroom | **O1−D1a** point Δ4-AUC **≥ +0.02** → handoff has headroom; Stage-1 is bottleneck (expected). |
| Kill / pivot (person-8vec privilege weak on C1) | **O1−D1a** point Δ4-AUC **&lt; +0.01** under matched aux pool → weak ceiling; still write REPORT; B4 still proceeds (dynamics motivation). |
| Stage-1 smoke gate | val mean R² on `{mean, sd, tar}` **> 0**. If all ≤ 0, fix Stage-1 before full grid. |

**B2 does not block B4** regardless of pass/fail.

---

## 3. Protocol details

### 3.1 Cohorts (verified)

| Pool | train | val | test | train insulin |
|---|---:|---:|---:|---:|
| wearable_core | 1277 | 270 | 277 | 80 |
| core ∩ aux_eligible | 1184 | 247 | 254 | 69 |
| core \ aux (non-aux) | 93 | 23 | 23 | 22 total non-aux insulin |

- **D\*/T\*:** full wearable_core.  
- **D1a / O1 / D0a / O0:** aux-only (matched).  
- Non-aux always get Ŷ in T\* (deployable full-core story).

### 3.2 Diagnostics (required in smoke / report)
1. **Ŷ drift:** percentile table of each Ŷ dim on val/test split by `aux_eligible` vs not (aux-trained Stage-1 may shift on non-aux).  
2. **OOF fold class counts:** print label counts per K-fold (insulin ~14/fold expected on aux-train).  
3. **C1 manifest assert:** feature list vs frozen wrap n_feat=47.  
4. **D1 vs frozen C1:** absolute Δ4-AUC; trigger §2.8 fallback if >0.01 low.

### 3.3 Seeds & HPO
- Global seed **42**; bootstrap n=2000 seed 42.  
- Stage-1 ≤30 trials; Stage-2 ~50 trials/family (smoke: 5 trials, ~200 train pids).

### 3.4 Package layout (implement when approved)

```
training/path_b/b2/
  config.yaml          # pins paths, manifests, HPO ref to path_a_blocks
  data.py              # blocks, OOF Ŷ incl. non-aux rule, deny cols
  stage1.py            # 8× LGBM (+ optional CatBoost multi sens)
  stage2.py            # Path A-style multiclass train/select
  evaluate.py          # 4-AUC, AUPRC, Brier, cal, bootstrap Δ
  run.py / __main__.py
  artifacts/<run_id>/
```

Reuse `path_a_watch` / `path_a_blocks` metrics/HPO helpers — **no drive-by refactors**.

### 3.5 Run ladder
1. **Smoke** `b2_smoke`: subset train, 5 trials; arms D1, T1, D1a, O1; gates §2.8 Stage-1 + fold counts + Ŷ drift.  
2. **Stage-1 full** `b2_s1_<date>`: persist OOF train Ŷ + val/test Ŷ + Stage-1 metrics.  
3. **Stage-2 grid** `b2_grid_<date>`: D0, D1, T0, T1, D1a, O1 (+ D0a/O0 if cheap).  
4. `REPORT_B2.md` + `DECISIONS.md` + `path_b/README.md`.

---

## 4. Out of scope (B2 v1)

- Re-clean / re-FE GREEN or daily matrices  
- B1 backbone / λ grids  
- B3, B4 5-min grid  
- Survey blocks beyond C1  
- Changing Path A frozen numbers  
- Primary claims from median-filled “oracle”  
- Daily→person Stage-1 (defer; different cell — do not over-read person-level null as “FE won’t help”)

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Stage-1 R² ~0 → T≈D | Pre-registered; publish null; O1−D1a shows ceiling |
| D1 ≠ frozen C1 | Manifest + HPO pin; §2.8 fallback (T1 vs D1 primary) |
| Train-pool confound on oracle | **D1a matched to O1** |
| Non-aux Ŷ leakage / asymmetry | K-fold mean prediction rule §2.4 |
| Non-aux Ŷ distribution shift | Required drift table |
| Collinear glu dims | Trees OK; optional drop-cv sensitivity |
| User bar too high | Ambition ≠ ablation pass |

---

## 6. Critique checklist — answers (post-disposition)

1. **OOF:** Completed via non-aux K-fold-mean rule (§2.4).  
2. **C1 vs W0:** Both arms kept; C1 for user bar; W0 for cleaner watch science.  
3. **Oracle pool:** Matched aux-only O1 vs D1a; no primary median-fill.  
4. **Daily Stage-1:** Deferred; person-level v1 ships; null does not kill daily FE later.  
5. **Bars:** Strict CI lo>0 for ablation; Path A +1pp block rule **not** imported.  
6. **Cleaning:** Still **no** re-clean required.

---

## 7. Implementation gate

**Done.** Smoke `b2_smoke_20260715` + full `b2_grid_20260715`. Outcome: predicted two-stage **null**; oracle headroom **pass**; user C1 bar **fail**. Package frozen unless new plan.

---

## 8. Critique disposition (2026-07-15)

| # | Critique item | Disposition | Action in plan |
|---|---|---|---|
| 1 | Oracle train-pool asymmetry (O1 vs D1) | **Accept (blocker)** | Add **D1a** (aux-only); primary ceiling = O1−D1a |
| 2 | Non-aux train Ŷ unspecified | **Accept (blocker)** | K-fold models predict non-aux; mean over K |
| 3 | AUPRC + calibration required | **Accept** | §2.7 aligned with Training.md §2 / Path A |
| 4 | Median-fill oracle contamination | **Accept** | Forbidden as primary; aux-only O* |
| 5 | Stage-1 family ambiguity | **Accept** | Primary **8× LGBM**; CatBoost multi = sens |
| 6 | Ŷ drift diagnostic | **Accept** | Required smoke/report table |
| 7 | HPO space + class_weight spelling | **Accept** | Pin to path_a_blocks config |
| 8 | User-bar fallback if D1 ≪ freeze | **Accept** | §2.8: T1 vs D1 primary when drift |
| 9 | OOF fold insulin counts | **Accept** | Smoke check |
| 10 | C1 manifest pin | **Accept** | Snapshot + assert |
| — | Daily Stage-1 mandatory | **Reject** | Out of scope v1; not blocking |
| — | Import Path A +1pp block bar | **Reject** | Different purpose; keep CI lo>0 |
| — | Re-clean / B1 reopen / B4 FE for B2 | **Reject** | Data sufficient; ladder intact |

**Verdict after disposition:** plan **ready for implement** pending user approval.
