# Path A progress report — diagnostics + 1A onboarding

**Date:** 2026-07-13  
**Packages:** `training/path_a_watch/` (watch-only floor), `training/path_a_blocks/` (deployable blocks)  
**Roadmap:** `PATH_AHEAD.md` · **Decisions log:** `DECISIONS.md`  
**Local run artifacts (gitignored):**  
- Floor: `path_a_watch/artifacts/full_20260713_221240/`  
- Diagnostics: `path_a_blocks/artifacts/diag_20260713_224549/`  
- 1A: `path_a_blocks/artifacts/onboarding_20260713_224744/`

---

## 1. Executive summary

| Track | Model | Test 4-AUC | Test binary AUC | Status |
|---|---|---:|---:|---|
| **Watch-only floor** (paper claim baseline) | CatBoost Ordered, 30 GREEN | **0.666** | **0.689** | Locked |
| **1A deployable** | CatBoost + 15 onboarding | **0.699** | **0.749** | **decision_bar_pass=True** |

- Onboarding clears the pre-registered decision bar (Δ 4-AUC **+0.033**, paired bootstrap lo **> 0**, onboarding perm stable).  
- Binary moves toward the honest band (~0.78–0.82) but is still short (~0.75).  
- 4-class remains below the honest band (~0.72–0.75).  
- SHAP is **mixed** (5/10 watch, 5/10 onboarding) — survey-dominance guardrail **not** tripped.  
- Healthy vs pre-diabetes remains nearly chance (pairwise 0.52); model still separates **extremes** better than mid severity.

---

## 2. Protocol (both packages)

- Cohort: wearable_core **n=1824**; train/val/test **1277 / 270 / 277** via `recommended_split`  
- No `clinical_site` / label / pool flags as features  
- Class weights: LGBM `balanced`, CatBoost `Balanced` (locked before importance)  
- HPO: 50 Optuna trials/family; ES = multi_logloss; rank by val macro-OVR AUC (AUPRC tie)  
- **Val-select family → immutable freeze → test once**  
- Primary metrics: 4-class macro-OVR AUC + macro AUPRC; binary = `1 − P(y=0)`  
- Calibration: diagnostic only (raw ranking is the claim)  
- 1A decision bar (three criteria):  
  1. Point ΔAUC > +0.01 vs watch floor  
  2. Person-bootstrap CI on paired ΔAUC has lower bound > 0  
  3. Onboarding block has stable positive permutation importance  

---

## 3. Watch-only floor (recap)

| Metric | Test |
|---|---:|
| macro-OVR AUC | 0.666 |
| macro AUPRC | 0.392 |
| binary AUC | 0.689 |
| QWK | 0.417 |
| Per-class OVR | 0: 0.689 · 1: 0.648 · 2: **0.565** · 3: **0.763** |

- physio_only ablation (drop coverage counts): ΔAUC ≈ 0  
- Top SHAP: `hr_cv`, `rar_amplitude`, `rhr`, MVPA, SRI  
- Honest literature bands: 4-class ~0.72–0.75, binary ~0.78–0.82 — **missed**

---

## 4. Diagnostics (floor model)

Recomputed floor test AUC matches reference bit-exactly (**0.6662**).

### Pairwise OVR-style AUCs (test)

| Pair | AUC | Read |
|---|---:|---|
| 0 vs 1 healthy ↔ pre | **0.521** | ≈ chance |
| 1 vs 2 pre ↔ oral | 0.675 | moderate |
| 2 vs 3 oral ↔ insulin | 0.627 | weak–moderate |
| 0 vs 3 healthy ↔ insulin | **0.821** | strong |
| 0 vs 2 | 0.701 | |
| 1 vs 3 | 0.801 | |

### Bootstrap CI (test, 1000 person resamples)

| Metric | 95% CI | Point |
|---|---|---:|
| 4-AUC | 0.628 – 0.704 | 0.666 |
| binary | 0.611 – 0.757 | 0.689 |

### Confusion (test, argmax)

```
true\pred   0   1   2   3
0          31  22  12   7
1          22  28  12   9
2          16  17  16  15
3           6  10  18  36
```

Class 2 is diffusely misclassified; class 3 has the strongest diagonal.

### Site-stratified test

| Site | n | 4-AUC | binary | Label mix note |
|---|---:|---:|---:|---|
| UAB | 98 | 0.651 | 0.691 | more insulin/oral |
| UCSD | 64 | **0.609** | **0.572** | weakest |
| UW | 115 | 0.656 | 0.675 | more healthy |

Site×label confounding remains a limitation; do not use site as a feature.

---

## 5. Phase 1A — watch + hard onboarding

**Features:** 30 GREEN + 15 onboarding  
`age`, BMI, waist/hip/WHR, height, weight, BP×2, `fh_dm2pt`, `fh_dm2sb`, pulse×2  
(Dropped: zero `age_years_at_interview`, high-missing `fh_*sp`; smoking unavailable this release.)

**Selected:** CatBoost (val AUC 0.722 vs LGBM 0.717)

### Test metrics

| Metric | Floor | **1A** | Δ |
|---|---:|---:|---:|
| macro-OVR AUC | 0.666 | **0.699** | **+0.0325** |
| binary AUC | 0.689 | **0.749** | **+0.0603** |
| macro AUPRC | 0.392 | **0.412** | +0.020 |
| QWK | 0.417 | 0.472 | +0.055 |
| Class 0 OVR | 0.689 | 0.749 | +0.060 |
| Class 1 OVR | 0.648 | 0.663 | +0.016 |
| Class 2 OVR | 0.565 | 0.591 | +0.026 |
| Class 3 OVR | 0.763 | 0.791 | +0.028 |

### Decision bar

| Criterion | Result |
|---|---|
| c1 point ΔAUC > 0.01 | **True** (+0.0325) |
| c2 paired bootstrap ΔAUC lo > 0 | **True** (lo ≈ +0.0085, hi ≈ +0.062) |
| c3 onboarding perm stable | **True** (mean onboard perm > 0; 9/15 in top half) |
| **decision_bar_pass** | **True** |

1A test 4-AUC bootstrap CI: **0.662 – 0.733** (point 0.699).  
Binary bootstrap CI: **0.680 – 0.812** (point 0.749).

### SHAP / permutation (val)

**SHAP top-10 (block tags):**  
watch: `hr_cv`, `rhr`, `hr_nocturnal_dip`, `hr_sd`, `sri`  
onboarding: `whr_vsorres`, `fh_dm2pt`, `waist_vsorres`, `bmi_vsorres`, `fh_dm2sb`  

- Onboarding in top-10: **5/10**  
- All-onboarding guardrail: **False**  

**Permutation top:** WHR, FH parent, hr_cv, FH sibling, waist, SRI, BMI, RHR, …

---

## 6. Interpretation

1. **Deployable Path A works.** Hard onboarding is a real, statistically supported lift under the fixed split and three-criterion bar.  
2. **Paper watch-only claim is unchanged** at 0.666 — still the honest wearable-only reference for Path B.  
3. **Severity-as-4-class remains hard.** Class 2 and 0↔1 separation limit macro AUC; binary screening is the stronger deployable story so far.  
4. **Not pure survey.** Autonomic/circadian watch features remain co-equal with anthropometrics/FH in SHAP — good for a mixed deployable narrative; still monitor if later mood/comorbidity blocks swamp wearables.  
5. **Honest bands:** binary ~0.75 is closer to 0.78–0.82; 4-class 0.70 still short of 0.72–0.75.

---

## 7. Next steps (from PATH_AHEAD)

## 9. Phase 1B comorbidity (2026-07-14)

Locked set: hbp, clsh, mi, strk, cvdot, ra + count; bar vs 1A; required no_hbp sensitivity.

| | 1A | **1B_core** | **1B_no_hbp** |
|---|---:|---:|---:|
| Test 4-AUC | 0.699 | **0.7085** | **0.7109** |
| Binary | 0.749 | **0.778** | **0.769** |
| ΔAUC vs 1A | — | +0.0098 | +0.0123 |
| Bootstrap Δ lo>0 | — | **No** | **No** |
| decision_bar_pass | True (vs floor) | **False** | n/a |

**Verdict (core):** comorbidity risk-factor checklist does **not** clear the bar vs 1A.

**1B plus_complications** (kidney+circulation): test 4-AUC **0.724**, binary **0.790**, Δ+0.025 with bootstrap lo>0 — numerical pass but **not** claim set (consequence markers).

## 10. Phase 1C mood (2026-07-14)

Primary `cestl`+`paidscore` on 1A. Run `mood_scores_20260714_014415`.

| | 1A | **1C** | Δ |
|---|---:|---:|---:|
| 4-AUC | 0.699 | **0.738** | **+0.039** |
| Binary | 0.749 | **0.831** | **+0.082** |
| decision_bar_pass | — | **True** | |

PAID drives lift (perm 0.038 vs cestl 0.0006). Binary enters honest band (~0.78–0.82); 4-class meets lower literature edge (~0.72–0.75).

**Deployable stack:** watch + onboarding + mood scores. Watch-only paper floor still **0.666**.

### Next steps (superseded by wrap)

See **§11 Phase A wrap** — Path A frozen; next is Path B.

---

## 11. Phase A wrap (2026-07-14) — Path A frozen

Full write-up: **`REPORT_A_WRAP.md`**. Pick: `artifacts/wrap_paper_pick.json`.

| Role | Frozen choice |
|---|---|
| Headline watch-only | W0 **0.666 / 0.689** |
| Secondary tabular | **C1** 0.738 / 0.831 (minimal_S/M failed retention) |
| Binary in tables | multiclass-derived `1−P0` (dedicated binary HPO never +0.01) |
| Severity stack | E3a narrative only; no 4-AUC lift over C1 |
| Next | **Path B** |

Key wrap facts: **paid_only ≈ C1**; **ces_only** fails; **watch+PAID** without onboarding 0.718; binary HPO underperforms derived scores.

---

## 8. How to reproduce

```bash
export DRI_PRIME=1
# Floor (already run)
.venv/bin/python -m training.path_a_watch --run-id <id>

# Diagnostics
.venv/bin/python -m training.path_a_blocks.diagnostics \
  --floor-run full_20260713_221240 --run-id diag_<id>

# 1A / 1B / 1C
.venv/bin/python -m training.path_a_blocks.run_1a --run-id onboarding_<id>
.venv/bin/python -m training.path_a_blocks.run_1b --feature-set core
.venv/bin/python -m training.path_a_blocks.run_1c --feature-set scores

# Wrap (all pre-registered)
.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --all
```

Deps: `training/path_a_watch/requirements.txt` (or project `.venv` with those pins).
