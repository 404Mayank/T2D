# Path A blocks — decisions log

> Living log for `training/path_a_blocks/`. Roadmap: `PATH_AHEAD.md`.  
> Watch-only floor package: `training/path_a_watch/`.

## Locks (2026-07-13)

- New package (not overwrite path_a_watch artifacts)
- Order: **diagnostics → 1A onboarding**; stop and reassess before 1B
- Floor reference: `path_a_watch/artifacts/full_20260713_221240` test 4-AUC **0.666**
- Decision bar: ΔAUC > +0.01 + bootstrap CI + stable perm importance
- No `clinical_site` as feature; no sex/race (unavailable)
- Onboarding keep list: age, anthropometrics, BP, fh_dm2pt/sb, pulse; drop zero age_years_at_interview and high-missing *sp fields for v1
- Smoking: not present in onboarding.parquet this release
- Class weights unchanged from watch floor
- HPO: 50 trials/family, val-select, freeze-before-test (same protocol)

## Chronology

| When | Event |
|---|---|
| 2026-07-13 | Package created; PATH_AHEAD.md written |
| 2026-07-13 | Critiquer (glm-5.2): revise — full 3-criterion decision bar + floor assert + paired Δ bootstrap |
| 2026-07-13 | Diagnostics `diag_20260713_224549` complete |
| 2026-07-13 | **1A complete** `onboarding_20260713_224744` — decision_bar_pass=True |

## Diagnostics summary (`diag_20260713_224549`)

Floor CatBoost recomputed test 4-AUC **0.6662** (bit-match reference).

| Pairwise test AUC | |
|---|---:|
| 0 vs 1 (healthy vs pre) | **0.521** (≈ chance) |
| 1 vs 2 | 0.675 |
| 2 vs 3 | 0.627 |
| 0 vs 3 | **0.821** |
| Bootstrap test 4-AUC 95% CI | 0.628 – 0.704 |

**Read:** severity ladder fails at healthy↔pre; extremes separable. Class-2 remains weak.

## 1A results (`onboarding_20260713_224744`)

| | Watch floor | **1A watch+onboarding** | Δ |
|---|---:|---:|---:|
| Test 4-AUC | 0.666 | **0.699** | **+0.0325** |
| Test binary AUC | 0.689 | **0.749** | **+0.060** |
| Test macro AUPRC | 0.392 | **0.412** | +0.020 |
| Val-selected | CatBoost | CatBoost | |
| Class-2 OVR | 0.565 | **0.591** | +0.026 |
| Class-3 OVR | 0.763 | **0.791** | +0.028 |

**Decision bar:** c1 point ΔAUC>0.01 **True**; c2 bootstrap ΔAUC lo>0 **True**; c3 onboarding perm stable **True** → **`decision_bar_pass=True`**.

**SHAP guardrail:** 5/10 top features onboarding (WHR, FH parent/sibling, waist, BMI); 5/10 watch (hr_cv, rhr, nocturnal dip, hr_sd, sri). **Not** all-survey — mixed deployable signal.

**Interpretation:** Onboarding clears the pre-registered bar and moves binary toward the honest band (0.78–0.82 still short). 4-class still below 0.72–0.75. Next per PATH_AHEAD: **1B comorbidity (HTN-first)** or stop and document deployable config.

## 1B results (locked plan post-critique)

User locks: arthritis **in**; HBP **with + without** runs; rnl/circ **sensitivity-only** (not run yet).

| Run | Test 4-AUC | Binary | ΔAUC vs 1A | Δbin vs 1A | c1 | c2 | c3 | bar |
|---|---:|---:|---:|---:|---|---|---|---|
| **1B_core** `comorb_core_20260714_010558` | **0.7085** | **0.778** | **+0.0098** | +0.029 | False | False | True | **False** |
| **1B_no_hbp** `comorb_no_hbp_20260714_011831` | **0.7109** | **0.769** | **+0.0123** | +0.020 | True | False | True | n/a (not primary) |

- Parent 1A: 0.6987 / 0.7492. Floor: 0.666 / 0.689.
- Core: LGBM selected (val 0.733); point ΔAUC **just under** +0.01 bar; bootstrap Δ CI includes 0.
- no_hbp: CatBoost selected; point Δ clears +0.01 but bootstrap still overlaps 0 — HBP not required for the small point lift; lift is **not** statistically bar-passing.
- Class-2 Δ vs 1A: core **+0.053**, no_hbp +0.024 (soft positive).
- SHAP (core): comorbidity present but not dominant (2/10 top); watch+onboarding still lead.

**Interpretation:** Comorbidity checklist gives a **small, non–bar-passing** lift over 1A (~+0.01 4-AUC, ~+0.03 binary). Binary ~0.78 touches the lower edge of the honest band; 4-class still short of 0.72. **Do not add comorbidity to the deployable stack under the pre-registered bar.** Optional: plus_complications / ge5pct for completeness only.

### 1B plus_complications (kidney+circulation sensitivity)

Run `comorb_plus_complications_20260714_012753` (CatBoost):

| | vs 1A |
|---|---:|
| Test 4-AUC | **0.724** (Δ **+0.025**) |
| Binary | **0.790** (Δ **+0.041**) |
| Bootstrap Δ lo | **+0.0036** (c2 True) |
| c1∧c2∧c3 | True |
| decision_bar_pass | **False** (not claim set) |

**Read:** Adding self-report kidney/circulation (consequence-proxies) **would** clear the numerical bar vs 1A, but primary excluded them as severity complications (like PDR). Documents a real trade-off: **screening purity vs severity signal**. Deployable claim without complications remains 1A; with complications is a separate “severity proxy” stack, not the risk-factor checklist.

## 1C mood results (`mood_scores_20260714_014415`)

Primary: `cestl` + `paidscore` on 1A baseline. Plan+code critique done; **decision_bar_pass=True**.

| | 1A | **1C scores** | Δ |
|---|---:|---:|---:|
| Test 4-AUC | 0.699 | **0.738** | **+0.039** |
| Binary | 0.749 | **0.831** | **+0.082** |
| Bootstrap Δ AUC lo | — | **>0** (pass c2) | |
| Family | CatBoost | CatBoost | |

- c1 True, c2 True, c3 True → **bar pass**
- Per-feature perm: `paidscore` **0.038**, `cestl` **0.0006** — PAID carries the block (as critique predicted)
- Class-2 Δ: **+0.046**
- SHAP: paidscore in top-10; watch still majority of top features (guardrail not all-survey)

**Deployable stack now:** watch + onboarding + **mood scores** (1A+1C). Comorbidity risk-factor checklist **not** included. Complications optional narrative only.

## Phase A wrap (2026-07-14) — Path A frozen

Plan: `PLAN_A_WRAP.md` / `PLAN_A_WRAP_IMPL.md`. Code: `build_minimal_ranks.py`, `run_wrap.py`.  
Report: **`REPORT_A_WRAP.md`**. Pick: `artifacts/wrap_paper_pick.json`.

### Dual-rank minimal (from C1 SHAP+perm; exclude cestl)

- **minimal_S (12):** paidscore, whr_vsorres, fh_dm2sb, fh_dm2pt, rhr, waist_vsorres, hr_mean, bmi_vsorres, weight_vsorres, sri, stress_sd, hr_min  
- **minimal_M (18):** + hip_vsorres, hr_nocturnal_dip, hr_n, sleep_short_frac, age, pulse_vsorres_2  

### Wrap test results (CatBoost selected on all runs)

| Exp | 4-AUC | Binary | vs C1 |
|---|---:|---:|---|
| paid_only | 0.7366 | 0.8329 | ≈ C1 |
| ces_only | 0.7030 | 0.7425 | large drop |
| minimal_S | 0.7090 | 0.8126 | **retain False** |
| minimal_M | 0.7245 | 0.8294 | **retain False** (ΔAUC −0.013) |
| watch_mood | 0.7179 | 0.8087 | onboarding still needed |
| severity (rnl+circ) | 0.7369 | 0.8287 | no 4-class lift over C1 |
| clinical_upper | 0.7357 | 0.8411 | binary +0.01; 4-AUC flat |
| bin_* (4 runs) | — | 0.677–0.824 | all **below** multiclass-derived; no binary-primary |

### Locks from wrap

- **Headline watch-only:** W0 **0.666 / 0.689**
- **Secondary tabular:** **C1** (minimal failed pre-registered retention)
- **Binary tables:** multiclass-derived `1−P0` (no E4 ≥ +0.01)
- **PAID** ≈ full mood block; **CES** negative control confirmed
- **Path A tabular frozen** → next **Path B** only (no diet / no reopening 1B as claim without new protocol)
