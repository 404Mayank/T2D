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
