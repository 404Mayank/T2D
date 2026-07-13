# Plan — Phase A wrap (REVISED post-critique)

> Last Path A tabular batch before Path B. Research paper focus.  
> Plan critique (glm-5.2): **revise** → incorporated below.  
> **Await user go before implement/run.**

---

## Critique incorporation

| ID | Fix |
|---|---|
| O1 minimal rule ad-hoc | **Exact rule below** (dual rank, not fake “intersection”); add `hr_min`; drop negative-perm-only SHAP pets unless dual-rank qualifies |
| O2 circularity | Keep val-importance → test eval; **state as limitation** in REPORT |
| O3 tolerances | Unify: retain if paired bootstrap Δ(minimal−C1) **AUC CI includes 0** OR point ΔAUC ≥ −0.01 **and** point Δbin ≥ −0.015; report both |
| O4 skip-on-test | **Run all pre-registered runs** (no test-conditional skip). Optional extras only if pre-listed |
| O5 missing ablation | Add **E5 watch+mood (no onboarding)** |
| O6 ceiling framing | Rename; **complications-only** severity add-on + separate optional “all clinical” stack |
| O7 count collinear | **No engineered count** in ceiling |
| O8 binary rank metric | **val binary AUC**, tie val AUPRC |
| O9 C1 vs minimal | Prefer **minimal if retention rule holds**, else C1 |
| O10 | Report train binary base rate; keep balanced weights; note in REPORT |

---

## 0. Locked baselines

| ID | Run | 4-AUC | Binary |
|---|---|---:|---:|
| W0 | full_20260713_221240 | 0.6662 | 0.6889 |
| A1 | onboarding_20260713_224744 | 0.6987 | 0.7492 |
| C1 | mood_scores_20260714_014415 | 0.7378 | 0.8309 |

Protocol: n=1824, splits, weights balanced, seed 42, 50 trials, val-select, freeze-before-test.

---

## 1. Minimal feature sets (E1) — primary wrap deliverable

### Selection rule (pre-registered, checkable)

From **frozen C1** val importances (`catboost_1c_scores_*` CSVs):

1. Rank features by SHAP (desc) and by permutation (desc) separately.  
2. Assign combined_score = rank_shap + rank_perm (lower better).  
3. **Exclude** `cestl` from candidates (construct: dead vs PAID; already established).  
4. **E1a `minimal_S`:** top **12** by combined_score among remaining.  
5. **E1b `minimal_M`:** top **18** by combined_score.  
6. Publish the ranked table in `artifacts/wrap_feature_ranks.json` before HPO.

**Fallback if code ranks differ slightly from hand list:** always trust the CSV recompute at run start (same files).

### Hand-check expected core (illustrative; runtime CSV is authority)

Likely includes: `paidscore`, `whr_vsorres`, `fh_dm2pt`, `fh_dm2sb`, `hr_mean`/`rhr`, `waist`, `hip`, `sri`, `hr_min`, `bmi`, …  
Will **not** force `hr_cv`/`rar_amplitude` if dual-rank excludes them (perm-negative).

### Retention rule vs C1 (on test, once)

Keep minimal as preferred secondary model if:
- point ΔAUC (min − C1) ≥ **−0.01**, **and**
- point Δ binary ≥ **−0.015**, **and**
- report paired bootstrap Δ; if CI entirely below 0 on AUC → **fail retention** even if point within tolerance.

**Paper pick:** if E1a retains → **minimal_S**; elif E1b retains → **minimal_M**; else **C1**.

**Limitation (mandatory REPORT text):** features chosen from C1 val importance; optimistic bias toward retention.

---

## 2. Full run list (all pre-registered — no test-gated skips)

| ID | Name | Features | Question |
|---|---|---|---|
| **E2a** | `paid_only` | A1 matrix − cestl + paidscore only | CES drop |
| **E2c** | `ces_only` | A1 + cestl only (no paidscore) | Negative control |
| **E1a** | `minimal_S` | top-12 dual-rank | Parsimony |
| **E1b** | `minimal_M` | top-18 dual-rank | Parsimony |
| **E5** | `watch_mood` | 30 GREEN + paidscore (+ optional cestl? **paidscore only**) | Mood without onboarding |
| **E3a** | `severity_addons` | C1 matrix + `mhoccur_rnl` + `mhoccur_circ` only | True complications ceiling |
| **E3b** | `clinical_upper` | C1 + hbp + clsh + rnl + circ (no count) | Broad clinical upper bound (label honestly) |
| **E4a** | `bin_watch` | GREEN, y=label>0 | Binary floor |
| **E4b** | `bin_c1` | full C1 matrix, binary y | Binary deployable |
| **E4c** | `bin_min_s` | minimal_S features, binary y | Binary minimal |
| **E4d** | `bin_severity` | E3a features, binary y | Binary severity stack |

**Out:** diet, E2b paid items (unless E2a fails badly — still skip; ces_only covers construct), re-HPO of C1.

---

## 3. Binary protocol

- y = `(label > 0).astype(int)`  
- LGBM `objective=binary`, CatBoost `Logloss`  
- **Val rank metric:** binary AUC; tie: binary AUPRC  
- Report AUC, AUPRC, Brier; compare to multiclass-derived `1−P0` from W0/C1  
- Prefer binary-primary in paper table only if test AUC − derived ≥ **+0.01**

---

## 4. Run order (fixed)

```text
0. Build wrap_feature_ranks.json from C1 SHAP/perm CSVs → lock E1a/E1b lists
1. E2a paid_only
2. E2c ces_only
3. E1a minimal_S
4. E1b minimal_M
5. E5  watch_mood
6. E3a severity_addons
7. E3b clinical_upper
8. E4a bin_watch
9. E4b bin_c1
10. E4c bin_min_s
11. E4d bin_severity
```

~11 runs × ~10–15 min ≈ **2–3 h** sequential.

---

## 5. Implementation

```text
run_wrap.py --exp paid_only|ces_only|minimal_s|minimal_m|watch_mood|severity|clinical_upper|bin_watch|bin_c1|bin_min_s|bin_severity
build_minimal_ranks.py   # writes wrap_feature_ranks.json
REPORT_A_WRAP.md         # final consolidation
```

Reuse loaders/HPO/metrics/bootstrap/explain from path_a_watch + path_a_blocks.

---

## 6. Paper deliverables

1. Ladder table + wrap experiments  
2. Minimal S/M vs C1  
3. PAID-only vs CES-only vs C1  
4. watch+mood vs C1 (onboarding value)  
5. Severity vs clinical upper  
6. Binary-primary vs derived  
7. SHAP figures  
8. **Path A frozen** recommendation block  

---

## 7. Frozen recommendation rule (pre-registered)

| Role | Rule |
|---|---|
| Headline watch-only | W0 |
| Secondary tabular (screening-ish) | minimal_S if retains else minimal_M if retains else C1 |
| Severity-enriched upper | E3a (report E3b as broader upper) |
| Binary reporting | derived unless E4 improves ≥0.01 |
| Next | **Path B** |

---

## 8. User defaults (locked unless you change)

1. Dual-rank minimal rule — **yes**  
2. Drop cestl in minimal/paid_only — **yes**  
3. Add ces_only + watch_mood — **yes**  
4. Severity = rnl+circ only for E3a; E3b broader — **yes**  
5. No diet — **yes**  
6. Run all 11 — **yes**  

---

## 9. Checklist

- [x] Plan draft  
- [x] Plan critique  
- [x] Revise plan  
- [x] User go  
- [x] Implement + code critique  
- [x] Run all  
- [x] REPORT_A_WRAP + freeze Path A  
