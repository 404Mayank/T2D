# Plan — C1 sensitivity: smoking / obesity / via

> Optional **post-freeze** survey sensitivities. Parent = **C1** (always original C1, never updated mid-batch).  
> Not required for Path A freeze; does not reopen 1B core.  
> Plan critique (glm-5.2): **revise** → incorporated below.  
> **Executed 2026-07-14:** all four fail decision bar; **C1 unchanged**. See DECISIONS.md.

## Design choice

**Independent first, then one joint.** No full combinatorial / pairwise grid.

| ID | Name | Add-on features | Bar-eligible |
|---|---|---|---|
| S1 | `smoke` | `smoke_ever`, `smoke_current` | yes (primary keep candidate) |
| S2 | `obs` | `mhoccur_obs` | yes (expect null: BMI-redundant) |
| S3 | `via` | `via1`, `via2`, `via3` | yes (severity-adjacent framing) |
| S4 | `all3` | union of S1–S3 | **joint ceiling only** — cannot alone expand C1 |

**Keep rule:** same 3-criterion decision bar vs **original C1**  
(point ΔAUC > +0.01 ∧ paired bootstrap ΔAUC lo > 0 ∧ new-block perm stable).

**c3 perm stable (small blocks):** mean perm-AUC-drop over **new** features > 0 **and** ≥1 new feature with perm > 0  
(same rule as 1C mood block; works for 1–3 features).

**Paper pick for secondary:**  
- Start from frozen C1.  
- Expand only by features whose **independent** run (S1/S2/S3) bar-passes.  
- S4 pass without singleton pass → report only, **do not** expand C1.  
- S4 parent is always original C1 (not C1+winners).

**Multiple comparisons:** 4 post-hoc tests, no formal Bonferroni. Framing = exploratory sensitivities;  
the decision bar is already conservative (Δ>+0.01 + CI lo>0). Do not claim “new deployable stack”  
without independent pass.

## Feature defs

### Smoking (manual extract — FE gap fix)
Pipeline prefixes (`smoking`/`smok`) miss AI-READI codes (`susmk*`).  
**One-off extractor** `build_smoking_features.py` → `data/processed/features/smoking.parquet`  
(does not require full re-clean).

- `smoke_ever` ← code `susmkncf` (0/1; **777→NA** only sentinel observed)
- `smoke_current` ← `susmkcdur` when ever=1; **0 when ever=0**; NA if ever NA or current 777
- Skip dose/years/vape for v1
- Caveat: current smoking may reflect post-diagnosis quitting (same post-dx caveat as wearables)

### Obesity
- `mhoccur_obs` from `comorbidity.parquet` (cherry-pick from failed 1B block; expect redundancy with BMI)

### Vision
- `via1–3` from `mood.parquet` (available but excluded from C1 `scores` set)
- Likert 1–5 self-report vision difficulty; **severity-adjacent** (not hard leakage like PDR)

## Protocol
- Parent assert C1 artifact metrics + recompute proba (tol 1e-6)  
- n=1824 splits; 50 trials; freeze-before-test **before HPO refuse-overwrite**  
- HPO both families; SHAP+perm; c3 on **new** cols only  
- Artifacts `sens_<name>_<ts>/` + DECISIONS/REPORT update  

## Out
- marital status, diet, full 1B reopen, Path B, pairwise combos  
