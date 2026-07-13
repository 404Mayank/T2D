# Plan 1C — Watch + onboarding + mood (LOCKED post-critique)

> Package: `training/path_a_blocks/`.  
> Parent for decision bar: **1A only** (1B comorbidity failed bar — not stacked).  
> Plan critique (glm-5.2): **revise** → incorporated below.

---

## Critique incorporation

| Objection | Resolution |
|---|---|
| cestl may be dead weight under block c3 | Report **per-feature** perm for `cestl` and `paidscore` separately; bar c3 still block-level |
| Sensitivity ladder vs bar multiplicity | **Only `1C_scores` is bar-eligible**; all other sets descriptive only |
| Parent assert incomplete | Explicit: recompute parent AUC == 0.6987; parent feature list == 1A freeze |
| Cohort join shrink | Assert mood covers all 1824; halt if not |
| paid_items + paidscore collinearity | `paid_items` set = **PAID items only (no paidscore)** for clean item read; still keep cestl optional off |
| Power expectation | Note: bar-pass likely needs ΔAUC ≳ +0.015 given 1B c2 failure at +0.01 |
| Column names | Use exact parquet names: `paid_scrd` (not scrd typo) |

---

## Goal

Test mood/diabetes-distress self-report on top of locked 1A (4-AUC **0.6987**, binary **0.7492**).

---

## Data (core n=1824)

| Feature | Form | ρ(label) | Notes |
|---|---|---:|---|
| `paidscore` | 0–20 | **+0.30** | Primary driver candidate |
| `cestl` | 0–30 | +0.09 | Weak; keep for product pair |
| `paid_*` items | 0–4 | +0.15…+0.32 | Sensitivity only |
| `ces1–10` | 0–3 | weak | Sensitivity full only |
| `via1–3` | 1–5 | +0.10…+0.14 | Sensitivity; not primary (product: vision not “mood”) |

PAID non-leakage per FEATURES.md (healthy non-zero; non-monotonic). Still severity-correlated self-report — deployable track only.

---

## Feature sets

**Baseline hard-lock:** exact 30 GREEN + 15 onboarding from 1A.

| ID | Mood cols | Bar-eligible? |
|---|---|---|
| **`scores`** (primary) | `cestl`, `paidscore` | **Yes** |
| `scores_via` | + via1, via2, via3 | No (descriptive) |
| `paid_items` | paid_cml, paid_dpr, paid_eng, paid_scrd, paid_wr (**no paidscore**) | No |
| `full` | scores + all ces + paid items + via | No |

---

## Protocol

- Same as 1A/1B: splits, weights, 50 trials, val-select, freeze-before-test  
- Parent: load 1A model; assert feature list; assert recomputed test AUC ≈ 0.6987  
- Mood join: require full pid coverage of 1824  
- Decision bar **only on `scores`**: c1 ΔAUC>+0.01 vs 1A; c2 bootstrap lo>0; c3 mood-block perm mean>0 & ≥1 mood feature perm>0  
- Report: per-feature perm (cestl, paidscore); class-2 Δ; SHAP mood count; Δ vs floor  
- Power note: expect c2 fail unless Δ ≳ ~0.015  

---

## Implementation

```text
run_1c.py --feature-set scores|scores_via|paid_items|full
```

Mirror `run_1b.py` parent-assert pattern. Code critique before 50-trial.

---

## Parallel 1B kidney sensitivity (user request)

| Run | Kidney/circ | Result |
|---|---|---|
| 1B_core | without | ΔAUC +0.0098, **bar fail** |
| 1B_plus_complications | **with** rnl+circ | ΔAUC **+0.025**, c1∧c2∧c3 true but **not claim set** (sensitivity) |

Complications markers help numerically; kept out of primary for consequence-vs-risk framing.

---

## Checklist

- [x] Plan critique  
- [x] Lock revisions  
- [ ] Implement run_1c + code critique  
- [ ] Run scores (+ descriptive sensitivities if useful)  
- [ ] Analyze + update DECISIONS/REPORT  
