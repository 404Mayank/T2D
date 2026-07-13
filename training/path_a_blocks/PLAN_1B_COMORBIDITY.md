# Plan 1B — Watch + onboarding + comorbidity (LOCKED post-critique)

> Package: `training/path_a_blocks/`.  
> Prior: watch floor 4-AUC **0.666**; 1A **0.699** (decision_bar_pass).  
> Critique (glm-5.2): **revise** → incorporated below. User locks 2026-07-13.

---

## User locks (final)

| # | Choice | Lock |
|---|---|---|
| 1 | Arthritis `mhoccur_ra` in primary? | **Yes** |
| 2 | HBP `mhoccur_hbp`? | **Both**: primary **with** HBP + required sensitivity **without** HBP |
| 3 | Kidney/circulation (`rnl`, `circ`)? | **Sensitivity only** (consequence-proxy risk) |

---

## Primary feature set — `1B_core` (claim run)

**Baseline (hard-locked from 1A — no re-selection):**
- Watch: exact 30 GREEN columns from `watch_green`
- Onboarding: exact 15 cols from 1A `onboarding_keep`

**Comorbidity binaries (yes/no):**
```
mhoccur_hbp      # HTN diagnosis (redundancy stress-tested via no_hbp)
mhoccur_clsh     # high cholesterol
mhoccur_mi       # heart attack
mhoccur_strk     # stroke
mhoccur_cvdot    # other heart
mhoccur_ra       # arthritis (≥5% + ρ; user lock)
```

**Engineered (single aggregate only — no nested max flags):**
```
comorb_count_core = sum of the 6 binaries above (NaN-safe: treat null as 0 for count only
                   after optional; prefer sum of available / or sum with null→0 documented)
```

**Excluded from primary:** obs, rnl, circ, plm, yn, fallot, rare neuro (ad/pd/ms), eye comfort (amd/ded/crt), fall, gi, etc.

---

## Sensitivity runs (pre-registered)

| ID | Delta vs primary | Purpose |
|---|---|---|
| **`1B_no_hbp`** | Drop `mhoccur_hbp` from binaries + recount | O1: lift not only BP re-encoding |
| **`1B_plus_complications`** | Add `mhoccur_rnl`, `mhoccur_circ` | O2: consequence markers as severity proxies |
| **`1B_plus_obs`** | Add `mhoccur_obs` | BMI redundancy |
| **`1B_ge5pct`** (optional) | All binary mhoccur with core yes%≥5%, exclude yn/fallot/rare neuro | Upper bound short of dump |

Primary **decision_bar_pass** is evaluated on **`1B_core` only**. Sensitivities are interpretive.

---

## Protocol

- Same as 1A: n=1824, splits 1277/270/277, weights balanced, 50 trials, val-select, freeze-before-test  
- **Δ bar parent = 1A** (`onboarding_20260713_224744`): 4-AUC 0.6987, binary 0.7492  
- Also report Δ vs watch floor  
- c1/c2/c3 as 1A; c3 = comorbidity-block perm stability + report fraction of comorb features with perm>0  
- Soft: report class-2 Δ (no hard fail)  
- SHAP tags: watch / onboarding / comorbidity  
- Assert 1A metrics_test.json match config references at startup  

---

## Implementation

```text
run_1b.py --feature-set core|no_hbp|plus_complications|plus_obs|ge5pct
```

Artifacts under `artifacts/<run_id>/`. Update DECISIONS + REPORT after runs.

---

## Critique incorporation log

| Objection | Resolution |
|---|---|
| O1 HBP↔BP | Primary keeps HBP; **required `1B_no_hbp`** |
| O2 rnl/circ consequences | **Sensitivity only** |
| O3 ra ≥5% | **In primary** (user) |
| O4 plm unjustified | **Out of primary** |
| O5 nested aggregates | **Only `comorb_count_core`** |
| O6 full_binary rares | Replaced by optional **ge5pct** |
| O7 c3 single-feature | Report n comorb with perm>0 |
| Baseline lock | Explicit frozen 1A feature lists |
| Class-2 | Soft report, not gate |
