# Path A ahead — raise floor before Path B

> Roadmap for **raising the Path A floor** after the watch-only GBM baseline.  
> Package: `training/path_a_blocks/`. Watch-only floor lives in `training/path_a_watch/`.  
> Authority: `Training.md` §6–7, `FEATURES.md` §0/§6, `PROCESSED.md`.  
> **Status (2026-07-14): Path A tabular frozen** — see `REPORT_A_WRAP.md`. Next package work is **Path B**.

## 0. Why a new package

| Package | Scope |
|---|---|
| `path_a_watch/` | Watch-only GREEN floor (scientific claim baseline) |
| **`path_a_blocks/`** | Diagnostics + **block ladder** (watch → +onboarding → …) |

Block runs are **not** the paper’s “from wearables” claim; they are the **deployable config** track and the honest Δ table vs the watch floor. Separate dir keeps freeze artifacts, SHAP guardrails, and narratives from overwriting each other.

## 1. Locked floor (do not move the goalposts)

Run: `training/path_a_watch/artifacts/full_20260713_221240/`

| Metric | Test claim |
|---|---:|
| 4-class macro-OVR AUC | **0.666** |
| Binary AUC (1−P0) | **0.689** |
| Macro AUPRC | **0.392** |
| Class-2 OVR AUC | **0.565** |
| Class-3 OVR AUC | **0.763** |
| Selected family | CatBoost Ordered |

Any block result is reported as **Δ vs this floor on the same test person_ids**.

## 2. Decision bar (Training.md §6)

A block **stays** in the deployable stack iff:

1. **Δ macro-OVR AUC > +1.0** (percentage points → **>+0.01** absolute) vs previous stage, **and**
2. Person-bootstrap CIs on test Δ do not freely overlap 0 (document method; fixed split ≠ nested CV), **and**
3. Stable permutation importance for the new block (not pure noise).

**SHAP guardrail:** if top-10 SHAP are all onboarding/survey → “from wearables” claim is strained; still valid as deployable config if labeled as such.

Class weights stay **locked** (`balanced` / `Balanced`) before any block importance narrative.

## 3. Execution plan (this package)

### Phase 0 — Diagnostics (first)

On the **frozen watch-only CatBoost** (or re-predicted from saved model):

| ID | Task | Output |
|---|---|---|
| D1 | 4×4 confusion (val + test) | `diagnostics/confusion_*.json/png` |
| D2 | Pairwise / per-class OVR already known; add 0vs3, 1vs2, 2vs3 binary AUCs | `diagnostics/pairwise_auc.json` |
| D3 | Site-stratified test metrics (UAB/UW/UCSD) | `diagnostics/site_stratified.json` |
| D4 | Base-rate table train/val/test | `diagnostics/base_rates.json` |
| D5 | Person-bootstrap CI on test macro-OVR AUC + binary AUC | `diagnostics/bootstrap_ci.json` |

**Kill-read:** if class-2 stays ~chance and site metrics swing hard, severity framing and confound notes go into DECISIONS before claiming lifts.

### Phase 1A — Watch + hard onboarding (next)

Training.md hard onboarding: **age, BMI/waist/height, family history, smoking, BP** — no sex/race.

From `onboarding.parquet` (available now):

| Keep | Notes |
|---|---|
| `age` | primary age |
| `bmi_vsorres`, `waist_vsorres`, `height_vsorres`, `weight_vsorres`, `hip_vsorres`, `whr_vsorres` | anthropometrics |
| `bp1_*`, `bp2_*` | BP (both readings) |
| `fh_dm2pt`, `fh_dm2sb` | family hx (parent/sibling) |
| `pulse_vsorres`, `pulse_vsorres_2` | clinic pulse (optional; not watch) |

| Drop / avoid | Why |
|---|---|
| `age_years_at_interview` | all zeros in release |
| `fh_dm2ptsp`, `fh_dm2sbsp` | high missing (60–74%); optional later |
| smoking | **not in onboarding parquet** this release |
| `clinical_site` | confound, never a feature |

**Protocol:** same as watch floor — 50 Optuna trials / family, val-select, freeze-before-test, physio coverage still in GREEN unless ablation flag, SHAP on full feature set with block tags.

**Success (M1):** test 4-AUC ≳ 0.72 **or** binary ≳ 0.78, **or** at least decision bar ΔAUC > +0.01 vs 0.666.  
**Then:** 1B comorbidity (HTN-first), not Path B.  
**If fail:** GREEN v2 FE and/or 3-class/binary reformulation before Path B.

### Later / closed on Path A

- 1B comorbidity → **bar fail** (core); complications sensitivity only  
- 1C mood → **bar pass** (PAID)  
- Phase A wrap → minimal/PAID/severity/binary done; **Path A frozen** (`REPORT_A_WRAP.md`)  
- 1D diet / GREEN v2 / CORN: optional, not required to leave Path A  
- **Next package work: Path B**

## 4. Milestones

| ID | Gate | Action if miss |
|---|---|---|
| M0 | Diagnostics complete + written | — |
| M1 | 1A clears decision bar or hits ~0.72/0.78 | Else GREEN v2 / label collapse |
| M2 | Watch-only GREEN v2 ΔAUC ≥ +0.01 | Else dynamics → SSL/Path B |
| M3 | Class-2 OVR ≥ 0.62 | Else lead with binary + insulin OVR |
| → Path B | M1 and M2 miss, gap documented | B1 cheap → B4 |

## 5. How to run (this package)

```bash
export DRI_PRIME=1
cd /path/to/T2D

# Phase 0
.venv/bin/python -m training.path_a_blocks.diagnostics \
  --floor-run full_20260713_221240 \
  --run-id diag_$(date +%Y%m%d_%H%M%S)

# Phase 1A
.venv/bin/python -m training.path_a_blocks.run_1a \
  --run-id onboarding_$(date +%Y%m%d_%H%M%S)
```

Artifacts: `training/path_a_blocks/artifacts/<run_id>/`.  
Decisions log: `DECISIONS.md` in this directory.

## 6. Chronology

| When | Event |
|---|---|
| 2026-07-13 | Watch floor complete (CatBoost test 4-AUC 0.666) |
| 2026-07-13 | PATH_AHEAD written; `path_a_blocks` started — diagnostics → 1A |
| 2026-07-13 | Diagnostics done; **1A decision_bar_pass=True** (test 4-AUC 0.699, binary 0.749, Δ+0.033) |
| 2026-07-14 | **1B_core decision_bar_pass=False** (0.709); plus_complications 0.724 numerical pass but non-claim |
| 2026-07-14 | **1C_scores decision_bar_pass=True** (0.738 / binary 0.831, Δ+0.039) — PAID-driven |
| 2026-07-14 | Phase A wrap plan revised post-critique: `PLAN_A_WRAP.md` |
| 2026-07-14 | **Wrap complete** (`run_wrap --all`); minimal retention fail → secondary = **C1**; Path A frozen |
| next | **Path B** (privileged CGM / distillation). See `REPORT_A_WRAP.md` |
