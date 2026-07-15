# T2D — AI-READI severity prediction

Predict **4-class T2D severity** (0 healthy → 3 insulin) from Garmin wearables (AI-READI).  
Paper headline claim = **watch-only**. Clinical self-report = secondary / deployable. CGM = Path B (privileged).

## Status (2026-07-16)

**Path A tabular is frozen.** **Path B B1–B4 (+B4-V2), B3, B1-GS, and B2-V2 are all concluded** — no deployable arm beats C1; oracle/teacher privilege is real.

### Path A (unchanged)

| Track | Test 4-AUC | Binary | Role |
|---|---:|---:|---|
| Watch-only W0 | **0.666** | 0.689 | paper headline |
| Deployable C1 (watch+onboarding+mood) | **0.738** | 0.831 | secondary tabular |
| 1B comorbidity core | 0.709 | 0.778 | bar fail — not in stack |

Wrap: PAID carries mood; CES null; minimal sets fail retention vs C1; binary HPO does not beat multiclass-derived `1−P0`.  
C1 sensitivities (smoke/obs/via): all bar-fail. Authority: `training/path_a_blocks/REPORT_A_WRAP.md`.

### Path B

| Stage | Role | Status |
|---|---|---|
| **B1** | Controlled multi-task (± day-level CGM) | **Frozen** — pure-seq **0.652**; plain-λ **null**; GREEN fuse no raise; **GS (PCGrad/UW) also null** (`REPORT_B1_GS.md`) |
| **B2** | Two-stage glucose emulator → T2D | **Frozen** — **no deployable arm beats C1**; T1 0.735; oracle O1 **0.823** (non-deployable, +0.09 vs D1a) |
| **B2-V2** | Daily MSE mid + variance-propagated stacking | **Frozen** — T1v 0.727 ≯ C1; O1 +0.096; Stage-1 val R²~0.09 (`REPORT_B2_V2.md`) |
| **B4** | Seq2seq traj multi-task + rep-distill (easy + hard teachers) | **Concluded** — **no deployable raise** vs C1; see `REPORT_B4*.md` |
| **B4-V2** | RKD/CRD distill + PCGrad MTL + OOF fusion | **Concluded null** — best hybrid F1 **0.726** ≯ D1 **0.736**; post-claim critique approve-with-caveats; see `REPORT_B4_V2.md` |
| **B3** | Logit-KD baseline (Diasense) | **Concluded** — G_α=0.3 0.747 vs C1 0.738; CI lo≯0; see `REPORT_B3.md` |

**Who beats C1?** Only **oracle/teacher** (true CGM, aux pool) — not deployable. B1/B2/B2-V2/B3/B4/B4-V2 deployable arms all fail the raise bar. Privilege is real; wear→glucose / soft-logit / RKD handoffs under tested recipes are null. Authority: `REPORT_B1.md`, `REPORT_B1_GS.md`, `REPORT_B2.md`, `REPORT_B2_V2.md`, `REPORT_B3.md`, `REPORT_B4*.md`, `REPORT_B4_V2.md`.

**Done:** cleaning/FE → Path A freeze → post-freeze CORN MLP raise **null** (`path_a_raise_corn/`) → post-freeze ensemble raise **null** (`path_a_raise_ensemble/`) → Path B ladder (B1→B2→B4→B3) + siblings B1-GS / B2-V2 / B4-V2.  
**Left:** optional future `PLAN_*` only (e.g. SSL backbone; Path A diet; cal/op-point) — not more B1 λ/GS, B2 HPO, B4 KD-objective reopens, CORN primary, or ensemble re-litigation without a new plan.

## Layout

| Path | What |
|---|---|
| `pipeline/` | clean + feature engineering |
| `data/processed/` | consumer tables (gitignored) |
| `training/path_a_watch/` | watch-only GBM floor |
| `training/path_a_blocks/` | block ladder + wrap (frozen) |
| `training/path_a_raise_corn/` | post-freeze CORN/CE MLP raise (**null**; C1 unchanged) |
| `training/path_a_raise_ensemble/` | post-freeze multi-seed bag + cross-family ensemble (**null**; C1 unchanged) |
| `training/path_b/` | privileged CGM ladder (B1–B4+B4-V2+B3+B1-GS+B2-V2 **concluded**) |
| `AGENTS.md` | doc index + agent process locks |
| `Training.md` | methodology |

Authority: `DATA_AUDIT.md` → `CLEANING.md` → `PROCESSED.md` → `FEATURES.md` → `Training.md`.

## Setup

See `COMPUTE.md` and package READMEs. Typical:

```bash
.venv/bin/python -m pipeline.run_fe --blocks watch          # Path A GREEN
.venv/bin/python -m pipeline.run_fe --blocks grid_5min      # Path B4 5-min grid
.venv/bin/python -m training.path_a_watch ...
.venv/bin/python -m training.path_b.b2 --run-id b2_grid_YYYYMMDD
.venv/bin/python -m training.path_b.b4 --run-id b4_grid_YYYYMMDD --mode all --device cuda
```
