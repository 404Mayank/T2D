# T2D — AI-READI severity prediction

Predict **4-class T2D severity** (0 healthy → 3 insulin) from Garmin wearables (AI-READI).  
Paper headline claim = **watch-only**. Clinical self-report = secondary / deployable. CGM = Path B (privileged).

## Status (2026-07-14)

**Path A tabular is frozen.**

| Track | Test 4-AUC | Binary | Role |
|---|---:|---:|---|
| Watch-only W0 | **0.666** | 0.689 | paper headline |
| Deployable C1 (watch+onboarding+mood) | **0.738** | 0.831 | secondary tabular |
| 1B comorbidity core | 0.709 | 0.778 | bar fail — not in stack |

Wrap ablations: PAID carries mood; CES null; minimal 12/18 fail retention vs C1; dedicated binary HPO does not beat multiclass-derived `1−P0`.  
Details: `training/path_a_blocks/REPORT_A_WRAP.md`.

**Done:** cleaning/FE → watch floor → diagnostics → 1A/1B/1C → wrap freeze.  
**Left:** Path B (privileged CGM / distillation). Optional: diet block, GREEN v2 FE, ordinal.

## Layout

| Path | What |
|---|---|
| `pipeline/` | clean + feature engineering |
| `data/processed/` | consumer tables (gitignored) |
| `training/path_a_watch/` | watch-only GBM floor |
| `training/path_a_blocks/` | block ladder + wrap |
| `AGENTS.md` | doc index |
| `Training.md` | methodology |

Authority: `DATA_AUDIT.md` → `CLEANING.md` → `PROCESSED.md` → `FEATURES.md` → `Training.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/path_a_watch/requirements.txt
# AMD dGPU for LightGBM OpenCL (this machine):
export DRI_PRIME=1
```

Raw data and `data/processed/` are local (not in git). Artifacts under `training/path_a_*/artifacts/` are gitignored.

## Run Path A (repro)

```bash
export DRI_PRIME=1
.venv/bin/python -m training.path_a_watch --run-id <id>
.venv/bin/python -m training.path_a_blocks.run_1a
.venv/bin/python -m training.path_a_blocks.run_1c --feature-set scores
.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --all
```

CatBoost is CPU-only here; LightGBM may use OpenCL GPU.

## Docs worth reading

- `T2D.md` — north star  
- `training/path_a_blocks/REPORT.md` — ladder progress  
- `training/path_a_blocks/REPORT_A_WRAP.md` — freeze + wrap analytics  
- `training/path_a_blocks/DECISIONS.md` — locks  
- `COMPUTE.md` — machines / GPU notes  
