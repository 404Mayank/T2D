# AGENTS.md — Project File Index

| File | Purpose |
|---|---|
| **T2D.md** | Project objective & north-star (1-pager). |
| **DATA_STRUCTURE.md** | Raw AI-READI layout, schemas, access. |
| **DATA_AUDIT.md** | Empirical data audit + cleaning checklist. |
| **CLEANING.md** | Cleaning/FE pipeline design, config, full-cohort results. |
| **PROCESSED.md** | Processed-data layout and train-time consumer contract. |
| **FEATURES.md** | Feature inventory, leakage rules, literature. |
| **Training.md** | ML methodology (Path A/B, metrics, build order). |
| **COMPUTE.md** | Machines, storage, GPU placement. |

**Supporting:** `pipeline/` (clean/FE code), `convert_pipeline.py` (raw ETL), `audit_data*.py`, `logs/`, `data/full/AI_READI/` (raw), `data/processed/` (outputs, gitignored).

**Training packages:**

| Path | Role |
|---|---|
| `training/path_a_watch/` | Path A **watch-only** GBM floor (scientific claim baseline) |
| `training/path_a_blocks/` | Path A **block ladder** (diagnostics → +onboarding → …); deployable track |
| `training/path_a_blocks/PATH_AHEAD.md` | Raise-floor roadmap & gates |
| `training/path_a_blocks/REPORT.md` | Latest diagnostics + 1A results report |
| `training/path_a_blocks/DECISIONS.md` | Living decisions log for blocks |

**Authority:** `DATA_AUDIT.md` → `CLEANING.md` → `PROCESSED.md` → `FEATURES.md` → `Training.md`.  
**Run results authority for Path A raises:** `training/path_a_blocks/REPORT.md` + package `DECISIONS.md` files.
