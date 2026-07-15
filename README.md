# T2D — AI-READI severity prediction

Predict **4-class T2D severity** (0 healthy → 3 insulin) from Garmin wearables (AI-READI).  
Paper headline claim = **watch-only**. Clinical self-report = secondary / deployable. CGM = Path B (privileged).

## Status (2026-07-15)

**Path A tabular is frozen.** **Path B B1, B2, and B4 (A+B easy+hard) are concluded.** Next: **B3 last**.

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
| **B1** | Controlled multi-task (± day-level CGM) | **Frozen** — pure-seq **0.652**; multi-task **null**; GREEN fuse no raise |
| **B2** | Two-stage glucose emulator → T2D | **Frozen** — **no deployable arm beats C1**; T1 0.735; oracle O1 **0.823** (non-deployable, +0.09 vs D1a) |
| **B4** | Seq2seq traj multi-task + rep-distill (easy + hard teachers) | **Concluded** — **no deployable raise** vs C1; see `REPORT_B4*.md` |
| **B3** | Logit-KD baseline (Diasense) | **next** |

**Who beats C1?** Only **oracle O1** (true CGM, aux pool) — not deployable. B1/B2/B4 deployable arms all ≤ C1. Privilege is real (oracle / teacher probes); wear→glucose handoff under tested recipes is null. Authority: `REPORT_B1.md`, `REPORT_B2.md`, `REPORT_B4.md`, `REPORT_B4_B.md`, `REPORT_B4_B_HARD.md`.

**Done:** cleaning/FE → Path A freeze → B1 → B2 → B4.  
**Left:** B3. Optional Path A leftovers: diet, GREEN v2 FE, ordinal.

## Layout

| Path | What |
|---|---|
| `pipeline/` | clean + feature engineering |
| `data/processed/` | consumer tables (gitignored) |
| `training/path_a_watch/` | watch-only GBM floor |
| `training/path_a_blocks/` | block ladder + wrap (frozen) |
| `training/path_b/` | privileged CGM ladder (B1+B2+B4 concluded → B3) |
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
