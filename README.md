# T2D — AI-READI severity prediction

Predict **4-class T2D severity** (0 healthy → 3 insulin) from Garmin wearables (AI-READI).  
Paper headline claim = **watch-only**. Clinical self-report = secondary / deployable. CGM = Path B (privileged).

## Status (2026-07-15)

**Path A tabular is frozen.** **Path B B1 is frozen.** Next: **B2 → B4 → B3 last**.

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
| **B1** | Controlled multi-task (± day-level CGM) | **Frozen** — pure-seq test **0.652**; λ=0.5 multi-task **null**; GREEN late-fuse **no raise** |
| **B2** | Two-stage ablation | **next** |
| **B4** | Seq2seq CGM trajectory + rep-distill (headline) | later |
| **B3** | Logit-KD baseline (Diasense) | last |

B1 pre-fix grid (~0.51) was broken sleep FE + unscaled inputs — not a ceiling. Authority: `training/path_b/REPORT_B1.md`.

**Done:** cleaning/FE → Path A freeze → B1 fix/retest/fuse/freeze.  
**Left:** B2 → B4 → B3. Optional Path A leftovers: diet, GREEN v2 FE, ordinal.

## Layout

| Path | What |
|---|---|
| `pipeline/` | clean + feature engineering |
| `data/processed/` | consumer tables (gitignored) |
| `training/path_a_watch/` | watch-only GBM floor |
| `training/path_a_blocks/` | block ladder + wrap (frozen) |
| `training/path_b/` | privileged CGM ladder (B1 frozen → B2…) |
| `AGENTS.md` | doc index + agent process locks |
| `Training.md` | methodology |

Authority: `DATA_AUDIT.md` → `CLEANING.md` → `PROCESSED.md` → `FEATURES.md` → `Training.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/path_a_watch/requirements.txt
# Path B (torch) extras when needed:
# pip install -r training/path_b/b1/requirements.txt
# AMD dGPU for LightGBM OpenCL (this machine):
export DRI_PRIME=1
# Path B ROCm: export HIP_VISIBLE_DEVICES=0
```

Raw data and `data/processed/` are local (not in git). Training artifacts are gitignored.

## Run (repro)

```bash
export DRI_PRIME=1
# Path A
.venv/bin/python -m training.path_a_watch --run-id <id>
.venv/bin/python -m training.path_a_blocks.run_1a
.venv/bin/python -m training.path_a_blocks.run_1c --feature-set scores
.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --all

# Path B B1 (frozen package; reference only)
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b1 --run-id <id> --lambdas 0,0.5 --device cuda
```

CatBoost is CPU-only here; LightGBM may use OpenCL GPU. B1 uses ROCm torch with `cudnn` disabled on gfx1010.

## Docs worth reading

- `T2D.md` — north star  
- `AGENTS.md` — file index + plan→critique→implement loop  
- `training/path_a_blocks/REPORT_A_WRAP.md` — Path A freeze  
- `training/path_b/REPORT_B1.md` — B1 freeze  
- `training/path_b/DECISIONS.md` — Path B locks  
- `COMPUTE.md` — machines / GPU notes  
