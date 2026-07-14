# Path A blocks — ladder + wrap (frozen)

Deployable / block-ablation track. Watch-only scientific floor: `../path_a_watch/`.

**Status (2026-07-14):** Path A tabular **frozen**. Secondary pick = **C1 mood** (0.738 / 0.831).  
Authority: `REPORT_A_WRAP.md`, `DECISIONS.md`. Next project work: **Path B**.

| Doc | Role |
|---|---|
| `REPORT_A_WRAP.md` | Freeze + wrap analytics |
| `REPORT.md` | Ladder progress |
| `PATH_AHEAD.md` | Roadmap / gates |
| `DECISIONS.md` | Locks + chronology |
| `config.yaml` | Paths, keep lists, HPO, parent refs |
| `PLAN_*.md` | Historical plans (1B / 1C / wrap) |

## Run

```bash
export DRI_PRIME=1

.venv/bin/python -m training.path_a_blocks.diagnostics \
  --floor-run full_20260713_221240

.venv/bin/python -m training.path_a_blocks.run_1a
.venv/bin/python -m training.path_a_blocks.run_1b --feature-set core
.venv/bin/python -m training.path_a_blocks.run_1c --feature-set scores

.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --all
```

Artifacts under `artifacts/<run_id>/` (gitignored).
