# Path A blocks — diagnostics + onboarding ladder

Deployable / block-ablation track. Watch-only scientific floor: `../path_a_watch/`.

| Doc | Role |
|---|---|
| `PATH_AHEAD.md` | Roadmap, gates, milestone table |
| `DECISIONS.md` | Living decisions + results |
| `config.yaml` | Paths, onboarding keep list, HPO |

## Run

```bash
export DRI_PRIME=1

# Phase 0 — floor diagnostics
.venv/bin/python -m training.path_a_blocks.diagnostics \
  --floor-run full_20260713_221240 \
  --run-id diag_$(date +%Y%m%d_%H%M%S)

# Phase 1A — watch + hard onboarding
.venv/bin/python -m training.path_a_blocks.run_1a \
  --run-id onboarding_$(date +%Y%m%d_%H%M%S)

# smoke
.venv/bin/python -m training.path_a_blocks.run_1a --quick --run-id smoke_1a
```

Artifacts under `artifacts/<run_id>/` (gitignored via parent pattern if configured).
