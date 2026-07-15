# Path A raise — multi-seed bagging + cross-family ensemble

Post-freeze **modelchoice** raise vs frozen C1. Isolated package.

- Plan: `PLAN_A_RAISE_ENSEMBLE.md`
- Decisions: `DECISIONS.md`
- Parent (read-only): `training/path_a_blocks/artifacts/mood_scores_20260714_014415/`

## Requirements

- `DRI_PRIME=1` (same as Path A)
- **GPU** for LightGBM (`device=gpu` locked to C1 freeze)
- Frozen C1 artifacts present

## Run

```bash
export DRI_PRIME=1
cd /path/to/T2D

# Smoke (2 seeds, K=2 stack, non-claim)
.venv/bin/python -m training.path_a_raise_ensemble \
  --quick --run-id ens_smoke_$(date +%Y%m%d_%H%M%S)

# Full (5 seeds + stack OOF)
.venv/bin/python -m training.path_a_raise_ensemble \
  --run-id ens_full_$(date +%Y%m%d_%H%M%S)

# Debug: bags only
.venv/bin/python -m training.path_a_raise_ensemble --skip-stack --run-id ens_bags_only_...
```

Artifacts: `training/path_a_raise_ensemble/artifacts/<run_id>/`.

## Primaries

| ID | Arm | Question |
|---|---|---|
| A | `Bag_cat` | Multi-seed CatBoost vs C1? |
| B | `E_arith` | Cross-family blend add on top of bags? |

Bar: Δ4-AUC > +0.01 vs C1, paired bootstrap CI lo > 0, arm c3.
