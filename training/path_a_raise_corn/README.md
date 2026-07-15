# Path A raise — CORN ordinal MLP

Post-freeze raise on the **exact C1** feature matrix (watch + onboarding + PAID/CES totals).  
Does **not** reopen or rewrite frozen Path A claim numbers.

**Status (2026-07-16): concluded null.** Primary bar fail. Authority: `REPORT.md`.

| Cell | Role | Full run | Test 4-AUC | vs C1 |
|---|---|---|---:|---:|
| CORN MLP | primary (bar) | `corn_full_20260715_211707` | **0.706** | **−0.031** FAIL |
| CE-MLP | attribution control | `ce_mlp_full_20260715_211707` | **0.713** | −0.025 |
| C1 parent | frozen CatBoost | `mood_scores_20260714_014415` | **0.738** | 0 |

CORN−CE Δ ≈ 0; soft class-2 win no; Path A freeze / C1 unchanged.  
Wording: “CORN MLP + median-impute + weighted recipe does not raise C1 GBM” (not bare “ordinal fails”).

| Cell | Role |
|---|---|
| CORN MLP | primary; decision bar vs C1 |
| CE-MLP | required attribution control |
| CORAL / σ-stack | optional — **not run** (primary not competitive) |

## Setup

```bash
.venv/bin/pip install -r training/path_a_raise_corn/requirements.txt
```

## Commands

```bash
export DRI_PRIME=1
cd /path/to/T2D

# unit checks (no data / no train)
.venv/bin/python -m training.path_a_raise_corn --self-check

# smoke (fixed hparams, ≤15 epochs; both arms) — non-claim
.venv/bin/python -m training.path_a_raise_corn --quick --arm both

# full HPO (20 trials/arm default) — refuse-overwrite on existing run ids
.venv/bin/python -m training.path_a_raise_corn --arm both
.venv/bin/python -m training.path_a_raise_corn --arm corn
.venv/bin/python -m training.path_a_raise_corn --arm ce
```

Artifacts: `training/path_a_raise_corn/artifacts/<run_id>/`.

## Authority

- **Report (numbers):** `REPORT.md`
- Plan: `PLAN_A_RAISE_CORN.md`
- Decisions + null audit: `DECISIONS.md`
- Parent C1: `training/path_a_blocks/artifacts/mood_scores_20260714_014415`
