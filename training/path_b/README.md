# Path B — privileged CGM (LUPI)

| Stage | Role | Status |
|---|---|---|
| **B1** | Controlled multi-task (same backbone ± day-level glucose) | **CONCLUDED** — `REPORT_B1.md` |
| **B2** | Two-stage ablation (glucose emulator → T2D) | **CONCLUDED** — `REPORT_B2.md` |
| **B4** | Seq2seq CGM trajectory + rep-distill (headline) | **A+B+hard CONCLUDED** — `REPORT_B4*.md`; no deployable raise |
| **B3** | Logit-KD baseline (Diasense) | **next** |

**B1 freeze (2026-07-15):** pure-seq test 4-AUC **0.652**; multi-task **null**; GREEN fuse no raise.

**B2 freeze (2026-07-15, `b2_grid_20260715`):**

| | 4-AUC | Binary | vs C1 |
|---|---:|---:|---|
| D1 (matched direct) | **0.7378** | **0.8309** | ≡ C1 |
| T1 (C1 + predicted CGM) | 0.7345 | 0.8141 | **no raise** |
| O1 (C1 + **true** CGM, aux, non-deployable) | **0.8227** | **0.8768** | oracle only |

- Ablation T1−D1 ΔAUC **−0.003** CI lo≯0; O1−D1a **+0.094** (Stage-1 R²~0.05 bottleneck).
- **No deployable B2 arm beats C1.** Residual knobs (daily Stage-1, C1→glu Stage-1) are optional footnotes — **not** required to close B2.
- See **`REPORT_B2.md`**.

Docs: `PLAN_B*`, `REPORT_B*`, `AUDIT_B1_UNDERPERF.md`, `DECISIONS.md`.

## B1 run (reference; package frozen)

```bash
export HIP_VISIBLE_DEVICES=0

# smoke
.venv/bin/python -m training.path_b.b1 --run-id smoke_b1 --quick --device cuda

# multi-task claim shape
.venv/bin/python -m training.path_b.b1 --run-id b1_grid_YYYYMMDD_fix \
  --lambdas 0,0.5 --device cuda

# optional GREEN late-fuse (confirmation; did not raise)
.venv/bin/python -m training.path_b.b1 --run-id b1_green_YYYYMMDD \
  --lambdas 0 --device cuda --green-fusion
```

Artifacts: `training/path_b/b1/artifacts/<run_id>/`.

**Primary science (B1):** per-λ test ΔAUC vs λ=0 (paired person bootstrap) — **null after fix**.  
Path A watch floor 0.666 is **informational** only (not arch-matched).

## B2 run (reference; package frozen)

```bash
.venv/bin/python -m training.path_b.b2 --run-id b2_smoke_YYYYMMDD --quick
.venv/bin/python -m training.path_b.b2 --run-id b2_grid_YYYYMMDD --n-trials 50 --stage1-n-trials 30
```

Artifacts: `training/path_b/b2/artifacts/<run_id>/`.

## B4 freeze

### B4-A (`b4_grid_20260715`) — traj multi-task
| | 4-AUC | vs D1 |
|---|---:|---|
| D1 | **0.736** | — |
| S0+C1 / Sλ+C1 | 0.713 / 0.714 | **no raise** |
| S0 neural | 0.646 | informational |

### B4-B — rep-distill (easy + hard teachers)
| Teacher | Student μ=0/0.3/1 | Best hybrid | vs D1 |
|---|---|---:|---|
| Easy X∥cgm | 0.646 / 0.625 / 0.636 | 0.735 | no raise |
| H1 cgm_only | 0.646 / **0.626** / 0.634 | 0.723 | no raise |
| H2 wear→cgm | 0.646 / **0.620** / 0.638 | 0.713 | no raise |

Easy-teacher loophole **closed** (hard modes still null). See `REPORT_B4.md`, `REPORT_B4_B.md`, `REPORT_B4_B_HARD.md`. **B3** next.

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b4 --run-id b4_grid_YYYYMMDD --mode all --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_distill_YYYYMMDD --mode distill_hybrid --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h1_YYYYMMDD \
  --mode distill_hybrid --teacher-mode cgm_only --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h2_YYYYMMDD \
  --mode distill_hybrid --teacher-mode wear_cgm --device cuda
```

## Next

**B3** (logit-KD baseline) last. Do **not** reopen B1/B2/B4 claim grids without a new `PLAN_*`.
