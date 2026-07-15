# Path B — privileged CGM (LUPI)

| Stage | Role | Status |
|---|---|---|
| **B1** | Controlled multi-task (same backbone ± day-level glucose) | **CONCLUDED** — `REPORT_B1.md` |
| **B2** | Two-stage ablation (glucose emulator → T2D) | **CONCLUDED** — `REPORT_B2.md` |
| **B2-V2** | Daily MSE mid + variance-propagated stacking (sibling) | **CONCLUDED** — `REPORT_B2_V2.md`; still null vs C1 |
| **B4** | Seq2seq CGM trajectory + rep-distill (headline) | **A+B+hard CONCLUDED** — `REPORT_B4*.md`; no deployable raise |
| **B4-V2** | RKD/CRD distill + PCGrad MTL + OOF fusion (sibling retry) | **CONCLUDED null** — `REPORT_B4_V2.md` (post-claim critique: approve-with-caveats) |
| **B3** | Logit-KD baseline (Diasense) | **CONCLUDED** — `REPORT_B3.md`; no deployable raise |

**B1 freeze (2026-07-15):** pure-seq test 4-AUC **0.652**; multi-task **null**; GREEN fuse no raise.

**B2 freeze (2026-07-15, `b2_grid_20260715`):**

| | 4-AUC | Binary | vs C1 |
|---|---:|---:|---|
| D1 (matched direct) | **0.7378** | **0.8309** | ≡ C1 |
| T1 (C1 + predicted CGM) | 0.7345 | 0.8141 | **no raise** |
| O1 (C1 + **true** CGM, aux, non-deployable) | **0.8227** | **0.8768** | oracle only |

- Ablation T1−D1 ΔAUC **−0.003** CI lo≯0; O1−D1a **+0.094** (Stage-1 R²~0.05 bottleneck).
- **No deployable B2 arm beats C1.** Residual knobs attacked in **B2-V2** (below).
- See **`REPORT_B2.md`**.

**B2-V2 freeze (2026-07-16, `b2v2_grid_20260716`):** daily watch→CGM MSE mid + quantile spread/daysd; reduced Y {mean,sd,tir,tar}.

| | 4-AUC | Binary | vs C1 |
|---|---:|---:|---|
| D1 (matched) | **0.7378** | **0.8309** | ≡ C1 |
| T1v (C1 + var pack) | 0.7271 | 0.8174 | **no raise** |
| T1p (C1 + mid only) | 0.7296 | 0.8129 | **no raise** |
| O1 (true CGM, aux) | **0.8378** | 0.8796 | oracle only |

- T1v−D1 ΔAUC **−0.011** CI lo≯0; O1−D1a **+0.096**; Stage-1 val mean R² ~0.09 / test ~0.03.
- **No deployable B2-V2 arm beats C1.** Post-hoc adversarial audit: null **authentic** (Ŷ still ~18% Stage-2 importance). See **`REPORT_B2_V2.md`**.

Docs: `PLAN_B*` / `REPORT_B*` (incl. **`REPORT_B3.md`**), `AUDIT_B1_UNDERPERF.md`, `DECISIONS.md`.

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

## B2-V2 run (reference; package frozen)

```bash
.venv/bin/python -m training.path_b.b2v2 --run-id b2v2_smoke_YYYYMMDD --quick
.venv/bin/python -m training.path_b.b2v2 --run-id b2v2_grid_YYYYMMDD --n-trials 50 --stage1-n-trials 20
```

Artifacts: `training/path_b/b2v2/artifacts/<run_id>/`.

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

Easy-teacher loophole **closed** (hard modes still null). See `REPORT_B4.md`, `REPORT_B4_B.md`, `REPORT_B4_B_HARD.md`.  
**B4-V2** (`REPORT_B4_V2.md`): RKD/CRD/PCGrad/OOF **null** for deployable raise (honest accept; post-claim critique approve-with-caveats — F3 residual disclosed, freeze stands). **B3** logit-KD is **frozen** (`REPORT_B3.md`). Residual LUPI → SSL plan only, not more B4 KD grids.

```bash
export HIP_VISIBLE_DEVICES=0
.venv/bin/python -m training.path_b.b4 --run-id b4_grid_YYYYMMDD --mode all --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_distill_YYYYMMDD --mode distill_hybrid --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h1_YYYYMMDD \
  --mode distill_hybrid --teacher-mode cgm_only --device cuda
.venv/bin/python -m training.path_b.b4 --run-id b4b_hard_h2_YYYYMMDD \
  --mode distill_hybrid --teacher-mode wear_cgm --device cuda
```

## B3 freeze (`b3_grid_20260715`)

| | 4-AUC | Binary | vs C1 |
|---|---:|---:|---|
| D1 (matched) | **0.7378** | **0.8309** | ≡ C1 |
| G_α=0.3 (decision) | 0.7469 | 0.8169 | Δ+0.009 CI lo≯0 **fail** |
| Tch (aux, non-deployable) | **0.8227** | 0.8768 | privilege |

Hinton `N_α=0.3`−N0 **null**. T∈{1,4} **null**. See **`REPORT_B3.md`**.

```bash
.venv/bin/python -m training.path_b.b3 --run-id b3_grid_YYYYMMDD --device cuda
```

## Next

Path B planned ladder **complete** (B1→B2→B4→B3). Do **not** reopen claim grids without a new `PLAN_*`.
