# Path B — privileged CGM (LUPI)

| Stage | Role | Status |
|---|---|---|
| **B1** | Controlled multi-task (same backbone ± day-level glucose) | **CONCLUDED** — see `REPORT_B1.md` |
| **B2** | Two-stage ablation | **next** |
| **B4** | Seq2seq CGM trajectory + rep-distill (headline) | later |
| **B3** | Logit-KD baseline (Diasense) | last |

**B1 freeze (2026-07-15):** post-fix pure-seq test 4-AUC **0.652**; multi-task λ=0.5 **null** (paired CI lo≯0); GREEN late-fuse **no raise**. Pre-fix `b1_grid_20260714` is broken-input history only.

Docs: `PLAN_B1_DATA.md`, `PLAN_B1_TRAIN.md`, `PLAN_B1_IMPL.md`, `PLAN_B1_FIX.md`, `AUDIT_B1_UNDERPERF.md`, `DECISIONS.md`, **`REPORT_B1.md`**.

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

## Next

Implement **B2** (two-stage; `cgm_person` ready) then **B4** headline (5-min grid FE still needed).
