# B1 implementation plan

**Date:** 2026-07-14  
**Status:** revised after critique; implementing.  
**Authority:** `PLAN_B1_TRAIN.md` (locks), `DECISIONS.md`, `PROCESSED.md`.

**Critique disposition:** revise. Accepted locks: cgm-aware truncate (not last-16), glu_mask requires watch_day_valid, 2h→h before glu+attn, per-λ seed reset, post-grid paired bootstrap, std floor, smoke-first (full grid later), no person-level head in v1.
**Superseded 2026-07-15 (PLAN_B1_FIX):** sleep NaN→0 removed; sleep_duration uses train median on observed; **train-only feature z-score** required (C2).

---

## 1. Scope

Implement `training/path_b/b1/` runnable package:

- Load daily watch + CGM → person sequences
- Train `attn_lstm_64` multi-task (day-level glu head)
- λ grid `{0, 0.3, 0.5, 1.0}` with fixed seed / ES / metrics / bootstrap
- Artifacts + short report skeleton

**In scope:** data, model, train, eval, CLI, config, smoke, docs touch.  
**Out:** B2/B3/B4, Optuna, survey features, h=128 unless free, person-level glu as primary.

---

## 2. Files

```
training/path_b/
  __init__.py
  README.md
  b1/
    __init__.py
    __main__.py
    config.yaml
    data.py
    model.py
    metrics.py      # thin wrap / reuse path_a_watch.metrics where possible
    train.py
    evaluate.py
    run.py
    requirements.txt  # torch + shared
```

Reuse: `training.path_a_watch.metrics` (macro_ovr_auc, binary, brier, per-class).  
Do **not** import CatBoost/LightGBM paths.

---

## 3. `data.py` contract

### Inputs
- `watch_daily.parquet`, `cgm_daily.parquet`, `pool_masks.parquet`

### Build
1. Filter masks to `wearable_core`.
2. Join watch_daily ⋈ meta on person_id.
3. Left-join cgm_daily on `(person_id, day_local)`.
4. For glu mask: `cgm_day_valid & aux_eligible` (aux from meta).
5. Day index = **union** of watch+cgm days for core pids; sort `day_local` ascending.
6. Features: **fixed ordered 18** cols in config (no coverage counts).
7. Impute: train median on valid watch days; **`sleep_duration_hours` leave NaN then median on observed** (not fill-zero after C1). Activity / `sleep_n_bouts` may fill 0. Then **train-only feature z-score** (observed-only mean/std for non-fill_zero cols).
8. Truncate to `max_len=16`: **prefer days with `cgm_day_valid`, earliest tie-break**, then fill remaining earliest watch days. **Never last-16** (destroys glu days on long-window aux pids).
9. `glu_mask = cgm_day_valid & watch_day_valid & aux_eligible`.
10. Tensors: `X[T,d]`, `watch_mask[T]`, `glu_y[T,8]`, `glu_mask[T]`, `y`, `pid`.

### Z-score glu
Fit mean/std of each of 8 targets on **train ∩ aux_eligible ∩ cgm_day_valid** days.  
Apply to all; where glu_mask=0, store 0 targets (ignored by loss).

### Class weights
Inverse freq on train labels, sum-normalize to 1 → length-4 vector.

### Dataset / collate
`PersonSeqDataset` + collate pad to batch max ≤16 (already truncated).

### Asserts
- No label/split in feature tensor
- Train/val/test pid disjoint
- n train/val/test match core split counts (~1277/270/277)
- Glu mask never True for non-aux

---

## 4. `model.py`

```
Input x[B,T,d], mask[B,T]
→ Linear(d, h) + ReLU + Dropout
→ pack/pad BiLSTM(h)  # bi → out 2h
→ Linear(2h, h) per timestep  # BOTH glu head and attention see h_t∈R^h
→ glu_head: Linear(h, 8) on each h_t
→ Attention: score = Linear(h,1); masked softmax; z = sum α h_t
→ class_head: Linear(h, 4)
```

Return: `logits[B,4]`, `glu_pred[B,T,8]`, `z[B,h]`.
Defaults: h=64, dropout=0.2, bidirectional=True.

---

## 5. `train.py`

- Loss: `CE(logits, y; weight=class_w) + λ * masked_mse(glu_pred, glu_y, glu_mask)`
- masked_mse: sum over mask / (n_mask * 8 + eps); if n_mask==0 → 0
- λ=0: still forward glu head but multiply glu loss by 0
- Opt: AdamW lr=1e-3 wd=1e-4; grad clip 1.0
- ES: patience 15 on **val macro-OVR AUC** (compute every epoch)
- LR schedule: ReduceLROnPlateau on val macro-AUC (mode max, factor 0.5, patience 5)
- Seed all RNGs 42
- Save best checkpoint by val macro-AUC
- Log epoch: train_ce, train_glu, val_ce, val_glu, val_auc, val_bin_auc

Quick mode: max_epochs=3, patience=2, subset optional via max_train_pids.

---

## 6. `evaluate.py` / bootstrap

- Per λ: load best ckpt; predict **once**; write `test_preds.parquet` (person_id, p0..p3, y)
- Metrics via path_a_watch.metrics re-export; cal via path_a_watch.calibrate if present
- Glu MSE/MAE on aux test days (mask)
- **Paired bootstrap lives in run.py post-grid** (not inside per-λ evaluate): n=2000, seed 42, person unit, shared resample mask across λ vs λ=0

---

## 7. `run.py` CLI

```
python -m training.path_b.b1 --run-id ID [--lambdas 0,0.3,0.5,1.0] [--quick] [--device cpu|cuda|auto]
```

Per λ: train → eval → `artifacts/<run_id>/lambda_<λ>/`
Aggregate: `summary.json` with all λ metrics + vs Path A floor constants.

---

## 8. Config (`config.yaml`)

Paths, feature list, glu list, h=64, max_len=16, batch=32, epochs=80, patience=15, seed=42, λ list, path A floor constants, bootstrap n.

---

## 9. Acceptance (smoke)

1. `--quick` completes on CPU without OOM  
2. Batch shapes correct; glu_mask sum > 0 on train  
3. λ=0 and λ=0.5 both run; metrics JSON written  
4. No aux leakage: test metrics computed without needing CGM at forward for class head  
5. Docs: DECISIONS note + README how to run  

Full grid may be long on CPU — smoke first; full grid if time/GPU, else document command for Lightning.

---

## 10. Risks / locks already decided

- Day-level glu primary  
- No coverage features  
- Aux-only glu  
- Path A floor informational  
- Matched dead glu head at λ=0  

## 11. Open choices — locked

| # | Lock |
|---|---|
| Truncate | Prefer cgm_valid days, earliest tie-break; never last-16 |
| BiLSTM | 2h→h per-timestep before glu + attention |
| Person glu head | **No** in v1 |
| Session scope | Smoke first (`--quick`); full grid after (AMD/CUDA if works) |
| Seed | Reset torch/np/random **before each λ** model init |
| Sleep NaN | train median on observed (post-C1); not fill-zero |
| Input scale | train-only StandardScaler on feature_cols (C2) |
| Z-score | std floor 1e-6 |
| Class weights | Train CE only |
