# Path B1 — controlled multi-task training plan

**Date:** 2026-07-14  
**Status:** **implemented / plain-λ frozen** (`REPORT_B1.md`). GS sibling: `PLAN_B1_GS.md` /
`REPORT_B1_GS.md` (also concluded).  
**Role:** ablation only — not paper headline. Headline is B4 later.  
**Authority:** `Training.md` §4 B1 / §7 step 2; bars from Path A freeze.

**Critique disposition:** critiquer → **revise**. Accepted: day-level glu head as primary
(teammate-faithful), tighter multi-task win rule (no best-of-3 cherry-pick as sole claim), pin
class weights / bootstrap / aux-only glu pool, drop coverage counts from primary X, calibration
pre-register, Path A floor comparison informational (architecture+FE not apples-to-apples).
Deferred: re-FE Path A GREEN for fair circadian match; h=128 as optional sensitivity only.

---

## 1. Goal

Settle whether **multi-task glucose supervision helps T2D** when the backbone is held fixed
(teammate’s +0.003 was confounded by a larger multi-task backbone).

**Scientific claim of B1 (only):** on identical backbone, does λ>0 beat λ=0?

**Experiment:** identical sequence backbone, **λ=0 (class-only)** vs **λ>0 (class + day-level glucose)**.

**Reference numbers (informational, not sole claim):** Path A watch floor test 4-AUC **0.666** /
binary **0.689**; C1 0.738 / 0.831. Note: B1 vs Path A is **not** architecture-matched
(LSTM vs CatBoost) and B1 daily FE is more UAB site-correct than frozen GREEN — floor-beat is
context, not a pure architecture win.

---

## 2. Data contract (ready)

| Asset | Use |
|---|---|
| `features/watch_daily.parquet` | Sequence input (1824 core pids) |
| `features/cgm_daily.parquet` | Daily glucose targets (8-vector + mask) |
| `meta/pool_masks.parquet` | `label`, `recommended_split`, `wearable_core`, `aux_eligible` |

- Split: fixed `recommended_split` only.
- **T2D pool:** `wearable_core` with **T_min = 7** valid watch days (equals empirical min → n=1824).
- **Glucose loss + z-score stats:** only days with `cgm_day_valid` on pids with
  **`wearable_core ∧ aux_eligible`** (not the extra ~239 CGM pids outside aux).
- **Never** drop non-CGM / non-aux pids from T2D head.
- Day alignment: left-join on `(person_id, day_local)`; sort days ascending before pad.
- Inference: watch days only; glucose head off.

**Primary backbone features** (drop coverage counts — LUPI bleed risk):

```
hr_mean, hr_sd, hr_min, hr_max, hr_nocturnal_mean, hr_day_mean,
stress_mean, stress_sd, stress_pct_medium_plus, stress_pct_high,
rr_mean, rr_sd,
sleep_duration_hours, sleep_n_bouts,
steps_sum, mvpa_min, light_min, sedentary_min
```

(~18 dims). `hr_n` / `stress_n` / `rr_n` = **ablation-only** add-back, not primary.

Glucose target vector (8; 6 dof — collinear OK for multi-output MSE):

```
cgm_mean, cgm_sd, cgm_cv, cgm_min, cgm_max,
cgm_tir_70_180, cgm_tbr_70, cgm_tar_180
```

---

## 3. Model (controlled)

### 3.1 Backbone (locked for λ comparison)

Match teammate scale that **worked**, not the larger multi-task fail:

- Input: day features \(x_t \in \mathbb{R}^{d}\), mask \(m_t\).
- Linear proj \(d \to h\), \(h=64\).
- **BiLSTM** hidden 64 (or unidirectional LSTM 64 if BiLSTM OOM — document).
- **Attention pool** over time (mask-aware) → embedding \(z \in \mathbb{R}^{64}\).
- Optional: 1-layer Transformer encoder (2 heads, d=64) as **sensitivity**, not primary.
- Optional sensitivity: same recipe at **h=128**, λ ∈ {0, 0.5} only if cheap GPU time.

Primary architecture name: **`attn_lstm_64`**.

### 3.2 Heads

- **T2D head:** Linear(64 → 4), CE with **pinned class weights** (§3.3).
- **Glucose head (train only):**
  - **Primary (v1b / day-level):** Linear on each timestep hidden \(h_t\) → 8; **masked MSE** on
    days with `cgm_day_valid` ∧ aux_eligible. Teammate-faithful (daily 8-vector); ~11× more
    regression targets than person-mean.
  - **Secondary (v1 / person-level):** Linear(64 → 8) on \(z\) → mean of valid daily CGM vectors
    (weight by `n_valid_days` or require ≥7). Informational only — weaker gradient.

### 3.3 Loss & class weights

\[
L = L_{\text{CE}}(y,\hat y) + \lambda \cdot L_{\text{glu}}
\]

- **Class weights (lock):** inverse-frequency on **wearable_core train** label counts, normalized
  so weights sum to 1. Recompute in code from actual train `y` (seed-independent).
- \(L_{\text{glu}}\): mean MSE over 8 z-scored outputs; z-score **mean/std fit on train
  aux_eligible valid days only**. If a batch has zero glu-mask mass → skip glu term (log).
- **λ grid:** `{0, 0.3, 0.5, 1.0}`.
- Uncertainty / GradNorm: only if fixed-λ shows clear task conflict. Not default.
  **Follow-up (2026-07-16):** conflict later measured and GS (PCGrad/UW) run under
  `PLAN_B1_GS.md` / `REPORT_B1_GS.md` — class Δ still null; do not reopen plain-λ here.

### 3.4 What is held fixed across λ

Same: features, split, backbone hyperparams, optimizer, seed, ES rule, class weights. Only λ changes.

**λ=0:** still build glu head but **zero its loss** (parameter count matched). Note: with loss
zeroed, glu params get no gradient into \(z\) — capacity matching is **cosmetic for gradients**;
it only equalizes reported parameter count / init. Acceptable; document in REPORT.

---

## 4. Training protocol

| Knob | Default |
|---|---|
| Framework | PyTorch |
| Seed | 42 |
| Max epochs | 80 |
| Batch size | 32 (person-level) |
| Optimizer | AdamW |
| LR | 1e-3 with ReduceLROnPlateau on val macro-AUC |
| Weight decay | 1e-4 |
| Early stop | patience 15 on **val macro-OVR AUC** |
| Grad clip | 1.0 |
| Sequence | sort days ascending; pad to batch max ≤ **16**; mask pad |
| Missing feats | train-set median impute per feature (fit on train valid watch days only); `sleep_duration_hours` not fill-zero |
| Input scale | **train-only z-score** on `feature_cols` after impute (C2; required for LSTM) |
| Device | CUDA if available; else CPU (smoke). Full train: Lightning L4 per `COMPUTE.md` |

**No Optuna on backbone for the first controlled grid** — freeze architecture so λ is the only
primary knob. Optional later: light shared LR/dropout sweep then re-run λ grid.

---

## 5. Metrics & reporting

### Required (test once after val selection; freeze before test)

- 4-class macro-OVR AUC + AUPRC (raw softmax)
- Binary healthy-vs-not AUC from multiclass (`1−P0`)
- Per-class OVR AUC (esp. insulin — test is insulin-enriched vs train)
- **Calibration (diagnostic, Path A precedent):** fit **per-class val isotonic** on val; report
  raw + calibrated test AUC, multiclass Brier, calibration curves. Claim ranking = **raw** unless
  cal clearly dominates without harming AUC.
- Glucose: val/test MSE/MAE on z-scored + raw (aux_eligible only)

### Bootstrap (pin)

- Unit: **person** (not day)
- n_resamples = **2000**, seed **42**, two-sided 95% CI
- Within-B1 λ comparisons: **paired** by person (same resample mask)
- Vs Path A floor: CI on B1 test AUC − 0.666 (informational)

### Comparison tables

1. Within-B1: λ=0 vs each λ>0 (ΔAUC + paired boot CI) — **primary science table**
2. Path A floor vs B1 λ=0 vs B1 selected λ — **informational**
3. Train/val CE + glu loss curves

SHAP not required for B1 neural ablation.

---

## 6. Decision rules (pre-register)

**λ selection (avoid best-of-3 as sole claim):**  
For each λ, early-stop on val macro-AUC. **Primary multi-task report** is the full λ grid table.
Optional “selected λ” = argmax val AUC among {0.3,0.5,1.0} only for a single deployable
checkpoint, and must be marked as selection-biased; scientific claim uses **per-λ test CIs vs λ=0**.

| Question | Rule |
|---|---|
| Does multi-task help? | For a given λ>0: **test** paired ΔAUC (λ − λ0) bootstrap **CI lo > 0**. Report all three λ. Soft note if val Δ ≥ +0.005 but test CI includes 0 (underpowered / unstable). |
| Does B1 beat Path A floor? | **Informational:** point test 4-AUC ≥ 0.676 **and** CI lo > 0.666 preferred; else “does not clear floor +1pp under Path A-style bar.” Not the B1 scientific claim. |
| Proceed to B4? | **Always** — B1 is ablation. |
| Kill B1 early? | If λ=0 val macro-AUC ≲ 0.64 after loader/debug, fix data before full grid. |
| Architecture size confound | Primary = h=64. Optional sensitivity: one λ∈{0,0.5} at **h=128** if cheap GPU time — not required to start. |

---

## 7. Package layout

```
training/path_b/
  PLAN_B1_DATA.md      # done (FE accepted)
  PLAN_B1_TRAIN.md     # this file
  DECISIONS.md
  README.md
  b1/
    __init__.py
    config.yaml
    data.py            # load watch_daily/cgm_daily → padded tensors
    model.py           # AttnLSTM64 + heads
    train.py           # loop, ES, λ
    evaluate.py        # metrics + bootstrap
    run.py             # CLI: --lambdas 0,0.3,0.5,1.0
    artifacts/<run_id>/
```

```bash
# smoke CPU
.venv/bin/python -m training.path_b.b1.run --quick --run-id smoke_b1

# full λ grid
.venv/bin/python -m training.path_b.b1.run --lambdas 0,0.3,0.5,1.0 --run-id b1_grid_YYYYMMDD
```

---

## 8. Explicit non-goals (B1 package)

- No survey/onboarding features in primary B1 (watch-only claim track)
- No B2 two-stage, B3 KD, B4 trajectory decoder
- No 5-min grid
- No SSL pretrain
- No Optuna architecture search in first grid
- No claim that B1 is the contribution

---

## 9. Docs to update after runs

| Doc | Update |
|---|---|
| `training/path_b/DECISIONS.md` | λ outcomes, architecture locks, bugs |
| `training/path_b/REPORT_B1.md` | metrics tables (λ grid primary; Path A informational) |
| `Training.md` §7 status line | B1 done / outcome |
| `T2D.md` | short B1 pointer |
| `AGENTS.md` | report path if new |

---

## 10. Implementation order

1. Critique this train plan → address real issues. **Done.**
2. Implement `b1/data.py` + unit smoke (one batch shapes; glu only on aux days).
3. Implement model + train loop; λ=0 smoke to completion.
4. Full λ grid on available compute (local CPU smoke → Lightning GPU full).
5. Write `REPORT_B1.md` + DECISIONS; stop before B2/B4 code.

---

## 11. Open choices — **locked after critique**

| # | Choice | Lock |
|---|---|---|
| 1 | Glu head | **Day-level (v1b) primary**; person-level v1 secondary |
| 2 | λ=0 capacity | Dead glu head (zero loss); cosmetic param match — document |
| 3 | ES / select | Val **macro-OVR AUC**; full λ table is science; argmax val = deployable only |
| 4 | Coverage counts | **Out** of primary X; ablation add-back only |
| 5 | Backbone | **BiLSTM + attn, h=64** primary; uni-LSTM fallback if OOM; Transformer / h=128 sensitivity |
| 6 | Class weights | Inverse-freq on train core, sum-normalize |
| 7 | Glu pool | `aux_eligible ∩ wearable_core` valid days only |
| 8 | Path A floor | Informational reference, not pure architecture claim |
