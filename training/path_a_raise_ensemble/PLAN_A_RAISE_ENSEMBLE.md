# Plan — Path A raise: multi-seed bagging + cross-family ensemble

> **Post-freeze raise** under a **new** protocol package. Does **not** reopen or rewrite frozen Path A claims.  
> Package: `training/path_a_raise_ensemble/` (own `artifacts/` root).  
> Authority: prompt `prompts/01_path_a_raise_ensemble.md`, `Training.md` §6, C1 freeze in `path_a_blocks/REPORT_A_WRAP.md`.  
> **Status:** **executed; concluded null** (2026-07-16). See `REPORT.md`.  
> Claim runs: `ens_full_20260715_impl` (S=5), `ens_s10_20260715_impl` (S=10).  
> Critique disposition: §12.

---

## 0. Framing

| Locked | Value |
|---|---|
| Frozen claim (untouched) | W0 0.666 / 0.689; secondary **C1** 0.738 / 0.831 |
| **Baseline to beat** | **C1** `mood_scores_20260714_014415` test 4-AUC **0.7378** / binary **0.8309** |
| Raise type | **Modelchoice layer** only — same C1 feature matrix, no new blocks, no ladder reopen |
| Isolation | Import `training.path_a_blocks.data_blocks` + `training.path_a_watch.*`; **never** write into their `artifacts/` |
| New run ids | All under `training/path_a_raise_ensemble/artifacts/<run_id>/` |

**Why this raise:** the frozen ladder runs `pick_family()` and keeps **one** seed. Standard small-n tabular practice keeps seeds (and optionally both families). This package tests whether **multi-seed bagging** and/or **cross-family blend/stack** clear the pre-registered decision bar vs C1.

**Honest prior (logged):** wrap nulls sat inside ~±0.01 of C1 with CIs overlapping 0. Clearing **+0.01 absolute** on n_test=277 via seed/blend alone is **unlikely**. The protocol is built to produce a **clean claim or a clean null** — not soft-promotion.

**Not this package:** new features, GREEN v2, CORN, diet, Path B reopen, re-HPO claim grids, rewriting W0/1A/1B/1C/wrap numbers.

---

## 1. Data readiness (verified)

Cleaning / FE: **no changes needed.** Ensembling operates on the existing C1 matrix.

| Check | Result |
|---|---|
| C1 parent artifact | `training/path_a_blocks/artifacts/mood_scores_20260714_014415/` present |
| `selected_model.json` + both family `models/*.joblib` + `best_params_*.json` | present |
| Feature reload via `load_watch_onboarding_mood` + `feature_set=scores` | **47 cols, order + hash match freeze** (`d63ec5713ada37bf`) |
| Splits | train/val/test **1277 / 270 / 277** (`recommended_split`) |
| Recompute frozen CatBoost on test | **bit-match** 4-AUC 0.7377859791966677 / binary 0.8308943089430895 |
| Loser family (LGBM) params + model | present; val 4-AUC 0.7314; **`device: gpu`** in freeze |
| C1 family margin | CatBoost val 0.7439 vs LGBM 0.7314 (Δ≈0.0125) — CatBoost won; blend can drag if LGBM is strictly weaker |

**Conclusion:** parent_c1 + feature parquets + `recommended_split` are sufficient. No pipeline edit.

---

## 2. Protocol locks (inherited)

From `path_a_blocks/config.yaml` + `Training.md` §6 — **unchanged**:

- n=1824 core; fixed `recommended_split`; deny-list cols never in X  
- Class weights: LGBM `balanced`, CatBoost `Balanced`  
- CatBoost Ordered → Plain fallback; **per-seed `boosting_type` logged**  
- **LGBM device locked to `gpu`** (matches frozen C1 HPO winner; assert at runtime; full run requires GPU)  
- ES rounds / `n_estimators_max` same as blocks config  
- **Freeze before test** (refuse overwrite if `metrics_test.json` exists)  
- Claim metrics: macro-OVR 4-AUC primary; binary `1−P0` secondary  
- Paired **person-bootstrap** Δ vs C1: `n_boot=1000`, **`bootstrap_seed=42`** (absolute), **`paired_bootstrap_seed=53`** (`seed+11`, same offset as `run_1c.py`)  
- Assert `n_boot_ok ≥ 950` or fail the run (CI calibration)  
- SHAP: optional on seed-42 CatBoost member only — not a gate  

### 2.1 What changes vs frozen ladder

| Ladder (frozen) | This raise |
|---|---|
| 50-trial HPO per family → `pick_family` → 1 model | **No re-HPO for primary.** Reuse **frozen C1** `best_params_{lgbm,catboost}.json` |
| Single seed 42 | **Multi-seed bag** per family |
| Discard loser family | Keep both for blend/stack; **primary raise path is CatBoost bag first** |
| Decision bar vs previous *block* | Decision bar vs **C1** (modelchoice parent) |

**Rationale for no re-HPO primary:** isolates seed/ensemble effect from a second Optuna lottery. Optional RHPO sensitivity only if bags are flat and underfit is suspected.

---

## 3. Pre-registered arms

All arms share the **identical** C1 feature matrix and splits. Parent **B0** always asserted first.

| ID | Definition | Role |
|---|---|---|
| **B0** | Frozen C1 selected CatBoost (reload + bit-match assert, tol 1e-9) | baseline |
| **S_cat** | Single-seed CatBoost @ seed 42, frozen C1 params (refit path) | diagnostic — **assert vs B0** (log Δ; if not bit-close, document CatBoost version / path drift) |
| **S_lgbm** | Single-seed LGBM @ seed 42, frozen C1 LGBM params, **device=gpu** | diagnostic |
| **Bag_cat** | Mean proba over **S=5** CatBoost seeds | **PRIMARY A** (seed bag of winning family) |
| **Bag_lgbm** | Mean proba over **S=5** LGBM seeds | sibling / blend ingredient |
| **E_arith** | Row-wise arithmetic mean of **Bag_cat** and **Bag_lgbm** proba | **PRIMARY B** (tests whether 2nd family adds on top of bagging) |
| **E_geom** | Row-wise geometric mean of bags, **clip proba ≥ ε=1e-6**, then renormalize | sibling |
| **E_stack** | Multinomial LR on OOF bag-mean family probs; C on val; test once | sibling (exploratory if only this clears) |

### 3.1 Seeds (locked)

```
seeds = [42, 43, 44, 45, 46]   # S=5 primary
```

- Seed 42 included so the frozen HPO seed is a bag member.  
- **Pre-committed S=10 rule:** if after S=5 full run, **primary-A or primary-B** has point ΔAUC ∈ (0, +0.01] (near bar, c1 fail) **or** c1 pass but c2 fail, run S=10 (`42..51`) as a **mandatory follow-up** under a new run_id (not optional peek-only). Otherwise S=10 stays optional.  
- **No val-subsample bagging in primary.**

### 3.2 Per-seed fit recipe

For each `(family, seed)`:

1. `make_*` with frozen C1 params; `random_state` / `random_seed` = **seed**.  
2. LGBM: **`device="gpu"`** always (assert).  
3. Fit on train; early-stop on **val** (same ES protocol as Path A).  
4. Persist model; log `best_iteration`, CatBoost `boosting_type`.  
5. Predict proba train/val/test; store arrays + person_id index.

**Bag_*** = elementwise mean over seeds.

**Note (honest):** all seeds ES on the **same** val → bag is “5 seed-perturbations of one val-selected solution,” not independent data partitions. Do not over-attribute a pass to classical ensemble theory.

### 3.3 Stacker (`E_stack`) — leakage rules (revised)

Hard locks:

1. Stacker features = **predicted class probabilities** (4 per family → 8 features).  
2. **Train OOF mirrors val/test generative regime (bag-mean):**  
   - K=5 stratified by label on train, `seed=42`.  
   - Assert min class count per fold ≥ 1 (log counts; fail if any class missing in a fold).  
   - For each fold *k*: for each seed in `seeds`, fit family models on train\fold with **ES on fold holdout** (nested val = fold holdout); average seed probas → OOF bag-mean features for persons in fold *k*.  
   - Cost: K × S × 2 ≈ **50** tree fits (still no Optuna; under one Path A HPO wall-clock).  
3. Val/test stacker inputs = **Bag_lgbm / Bag_cat** (same objects as `E_arith`).  
4. Stacker: `LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000, class_weight="balanced")`.  
   - `C ∈ {0.01, 0.1, 1.0, 10.0}` selected on **val** macro-OVR AUC.  
5. Freeze chosen `C` + coefs → score test **once**. Never fit on test.  
6. **Degeneracy / collapse (explicit):**  
   - Family L2-norm ratio (min family block ‖β‖₂ / max family block ‖β‖₂) **≤ 0.05** → degenerate;  
   - **or** val stacker AUC < max(val Bag_cat, val Bag_lgbm) − 0.005 → collapse.  
   - On either: mark `E_stack` `claim_eligible=false`, report, do not promote.

**σ-stacking note:** logistic stacking on probability features (not variance-weighted).

### 3.4 Optional consistency read (does not move point numbers)

- Per-seed test 4-AUC table (post-hoc; **not** used to pick seeds).  
- Sign count of ΔAUC(seed − C1).  
- Does **not** replace single-test paired CI.

---

## 4. Decision bar (modelchoice vs C1)

Same numeric gates as Path A blocks, parent = **C1**:

| Criterion | Rule |
|---|---|
| **c1** | point Δ macro-OVR AUC(test) **> +0.01** vs C1 |
| **c2** | paired person-bootstrap ΔAUC **ci_lower > 0** (`paired_bootstrap_seed=53`, n_boot=1000, n_boot_ok≥950) |
| **c3** | **modelchoice superiority / consistency** — arm-specific (below) |

### 4.1 Dual primary (pre-registered)

| Primary | Arm | c3 |
|---|---|---|
| **A — seed bag** | **`Bag_cat`** | Multi-seed consistency: ≥4/5 seeds have **val** 4-AUC ≥ C1 freeze val − 0.01 **and** mean(Bag_cat seed val AUCs) ≥ S_cat val − 0.002. (Does **not** require beating LGBM.) |
| **B — cross-family** | **`E_arith`** | Blend superiority: test paired-bootstrap Δ(E_arith − best_single_bag) has **ci_lower > 0** where `best_single_bag = argmax_val(Bag_cat, Bag_lgbm)` **or** point test AUC(E_arith) ≥ max(test Bag_*) + 0.005. If c3 fails but Bag_cat passes A: **unpack** — seed bag is the raise; blend does not add. |

**Pass A:** c1 ∧ c2 ∧ c3_A on `Bag_cat`.  
**Pass B:** c1 ∧ c2 ∧ c3_B on `E_arith` (implies blend helps beyond best bag).

**Reporting discipline:**

- Evaluate both primaries; both may pass, either, or neither.  
- **No max-of-arms primary.** Do not take `max(Bag_cat, E_arith, E_stack)` as the claim.  
- `E_geom` / `E_stack` / `Bag_lgbm`: siblings. If only a sibling clears c1∧c2: mark **`claim_eligible: false / exploratory`** in manifest — requires a **new PLAN_*** to re-nominate as primary.  
- Binary: always report; **never** promote on binary alone if 4-AUC bar fails.

### 4.2 Outcome grid (named)

| Outcome | Write as |
|---|---|
| Pass A only | **Seed-bag raise accepted** (`Bag_cat`); C1 remains ladder parent; new secondary modelchoice = multi-seed CatBoost |
| Pass B (with or without A) | **Cross-family ensemble raise** (`E_arith`); document whether A also passed |
| c1∧c2 on Bag_cat, c3_B fail on E_arith | **Bag wins; blend does not dominate members** — claim is seed-bag, not ensemble |
| Point up, c2 fail (A or B) | **Near-miss / noise floor** — trigger S=10 if Δ∈(0,+0.01] or c1∧¬c2 |
| Flat / down all primaries | **Ensemble/bag null** — single-seed C1 remains best honest tabular; close raise |
| Only sibling (stack/geom) clears | **Exploratory only** — not claim-grade |

---

## 5. Package layout

```
training/path_a_raise_ensemble/
  PLAN_A_RAISE_ENSEMBLE.md   # this file
  DECISIONS.md               # living log
  README.md                  # how to run
  config.yaml                # seeds, parent pin, device, bootstrap seeds
  __init__.py
  __main__.py
  data.py                    # load C1 matrix + assert parent
  bag.py                     # multi-seed fit / mean proba
  stack.py                   # OOF bag-mean + logistic stacker
  ensemble.py                # arith / geom (ε-clip)
  metrics_raise.py           # paired Δ vs C1, c1/c2/c3, tables
  run.py                     # smoke + full orchestration
  artifacts/
```

**Imports only (no edits):**

- `training.path_a_blocks.data_blocks`  
- `training.path_a_watch.models` / `metrics` / `evaluate.write_json`  
- `training.path_a_blocks.diagnostics` — `bootstrap_ci`, `paired_delta_bootstrap`  
- Parent artifacts **read-only**

**config.yaml pins (assert at runtime):**

```yaml
parent_c1:
  run_id: mood_scores_20260714_014415
  test_macro_ovr_auc: 0.7377859791966677
  test_binary_auc: 0.8308943089430895
  feature_hash: d63ec5713ada37bf
  n_features: 47
  feature_set: scores
  family_selected: catboost
  lgbm_device: gpu          # must match best_params_lgbm.json
  catboost_best_iteration: 72
  lgbm_best_iteration: 103
run:
  seeds: [42, 43, 44, 45, 46]
  bootstrap_seed: 42
  paired_bootstrap_seed: 53   # 42+11, matches run_1c
  bootstrap_n: 1000
  min_n_boot_ok: 950
  lgbm_device: gpu
  geom_eps: 1.0e-6
stack:
  k_folds: 5
  C_grid: [0.01, 0.1, 1.0, 10.0]
  oof_mode: bag_mean_es_on_fold   # revised post-critique
```

Parent assert checks: metrics + feature_hash + feature_cols **order** + `feature_set==scores` + n_features + lgbm device pin + B0 recompute bit-match.

---

## 6. Run protocol

```bash
export DRI_PRIME=1
cd /path/to/T2D

# Smoke: 2 seeds, 2-fold stack, claim_eligible=false
.venv/bin/python -m training.path_a_raise_ensemble --quick --run-id ens_smoke_<ts>

# Full primary (requires GPU for LGBM)
.venv/bin/python -m training.path_a_raise_ensemble --run-id ens_full_<ts>
```

### 6.1 Orchestration order

1. Load config; log `DRI_PRIME`; create artifacts dir.  
2. **Parent assert** (metrics, hash, order, feature_set, device pin, B0 bit-match).  
3. Load C1 matrix; splits.  
4. Fit seed grid → bags (log boosting_type / device / best_iteration per seed).  
5. Build `E_arith`, `E_geom` (ε-clip).  
6. Build train-OOF bag-mean + fit `E_stack` (val-select C; degeneracy check).  
7. **Freeze** `selected_ensemble.json` **before** test metrics write.  
8. Score all arms on test; paired Δ vs B0; c1/c2/c3 for primaries A/B.  
9. Write metrics, REPORT, DECISIONS.  

### 6.2 Smoke gates (`--quick`)

- seeds → `[42, 43]`; stack K → 2; C_grid → `{1.0}`  
- Parent bit-match still required  
- `claim_eligible: false`

### 6.3 Artifacts (per full run)

| Path | Content |
|---|---|
| `run_manifest.json` | seeds, parent pin, git hash, claim_eligible, GPU/device |
| `parent_assert.json` | bit-match proof |
| `features.json` | cols + hash + feature_set |
| `models/{family}_seed{s}.joblib` | bag members |
| `proba/…` | per-seed + bag/ensemble + pid index |
| `stacker.joblib` + `stacker_meta.json` | C, coefs, OOF, degeneracy flags |
| `selected_ensemble.json` | freeze record |
| `metrics_val.json` / `metrics_test.json` | all arms + bar |
| `REPORT.md` | human table |

---

## 7. Compute budget (estimate)

| Step | Fits |
|---|---:|
| Bag S=5 × 2 families | 10 |
| Stack OOF K=5 × S=5 × 2 families | 50 |
| **Total primary** | **~60** tree fits (no Optuna) |

Wall-clock: still under one Path A 50×2 HPO run. **GPU required** for LGBM arm.

---

## 8. Out of scope / non-goals

- Editing `path_a_blocks` / `path_a_watch` code or artifacts  
- Re-running W0 / 1A / 1B / 1C / wrap claim grids  
- Claiming Path A ladder numbers “changed” if this raise fails or is not run  
- New feature blocks  
- Nested CV replacement of fixed `recommended_split`  
- Soft-promising binary-only lifts  
- Full combinatorial seed×HPO×blend grid  
- Taking max-of-arms as the claim  

### Optional sensitivities (non-primary unless new PLAN)

| ID | What | When |
|---|---|---|
| S10 | seeds 42..51 | **mandatory** if near-bar rule (§3.1); else optional |
| RHPO | 20-trial re-HPO then bag | bag null + underfit suspected |
| VSAMP | val-subsample diversity | seed bag highly correlated |
| WMEAN | val-AUC-weighted family blend | arith null, one family much stronger on val |

---

## 9. Success / failure (summary)

See §4.2 outcome grid. Either way: package `REPORT.md` + `DECISIONS.md` with numbers and run ids.  
**Do not** edit `REPORT_A_WRAP.md` claim table; optional one-line chronology pointer in `path_a_blocks/DECISIONS.md` only after full run completes.

---

## 10. Implementation order (after user go)

1. Critique disposition already in §12; user ack.  
2. Scaffold package + config + parent assert (incl. GPU pin).  
3. `bag.py` + smoke 2-seed fit.  
4. `ensemble.py` arith/geom.  
5. `stack.py` OOF bag-mean + LR.  
6. `run.py` full orchestration + dual-primary metrics.  
7. Smoke → full → REPORT/DECISIONS.  

---

## 11. Open points resolved by critique

| # | Was open | Resolution |
|---|---|---|
| 1 | Primary = E_arith only? | **No** — dual primary: **Bag_cat (A)** + **E_arith (B)** |
| 2 | OOF fixed-iter vs ES | **OOF bag-mean with ES on fold holdout** (mirrors val/test) |
| 3 | c3 adaptation | **Superiority/consistency**, not vacuous within-0.02 |
| 4 | No re-HPO primary | **Kept** |
| 5 | S=5 vs S=10 | S=5 primary; **pre-committed S=10 near-bar rule** |

---

## 12. Critique disposition (glm-5.2, fresh)

| Finding | Severity | Disposition |
|---|---|---|
| Primary E_arith handicaps vs Bag_cat; Bag_cat not primary | **High** | **Accepted** — dual primary A=`Bag_cat`, B=`E_arith` |
| c3 vacuous (within 0.02 OR drop-one 0.002) | **High** | **Accepted** — replaced with superiority/consistency c3 |
| Stacker OOF fixed-iter ≠ bag-mean test regime | **High** | **Accepted** — OOF = bag-mean ES-on-fold |
| LGBM device gpu not locked; “no GPU” false | **High** | **Accepted** — pin + assert `lgbm_device: gpu` |
| Bootstrap seed not pinned | **Medium** | **Accepted** — 42 / 53 (run_1c offset) |
| Val-tiered c3 noise from shared ES | **Medium** | **Accepted** — c3_B uses **test** paired bootstrap vs best bag |
| Sibling pass after primary fail = garden path | **Medium** | **Accepted** — `claim_eligible: false / exploratory` |
| E_geom zero proba | **Missing** | **Accepted** — ε=1e-6 clip |
| Fold class counts / n_boot_ok | **Missing** | **Accepted** — asserts |
| S_cat vs B0 bit-close check | **Risk** | **Accepted** — diagnostic assert/log |
| S=10 pre-commit near-bar | **Risk** | **Accepted** — mandatory follow-up rule |
| Drop “no GPU required” | **Risk** | **Accepted** |

**False positives / dropped:** none material. Nits on soft wording folded into §0 honest prior.

**Verdict after address:** plan is implementable. Awaiting user go before code.  
