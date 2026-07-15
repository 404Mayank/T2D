# Plan — Path A raise: CORN/CORAL ordinal head

> **Status:** **IMPLEMENTED / CONCLUDED NULL** (2026-07-16). Authority for numbers: `REPORT.md`.  
> **Package:** `training/path_a_raise_corn/` (isolated artifacts).  
> **Parent / bar comparator:** frozen **C1** (`mood_scores_20260714_014415`).  
> **Authority:** `Training.md` §2/§6/§7/§9, `FEATURES.md`, `REPORT_A_WRAP.md`, `path_a_blocks/config.yaml`.  
> **Not a reopen of the frozen claim.** New run ids; C1/W0 unchanged.  
> **Plan critique:** fresh `critiquer` → **revise** (see §10). **Null audit:** primary null **JUSTIFIED** (see `DECISIONS.md`).
>
> | Arm | Run | Test 4-AUC | Δ vs C1 | Bar |
> |---|---|---:|---:|---|
> | CORN | `corn_full_20260715_211707` | **0.706** | **−0.031** (CI entirely &lt;0) | **FAIL** |
> | CE control | `ce_mlp_full_20260715_211707` | **0.713** | −0.025 | n/a |
> | C1 parent | frozen | **0.738** | 0 | — |

---

## 0. Why this raise

Path A tabular is **frozen** (`REPORT_A_WRAP.md`):

| Role | Number |
|---|---:|
| Watch-only W0 | 4-AUC **0.666** / binary **0.689** |
| Secondary deployable **C1** | 4-AUC **0.738** / binary **0.831** |

Diagnosed 4-class ceiling: **class-2 (oral/non-insulin injectable) OVR ~0.62–0.63** across every wrap stack — the mid-severity sandwich that independent OVR softmax handles poorly. The 4-class label is **ordinal**. `Training.md` §2/§7/§9 already list **CORN ordinal regression** as required/lean-CORN and it was **never run**.

This raise fits a **tabular CORN MLP head** on the **exact C1 feature matrix** so Δ vs C1 is apples-to-apples.

**Claim cells (locked):**

| Priority | Cell | Role |
|---|---|---|
| **Primary** | CORN MLP | ordinal raise vs C1 bar |
| **Required control** | CE-MLP (same backbone + impute/scale) | isolates ordinal loss vs architecture |
| Optional sibling | CORAL MLP | proportional-odds ablation |
| Optional late | OOF/σ blend CORN⊕C1 | only if CORN competitive |

Without the CE-MLP control, a CORN>C1 win is unattributable (ordinal vs MLP-vs-GBM vs impute). B3 already has the CE student pattern — near-zero marginal cost.

---

## 1. Data readiness (judgement — no pipeline change)

### 1.1 Matrix is already present

| Source | Role |
|---|---|
| `data/processed/features/watch_green.parquet` | 30 GREEN watch features (0 nulls) |
| `data/processed/features/onboarding.parquet` | hard onboarding keep list |
| `data/processed/features/mood.parquet` | `cestl`, `paidscore` |
| `data/processed/meta/pool_masks.parquet` | `label`, `recommended_split`, pool flags |
| C1 artifact `features.json` | **canonical** 47-col order + `feature_hash=d63ec5713ada37bf` |

Load via **import** of `training.path_a_blocks.data_blocks.load_watch_onboarding_mood` + `resolve_mood_cols(..., "scores")`. **Do not edit** `path_a_blocks` / `path_a_watch` code or write into their `artifacts/`.

**Hard assert at run start:**
```
feature_hash(feature_cols) == "d63ec5713ada37bf"
n_total == 47
n == 1824
```
Mismatch → abort (do not train on a drifted matrix).

### 1.2 Missingness (the one real FE judgement)

C1 null profile (from frozen `features.json` + recompute):

| Col group | Max null frac | Notes |
|---|---:|---|
| `fh_dm2sb` / `fh_dm2pt` | ~10% / ~9% | binary 0/1; highest missing |
| `paidscore` | ~1.9% | continuous-ish score |
| anthropometrics / BP / pulse | ≤1.4% | continuous |
| `cestl` | ~0.16% | continuous |
| **all watch GREEN** | **0** | clean |

- Rows with any null: **345 / 1824** (~19%); train 236/1277, val 60/270, test 49/277.
- GBMs ate NaN natively; **MLP cannot**.

### 1.3 Imputation policy (locked in this plan)

**Decision: train-split median impute + z-score, no missingness mask layer for v1.**

Rationale:
1. Missingness is **sparse and mostly onboarding/mood**, not systematic watch dropout. A mask layer doubles input dim (47→94) and burns capacity at **n_train=1277** for a small expected gain.
2. Project already uses this exact recipe on C1 for B3 student MLP (`path_b/b3/student_mlp.py::_impute_scale_fit`): `median(train) → fillna → mean/std(train) → z`. Reuse the pattern so Path A/B student tables stay comparable.
3. Binary FH cols (`fh_dm2pt`, `fh_dm2sb`): train median is **0** (positives ~40% / ~26%) — this is **majority fill**, not “mode-ish on a near-balanced bit.” It **attenuates** a top-ranked C1 signal (~9–10% of rows forced to 0). Acceptable for v1 to match B3 and avoid a second impute path; **must** log post-impute positive rates at smoke and optionally complete-case val AUC if class-2 is near-bar. Do **not** invent kNN/MICE unless that diagnostic fires.
4. **Fit medians / mean / std on train only.** Apply to val/test. Persist `scale_state.json` in the run artifact.
5. **Out of scope for v1:** kNN impute, MICE, learnable missing embeddings, per-label medians (leakage risk), missingness mask layer.
6. **Tradeoff noted:** a GBM with an ordinal objective would keep native-NaN handling and stay GBM-vs-GBM. Out of scope — `Training.md` prefers **neural CORN/CORAL**; this raise answers that open item, not “best ordinal tabular overall.”

**No cleaning-pipeline change.** No new feature columns. No re-run of convert/clean.

### 1.4 Label / splits

- `label` ∈ {0,1,2,3}, already 0-indexed (CORN-ready).
- Fixed `recommended_split`: train **1277** / val **270** / test **277**.
- Class counts overall: 636 / 453 / 536 / 199 (insulin minority binds).

---

## 2. Method

### 2.1 Primary: CORN MLP

Library: **`coral-pytorch==1.4.0`** (pin in package `requirements.txt`; not currently installed).

| Piece | Spec |
|---|---|
| Output dim | `num_classes - 1 = 3` |
| Loss | `coral_pytorch.losses.corn_loss(logits, y, num_classes=4)` |
| Hard labels | `corn_label_from_logits(logits)` (cumprod of sigmoids > 0.5) |
| **Class probabilities (required for our metrics)** | Convert CORN conditional logits → rank-consistent 4-vector (see §2.2). **Do not** report macro-OVR on hard labels only. |

Architecture (regularized tabular MLP; small-n bias toward simple):

```
Linear(d_in=47, h) → GELU/ReLU → Dropout(p)
→ Linear(h, h) → GELU/ReLU → Dropout(p)
→ Linear(h, 3)   # CORN head
```

Defaults (HPO may move inside ranges):

| Hyper | Default / range |
|---|---|
| `hidden` | {32, 64, 128} (default 64) |
| `dropout` | {0.3, 0.4, 0.5} (default **0.4** — heavy) |
| `weight_decay` | log-uniform ~1e-4 … 1e-2 (default 1e-3) |
| `lr` | log-uniform ~3e-4 … 3e-3 (default 1e-3) |
| `batch_size` | {64, 128} |
| `epochs` max | 200 |
| early stop | patience **15** on **val macro-OVR AUC** (min_delta 1e-4); track val macro-AUPRC |
| optimizer | AdamW |
| seed | **42** (+ optional multi-seed stability: 43, 44 report-only) |
| HPO budget | **40** Optuna trials **shared pattern**: 20 CORN + 20 CE-MLP (same space minus head/loss); smoke=2 total |

**Class imbalance (locked — no WeightedRandomSampler):**

`corn_loss` averages BCE over **conditional** task subsets. A class-balancing sampler rewrites the empirical P(Y>k | Y≥k) (e.g. pushes P(Y>2|Y≥2) toward 0.5) and can **hurt class-2 OVR** — the diagnosed failure mode. It also mismatches B3 (loss-level `weight=` on CE, not a sampler).

**Primary:** implement a thin **weighted CORN loss** wrapper:
- person weight `w_i = n / (K * n_{y_i})` from **train** counts (Path A balanced spirit);
- inside each conditional task, multiply per-row BCE by `w_i`, then mean;
- keep val/test unweighted (natural base rates) for early stop + metrics.

Official unweighted `corn_loss` remains available as a smoke ablation flag (`imbalance: none|weighted_corn`), default **`weighted_corn`**.

Import paths (pin):
```python
from coral_pytorch.losses import corn_loss, coral_loss
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch
```
If install fails: hand-port ~15-line `corn_loss` from coral-pytorch 1.4.0 source into `losses_proba.py` (MIT) and note in DECISIONS — do not block the raise.

### 2.2 Logits → 4-class probability (metric contract)

CORN models **conditional** P(Y > k | Y ≥ k). With `s = sigmoid(logits)` and cumprod chain:

```
# conditional s_k = P(Y > k | Y ≥ k), k=0..2
# unconditional rank-survival u_k = P(Y > k) = ∏_{j≤k} s_j
u = cumprod(s, dim=1)          # shape (n, 3) = [P(Y>0), P(Y>1), P(Y>2)]
P0 = 1 - u[:, 0]
P1 = u[:, 0] - u[:, 1]
P2 = u[:, 1] - u[:, 2]
P3 = u[:, 2]
proba = stack([P0,P1,P2,P3], -1)
# numerical clamp + renormalize to sum 1
```

This matches the chain rule used inside `corn_label_from_logits` and yields a proper simplex for:
- macro-OVR AUC / AUPRC / per-class OVR (incl. **class-2**)
- binary = `1 - P0` (same derived definition as Path A)
- calibration (temperature / sigmoid-on-binary diagnostic; multiclass cal secondary)

Unit tests (fixed logit fixture):
- `proba.sum(1) ≈ 1`, all `proba ≥ 0` after clamp/renorm
- `u` nonincreasing along rank (cumprod of sigmoids in [0,1])
- **hard label** from our helper equals `corn_label_from_logits` (same `count(u > 0.5)` rule)

**Do not** assert `argmax(proba) == corn_label_from_logits`. Counterexample: logits `[0.3, 2.0, -0.5]` → hard label **2** via thresholded survival, but `argmax(P)` = **0**. Mode of the simplex ≠ rank-threshold label; both are valid objects — metrics use **proba**, ordinal MAE/QWK use **hard labels**.

### 2.3 Required control: CE-MLP (same package)

Same impute/scale, same backbone hyper ranges, **output dim = 4**, `F.cross_entropy` with balanced `weight=` (B3-style). Softmax proba → same `full_report` + paired Δ vs C1.

**Interpretation rules:**
- CORN beats C1 **and** CORN beats CE-MLP → ordinal structure credited.
- CORN beats C1 but **not** CE-MLP → “MLP beat GBM”; ordinal claim **not** supported.
- CE-MLP beats C1, CORN does not → architecture/impute story; close ordinal raise as null.
- Both null vs C1 → publishable small-n null.

CE-MLP is **not** bar-eligible as a new deployable stack by itself (Path A deployable stays GBM-shaped unless a later plan says otherwise); it is an **attribution control**.

### 2.4 Sibling ablation: CORAL MLP (optional, same package)

Same backbone; head still dim=3; loss = `coral_loss` with `levels_from_labelbatch`. Proba via CORAL extended-binary → rank-consistent conversion. **Not** the primary claim. Run only after CORN + CE full results, if CORN is competitive and we want proportional-odds contrast.

### 2.5 Optional late: OOF σ-stack with frozen C1 GBM

Only if CORN alone is competitive (val within ~0.02 of C1 val) or beats C1 on class-2:
- 5-fold **person** OOF on train+val **or** simple val-fit logistic / temperature blend of `[p_corn, p_c1]` → test once.
- Must not peek test for blend weights.
- Report as **secondary** cell; primary remains CORN alone.

### 2.6 Explicitly out

- New feature blocks (diet, smoking, comorb reopen, via, …)
- Editing C1 / W0 / wrap artifacts or code paths
- Nested CV / re-split
- Claiming Path A numbers moved without a completed raise run id
- Cold-start raw CNN / Path B reopen

---

## 3. Protocol locks (mirror Path A §6)

| Lock | Spec |
|---|---|
| Parent | C1 CatBoost `mood_scores_20260714_014415` |
| Parent assert | `metrics_test.json` matches `parent_c1_reference` (tol 1e-9) **and** recompute test proba macro-OVR within 1e-6 |
| Feature matrix | exact C1 cols + hash |
| Select | **val macro-OVR AUC** primary; val macro-AUPRC tie-break (eps 0.005) |
| Freeze | write `selected_model.json` + weights **before** test metrics |
| Test | **once** after freeze |
| Bootstrap | person-bootstrap n=1000, seed 42, α=0.05; **paired Δ** vs C1 parent proba on same test person_ids (`paired_delta_bootstrap` from `path_a_blocks.diagnostics`) |
| Decision bar (pre-registered, **CORN only**) | (1) point Δ macro-OVR AUC **> +0.01** vs C1 **and** (2) paired boot ΔAUC **lo > 0** **and** (3) perm guardrail not pure noise (see §3.1). CE-MLP is control-only (no deployable bar). |
| Attribution | Report CORN vs CE-MLP Δ on test (point + paired boot); required for any positive CORN narrative |
| Soft win (documented, not bar) | class-2 OVR **Δ ≥ +0.03** vs C1 class-2 (0.638) even if macro bar fails — publishable mechanistic note |
| Null is publishable | GBM-usually-wins at n=1824; documented tie/null OK |

### 3.1 SHAP / perm guardrail for MLP

Tree SHAP does not apply. Plan:
1. **Permutation importance on val** (macro-OVR drop; same spirit as Path A c3) — primary stability check.
2. **Gradient×input or Captum Integrated Gradients** optional; if heavy, skip and rely on perm.
3. Guardrail narrative: if top signal is only `paidscore` + anthro (same as C1), ordinal head did not invent a new wearable story — still valid as method raise if metrics pass.

### 3.2 Calibration

- Fit temperature or per-class sigmoid on **val** proba (reuse `path_a_watch.calibrate.fit_calibrators` if shape matches; else temperature on logits before CORN→proba).
- Report raw + cal test metrics; **claim numbers = raw** (Path A convention for C1 selected_raw), cal diagnostic.

### 3.3 Multi-seed

- Primary claim seed **42**.
- If bar-pass on seed 42: run seeds 43/44 as **stability** (report mean±sd; do not re-select).
- If null on 42: still optional 1 extra seed to show variance; do not fish for a passing seed.

---

## 4. Package layout

```
training/path_a_raise_corn/
  PLAN_A_RAISE_CORN.md      # this file
  DECISIONS.md              # living log
  REPORT.md                 # filled after full run
  README.md                 # how to run
  requirements.txt          # coral-pytorch==1.4.0 (+ note torch already present)
  config.yaml               # seeds, HPO ranges, paths, parent refs, bar
  __init__.py
  __main__.py               # python -m training.path_a_raise_corn
  data.py                   # load C1 matrix, hash assert, impute/scale
  model.py                  # CornMLP / CoralMLP
  losses_proba.py           # corn/coral proba conversion + unit checks
  train.py                  # train loop, early stop, HPO
  evaluate.py               # full_report, bootstrap Δ, bar
  explain.py                # val permutation importance
  run.py                    # smoke / full orchestration
  artifacts/                # gitignored run roots
```

**Imports allowed:** `training.path_a_blocks.data_blocks`, `training.path_a_blocks.diagnostics`, `training.path_a_watch.{metrics,calibrate,evaluate,data}` — read-only.

**Artifacts root:** `training/path_a_raise_corn/artifacts/<run_id>/` only.

---

## 5. Config (sketch)

```yaml
run:
  seed: 42
  n_trials_per_arm: 20  # CORN + CE-MLP; smoke=2 total
  max_epochs: 200
  patience: 15
  min_delta_auc: 1.0e-4
  imbalance: weighted_corn   # none | weighted_corn
  bootstrap_n: 1000
  bootstrap_ci: 0.95
  device: auto          # cuda if available else cpu

paths:
  # reuse path_a_blocks paths by relative repo paths
  watch_green: data/processed/features/watch_green.parquet
  onboarding: data/processed/features/onboarding.parquet
  mood: data/processed/features/mood.parquet
  pool_masks: data/processed/meta/pool_masks.parquet
  artifacts_root: training/path_a_raise_corn/artifacts
  parent_c1_artifacts: training/path_a_blocks/artifacts
  parent_c1_run_id: mood_scores_20260714_014415

data:
  expected_n: 1824
  n_features: 47
  feature_hash: d63ec5713ada37bf
  n_classes: 4
  impute: train_median
  scale: zscore_train

parent_c1_reference:
  test_macro_ovr_auc: 0.7377859791966677
  test_binary_auc: 0.8308943089430895
  test_macro_auprc: 0.4687470484302076
  family: catboost
  class2_ovr_auc: 0.6377640845070423

decision_bar:
  delta_macro_ovr_auc_gt: 0.01
  require_boot_lo_gt_0: true
  soft_class2_delta: 0.03
```

---

## 6. Execution order

| Step | Action | Gate |
|---|---|---|
| E0 | Write package skeleton + `losses_proba` unit tests | tests pass without data |
| E1 | `data.py` load + hash assert + impute/scale | smoke print nulls=0 after impute; hash match |
| E2 | Install `coral-pytorch==1.4.0` | import OK |
| E3 | **Smoke fit** (`--quick`: fixed hparams, ≤15 epochs; CORN + CE one-shot; skip heavy perm) | finishes; proba (n,4); FH post-impute rates logged; `smoke_*` run id |
| E4 | Full HPO **CORN** then **CE-MLP** (20 trials each); freeze each arm before its test | freeze files before `metrics_test` |
| E5 | Paired bootstrap Δ vs C1 (both arms); CORN vs CE Δ; perm on CORN; REPORT + DECISIONS | bar true/false + attribution explicit |
| E6 (opt) | CORAL and/or σ-stack | only if E5 warrants |

Run ids:
- smoke: `corn_smoke_YYYYMMDD_HHMMSS`
- full CORN: `corn_full_YYYYMMDD_HHMMSS`
- CE control: `ce_mlp_full_YYYYMMDD_HHMMSS`
- coral: `coral_full_...`
- blend: `corn_c1_blend_...`

**Contract:** `proba` column 0 is always P(class=0) so `paired_delta_bootstrap` binary path `1 - p[:,0]` stays valid.

---

## 7. Success / failure framing

| Outcome | Framing |
|---|---|
| CORN bar pass **and** CORN > CE-MLP | Ordinal head raises Path A secondary; C1 historical freeze kept; new run = optional stack note |
| CORN bar pass **but** CE-MLP ≥ CORN | MLP/impute story, **not** ordinal; do not sell CORN as the raise |
| Soft class-2 lift ≥0.03, macro bar fail | Mechanistic note; **do not** replace C1 |
| Both null / tie | Publishable small-n null; Path A freeze unchanged |
| Clear underperform | Expected GBM prior; close raise |

**Never** rewrite `REPORT_A_WRAP.md` claim table as if C1 moved. Append a “post-freeze raises” section or keep numbers only in this package’s `REPORT.md` + index note in `AGENTS.md` / `DECISIONS.md`.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CORN proba conversion off-by-one / not simplex | Unit tests (sum-to-1, ≥0, hard-label match library; **no** argmax≡hard) |
| Imputation attenuates FH | Log post-impute rates; complete-case val if near-bar |
| Sampler distorts conditionals | **No sampler**; weighted CORN loss default |
| Positive result unattributable | Required CE-MLP control + interpretation rules |
| Overfit MLP at n=1277 | Heavy dropout, WD, patience 15 + min_delta, small h |
| HPO peeks test | Freeze-before-test enforced in `run.py` |
| coral-pytorch API drift / install | Pin 1.4.0; single import module; hand-port fallback |
| ROCm/CUDA quirks | CPU fallback; disable cudnn if needed |
| Cal vs raw | Claim on raw; cal diagnostic |
| σ-stack leakage | Val-only blend weights; test once |
| `proba[:,0]` contract break | Assert column order in evaluate before bootstrap |

---

## 9. Critique checklist (for subagent)

Please attack:
1. Is train-median impute the right call vs mask layer given ~10% FH missing?
2. Is WeightedRandomSampler enough for CORN’s conditional imbalance?
3. Is the CORN→4-class proba formula correct / rank-consistent?
4. Is 30 Optuna trials + patience 20 adequate or underpowered / overfit-prone?
5. Decision bar vs C1 (GBM) fair for an MLP, or should we also report vs a CE-MLP control?
6. Any protocol leak (test in impute, hash, freeze order)?
7. Scope creep (blend/CORAL) risking delayed primary result?

---

## 10. Disposition log (post-critique)

**Critique:** fresh `critiquer`, model `opencode-go/glm-5.2:high`, verdict **revise**.

| Finding | Sev | Disposition |
|---|---|---|
| `argmax(proba) == corn_label_from_logits` unit test is false | Blocker | **Fixed** — drop argmax test; keep sum/≥0 + hard-label match via `count(u>0.5)` |
| No CE-MLP control → positive CORN unattributable | High | **Fixed** — CE-MLP required control + interpretation rules; 20 trials/arm |
| WeightedRandomSampler distorts CORN conditionals; ≠ B3 | High | **Fixed** — default `weighted_corn` loss wrapper; sampler out |
| FH median=0 attenuates top signal; “mode-ish” understated | Medium | **Fixed** — wording + smoke rate log + complete-case optional |
| Patience 20 long for noisy val | Medium | **Fixed** — patience 15 + min_delta 1e-4 |
| Import path / install fallback | Low | **Fixed** — explicit imports + hand-port note |
| GBM-ordinal alternative unmentioned | Low | **Noted** as out-of-scope tradeoff (§1.3.6) |
| Early-stop under sampler | n/a | **Dropped** with sampler |
| numpy/joblib env drift on parent load | Low | **Accept** — parent recompute assert already fails loud |

**False positives / nits dropped:** none material; critique was tight.

**Plan status:** implemented; full runs concluded **null** (`REPORT.md`). Optional follow-ups (unweighted CORN / impute sens / multi-seed) need a new go — not required to close the raise.
