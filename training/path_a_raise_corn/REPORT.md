# Path A raise CORN — final report

**Status:** **concluded null** (decision bar fail).  
**Protocol:** post-freeze raise on exact C1 matrix; does **not** change frozen Path A claim numbers.  
**Parent:** C1 CatBoost `mood_scores_20260714_014415` — test 4-AUC **0.7378** / binary **0.8309** / class-2 OVR **0.6378**.  
**Feature hash:** `d63ec5713ada37bf` (47 cols; asserted).

---

## 1. Headline

| Arm | Run id | Val 4-AUC | **Test 4-AUC** | Test binary | Class-2 OVR | ΔAUC vs C1 | Bar |
|---|---|---:|---:|---:|---:|---:|---|
| **CORN MLP** (primary) | `corn_full_20260715_211707` | 0.7416 | **0.7063** | 0.7628 | 0.6312 | **−0.0314** | **FAIL** |
| CE-MLP (control) | `ce_mlp_full_20260715_211707` | 0.7416 | **0.7128** | 0.7695 | 0.6531 | −0.0249 | n/a (control) |
| C1 parent (GBM) | `mood_scores_20260714_014415` | — | **0.7378** | **0.8309** | **0.6378** | 0 | — |

**Decision bar (CORN only):** point ΔAUC > +0.01 ✗ · paired boot ΔAUC lo > 0 ✗ (CI **[−0.058, −0.005]**, entirely &lt; 0) · perm stable ✓ → **`decision_bar_pass=False`**.

**Soft class-2 win (Δ ≥ +0.03):** CORN class-2 **0.631** (Δ **−0.007**) → fail. CE class-2 **0.653** (Δ +0.015) — mild, under soft bar.

**Attribution (CORN − CE):** point ΔAUC **−0.0065**, boot CI **[−0.022, +0.008]** includes 0. Ordinal head does **not** beat same-backbone CE; neither beats C1 GBM.

**Interpretation:** Publishable small-n null. Rank-consistent CORN MLP and CE-MLP both sit ~2.5–3 pp below frozen C1 CatBoost on this matrix. The diagnosed class-2 ceiling is **not** lifted by an ordinal neural head under the locked protocol. Path A freeze / C1 secondary **unchanged**.

**Wording (null audit):** Prefer “CORN MLP + median-impute + weighted recipe does not raise C1 GBM.” CE independently trails C1 by ~2.5pp → most of the macro gap is **MLP+impute**, not CORN loss specificity. Primary bar null is **JUSTIFIED** (post-hoc critiquer 2026-07-16; `DECISIONS.md`).

---

## 2. Smoke (non-claim)

| Arm | Run | Test 4-AUC | Binary | Class-2 |
|---|---|---:|---:|---:|
| CORN | `corn_smoke_20260715_211638` | 0.7031 | 0.7568 | 0.6290 |
| CE | `ce_mlp_smoke_20260715_211638` | 0.7066 | 0.7615 | 0.6477 |

Smoke proved pipeline integrity (hash, freeze-before-test, proba contract, parent recompute). Full HPO did not reverse the underperformance pattern.

---

## 3. Full protocol detail

### 3.1 CORN (`corn_full_20260715_211707`)

| Item | Value |
|---|---|
| HPO | 20 Optuna trials; winner trial 6 |
| Params | hidden=128, dropout=0.5, bs=128, lr≈3.56e-4, wd≈9.41e-3, `weighted_corn` |
| Best epoch | 32 (patience 15, min_delta 1e-4) |
| Val | 4-AUC **0.7416**, AUPRC 0.498, binary 0.769 |
| Test raw | 4-AUC **0.7063**, AUPRC 0.444, binary **0.7628** |
| Per-class OVR test | c0 0.763 · c1 0.670 · c2 **0.631** · c3 0.761 |
| Ordinal (hard labels) | MAE 0.809, QWK 0.467 |
| Perm (val) | mean drop 0.0029, stable=True |
| Paired Δ vs C1 | macro **−0.0314** CI [−0.058, −0.005]; binary −0.068 CI [−0.110, −0.030] |

### 3.2 CE-MLP control (`ce_mlp_full_20260715_211707`)

| Item | Value |
|---|---|
| HPO | 20 trials; winner trial 0 |
| Params | hidden=64, dropout=0.3, bs=128, lr≈1.20e-3, wd≈2.61e-3 |
| Best epoch | 15 |
| Val | 4-AUC **0.7416**, AUPRC 0.494 |
| Test raw | 4-AUC **0.7128**, AUPRC 0.453, binary **0.7695** |
| Per-class OVR test | c0 0.770 · c1 0.671 · c2 **0.653** · c3 0.757 |
| Paired Δ vs C1 | macro **−0.0249** CI [−0.054, +0.004] (includes 0 on hi side); binary −0.061 |

### 3.3 CORN vs CE

Artifact: `artifacts/compare_corn_ce_20260715_211707.json`

| | Point | Boot 95% CI |
|---|---:|---|
| Δ macro-OVR (CORN−CE) | **−0.0065** | [−0.022, +0.008] |
| Δ binary | −0.0067 | [−0.027, +0.011] |

No ordinal advantage over CE on test.

---

## 4. Data / impute (locked)

- Matrix: watch GREEN (0 nulls) + hard onboarding + `cestl`/`paidscore`.
- Impute: train-median → z-score (B3 recipe). No mask layer.
- FH post-impute (train): `fh_dm2pt` pos 0.425→0.385; `fh_dm2sb` 0.276→0.249 (median fill 0).
- No cleaning-pipeline change.

---

## 5. What this does / does not claim

| Claim | Status |
|---|---|
| Frozen Path A W0 / C1 numbers changed | **No** |
| CORN raises deployable secondary over C1 | **No** (bar fail; CI entirely below 0) |
| Ordinal structure helps vs CE-MLP | **No** (Δ≈0, CI includes 0) |
| Class-2 soft win (≥+0.03) | **No** |
| Null is publishable | **Yes** — neural ordinal on C1 matrix underperforms C1 GBM at n=1824 |
| Unweighted CORN / mask impute / multi-seed | **Not run** — optional follow-ups only; do not re-open primary bar |

Optional CORAL / σ-stack **not run** (E6 gate: primary not competitive).

### 5.1 Post-hoc null audit (summary)

| Question | Verdict |
|---|---|
| Primary bar null justified? | **Yes** — Δ−0.031, CI entirely &lt;0; smoke≈full |
| Experiment ideal? | No blocker; caveats: impute×arch vs C1, weighted-CORN unablated, single seed |
| Could non-idealities flip primary bar? | **No** (~+4pp needed) |
| Soft class-2 more sensitive? | Plausibly (unweighted ablation) |

---

## 6. Artifact index

| Artifact | Path |
|---|---|
| CORN full | `artifacts/corn_full_20260715_211707/` |
| CE full | `artifacts/ce_mlp_full_20260715_211707/` |
| CORN−CE compare | `artifacts/compare_corn_ce_20260715_211707.json` |
| CORN smoke | `artifacts/corn_smoke_20260715_211638/` |
| CE smoke | `artifacts/ce_mlp_smoke_20260715_211638/` |

Each full run: `selected_model.json`, `models/selected.pt`, `metrics_test.json`, `features.json`, `scale_state.json`, `perm_importance.json`, `proba_test.npz`, `run_manifest.json`, `run.log`.

---

## 7. How to reproduce

```bash
export DRI_PRIME=1
.venv/bin/pip install -r training/path_a_raise_corn/requirements.txt
.venv/bin/python -m training.path_a_raise_corn --self-check
.venv/bin/python -m training.path_a_raise_corn --quick --arm both
.venv/bin/python -m training.path_a_raise_corn --arm both
```

Refuse-overwrite protects existing run ids; use a new `--run-id` to re-run.
