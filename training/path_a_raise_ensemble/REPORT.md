# Path A raise — multi-seed bagging + cross-family ensemble

**Status:** **concluded null** (valid protocol; bar fail).  
**Does not reopen** frozen Path A claims (W0 / C1).  
**Parent:** C1 `mood_scores_20260714_014415` — test 4-AUC **0.7378** / binary **0.8309**.  
**Package:** `training/path_a_raise_ensemble/` (isolated artifacts).  
**Plan:** `PLAN_A_RAISE_ENSEMBLE.md` (post-critique revised).

---

## 1. Claim (pre-registered)

Dual primary vs C1:

| Primary | Arm | Question |
|---|---|---|
| **A** | `Bag_cat` | Multi-seed CatBoost bag raise C1? |
| **B** | `E_arith` | Arithmetic mean of family bags add on top? |

Bar: **c1** Δ4-AUC > +0.01 ∧ **c2** paired person-bootstrap CI lo > 0 ∧ arm-specific **c3**.  
Siblings (`E_geom`, `E_stack`, `Bag_lgbm`) = exploratory only.

**Verdict:** **no primary pass** on S=5 or mandatory S=10. Single-seed C1 CatBoost remains best honest tabular secondary.

---

## 2. Runs

| Run id | Seeds | claim_eligible | Role |
|---|---|---|---|
| `ens_smoke_20260715_impl` | 42–43 | false | path check (non-claim) |
| **`ens_full_20260715_impl`** | 42–46 (S=5) | **true** | **primary claim** |
| **`ens_s10_20260715_impl`** | 42–51 (S=10) | **true** | **mandatory near-bar follow-up** |

Protocol locks held: frozen C1 params (no re-HPO); LGBM `device=gpu`; parent bit-match; freeze-before-test; bootstrap seeds 42 / paired 53; stacker OOF bag-mean ES-on-fold.

Implementation notes (non-claim): sklearn 1.9 dropped `LogisticRegression(multi_class=…)`; stacker uses default multinomial path.

---

## 3. Primary results (S=5 claim)

Run: `ens_full_20260715_impl`

| Primary | Arm | Test 4-AUC | Binary | ΔAUC vs C1 | Δbin | c1 | c2 | c3 | bar |
|---|---|---:|---:|---:|---:|---|---|---|---|
| A | Bag_cat | **0.7392** | 0.8278 | **+0.0014** | −0.0030 | F | F | F | **F** |
| B | E_arith | **0.7439** | 0.8241 | **+0.0062** | −0.0068 | F | F | F | **F** |
| — | C1 parent | **0.7378** | **0.8309** | 0 | 0 | — | — | — | baseline |

Paired ΔAUC 95% CI vs C1 (point inside noise):

| Arm | ΔAUC | CI lo | CI hi |
|---|---:|---:|---:|
| Bag_cat | +0.0014 | −0.0078 | +0.0106 |
| E_arith | +0.0062 | −0.0092 | +0.0199 |

**c3 detail**

- **A:** only 2/5 seeds ≥ val floor (need ≥4); mean seed val < S_cat val − 0.002 → fail.  
- **B:** best bag = Bag_cat; point Δ(E_arith − Bag_cat) = +0.0047 < 0.005 margin; paired CI vs best bag includes 0 → fail.

**S_cat seed-42 refit:** test 4-AUC **0.7378**, ΔB0 = **0.0000** (reproduces frozen C1).

---

## 4. Sibling arms (S=5, exploratory)

| Arm | Test 4-AUC | Binary | ΔAUC vs C1 | note |
|---|---:|---:|---:|---|
| Bag_lgbm | 0.7380 | 0.8110 | +0.0002 | weaker family |
| E_geom | 0.7437 | 0.8243 | +0.0059 | ≈ E_arith |
| E_stack | 0.7386 | 0.8169 | +0.0008 | C=0.1; non-degenerate; no raise |
| S_lgbm | 0.7407 | 0.8080 | +0.0029 | single seed; binary down |

None approach +0.01; all paired CIs vs C1 include 0 (where computed for primaries/siblings with Δ).

---

## 5. S=10 mandatory follow-up

Run: `ens_s10_20260715_impl` (triggered: S=5 Δ ∈ (0, +0.01] on both primaries).

| Primary | Arm | Test 4-AUC | ΔAUC vs C1 | c1 | c2 | c3 | bar |
|---|---|---:|---:|---|---|---|---|
| A | Bag_cat | 0.7384 | **+0.0006** | F | F | F | **F** |
| B | E_arith | 0.7410 | **+0.0032** | F | F | F | **F** |

S=10 did **not** help vs S=5 (E_arith point Δ fell +0.0062 → +0.0032). Further seed averaging is not a path to the bar.

---

## 6. Interpretation

1. **Seed bagging of the winning family does not raise C1.** Bag_cat ≈ single-seed C1 (Δ ~0–0.001). Extra seeds are near-redundant once HPO + ES already fit seed 42.  
2. **Cross-family blend gives a small point lift (~+0.3–0.6 pp)** that sits **inside test noise** (CI includes 0; short of +0.01). Binary slightly **worse** than C1.  
3. **Stacking does not beat simple blends** under this protocol.  
4. Matches the plan’s honest prior (wrap nulls already sat inside ±0.01 of C1): modelchoice alone is not enough to clear the pre-registered bar on n_test=277.

**Deployable secondary remains C1.** Do not replace C1 with Bag_cat / E_arith / E_stack in paper tables.

---

## 7. What this does *not* change

- Frozen W0 **0.666 / 0.689**  
- Frozen C1 **0.738 / 0.831**  
- Path A wrap / sensitivity / CORN dispositions  
- Path B ladder status  

Optional later (new PLAN only): RHPO-then-bag, val-subsample diversity, weighted blend — **not** claimed here.

---

## 8. Artifacts

```
training/path_a_raise_ensemble/artifacts/
  ens_smoke_20260715_impl/     # non-claim
  ens_full_20260715_impl/      # S=5 claim
  ens_s10_20260715_impl/       # S=10 follow-up
```

Each claim run: `metrics_test.json`, `metrics_val.json`, `selected_ensemble.json`, `parent_assert.json`, `bag_*.json`, `stacker_meta.json`, `proba/`, `REPORT.md`, `run.log`.
