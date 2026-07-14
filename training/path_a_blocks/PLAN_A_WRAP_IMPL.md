# Implementation plan — Phase A wrap (`run_wrap`)

> Status: **executed 2026-07-14** (all 11 experiments + `REPORT_A_WRAP.md` + Path A freeze).  
> Historical plan (post code-plan critique). Source of truth for results: `REPORT_A_WRAP.md`.  
> Follows locked `PLAN_A_WRAP.md`. Critique verdict at plan time: **REVISE** → O1/O2 + O3–O8 incorporated before coding.

---

## 0. Goal / success

Path A is frozen when:

1. `artifacts/wrap_feature_ranks.json` exists (dual SHAP+perm ranks; `minimal_S` top-12; `minimal_M` top-18; `cestl` excluded).
2. All 11 experiments have artifact dirs with freeze → metrics_test → REPORT.md.
3. `REPORT_A_WRAP.md` consolidates tables + limitations + frozen recommendation.
4. `DECISIONS.md`, `PATH_AHEAD.md`, `REPORT.md` updated; next = Path B.

**Out of scope:** Path B, SSL, diet, re-FE GREEN, claiming 1B core / complications as primary bar-pass.

---

## 1. Verified inputs (do not invent)

| Ref | Path / value |
|---|---|
| W0 | `path_a_watch/artifacts/full_20260713_221240` · 4-AUC **0.6662** · bin **0.6889** |
| A1 | `path_a_blocks/artifacts/onboarding_20260713_224744` · **0.6987** / **0.7492** |
| C1 | `path_a_blocks/artifacts/mood_scores_20260714_014415` · **0.7378** / **0.8309** · CatBoost |
| C1 SHAP/perm | `.../shap/catboost_1c_scores_{shap,perm}_importance.csv` (47 feats each) |
| Cohort | n=1824; splits 1277/270/277; seed 42; no site feature |

**Precomputed dual-rank (runtime CSV is authority; expected):**

- **minimal_S (12):** `paidscore, whr_vsorres, fh_dm2sb, fh_dm2pt, rhr, waist_vsorres, hr_mean, bmi_vsorres, weight_vsorres, sri, stress_sd, hr_min`
- **minimal_M (18):** S + `hip_vsorres, hr_nocturnal_dip, hr_n, sleep_short_frac, age, pulse_vsorres_2`
- Note: `hr_cv` / `rar_amplitude` excluded by dual-rank (perm-negative / low dual).

---

## 2. Files to add / touch

| File | Action |
|---|---|
| `training/path_a_blocks/build_minimal_ranks.py` | **New** — write `wrap_feature_ranks.json` |
| `training/path_a_blocks/run_wrap.py` | **New** — CLI for all 11 exps |
| `training/path_a_watch/models.py` | **Extend** — binary LGBM/CatBoost builders (or local helpers in run_wrap if cleaner) |
| `training/path_a_watch/hpo.py` | **Extend** — `tune_lightgbm_binary`, `tune_catboost_binary`, `pick_family_binary` |
| `training/path_a_watch/metrics.py` | **Extend** — `binary_report(y_bin, score_or_proba)` |
| `training/path_a_blocks/diagnostics.py` | **Extend** — `bootstrap_ci_binary`, `paired_delta_bootstrap_binary` (O2) |
| `training/path_a_watch/explain.py` | **Extend** — optional `permutation_on_val_binary` (O2) |
| `training/path_a_blocks/data_blocks.py` | **Extend** — loaders/helpers for wrap feature matrices (mood±comorb subsets) |
| `training/path_a_blocks/config.yaml` | **Add** — parent_c1 refs, wrap paths, ranks path |
| Docs | `REPORT_A_WRAP.md`, update `DECISIONS.md`, `PATH_AHEAD.md`, `REPORT.md`, tick `PLAN_A_WRAP.md` |

Prefer **minimal invasive** binary HPO: add parallel functions rather than branching multiclass paths with flags that risk breaking 1A–1C.

---

## 3. Step 0 — `build_minimal_ranks.py`

```text
Inputs (config or defaults):
  shap_csv = C1 .../catboost_1c_scores_shap_importance.csv
  perm_csv = C1 .../catboost_1c_scores_perm_importance.csv
  exclude = ["cestl"]
  n_S=12, n_M=18

Algorithm:
  1. Load both as Series; assert same feature index set.
  2. Drop exclude.
  3. rank_shap = rank(desc, method=average); rank_perm same.
  4. combined_score = rank_shap + rank_perm (lower better).
  5. Sort by (combined, -shap, -perm).
  6. minimal_S = head(12); minimal_M = head(18).
  7. Write artifacts/wrap_feature_ranks.json:
     {
       source_run_id, shap_csv, perm_csv, exclude,
       ranks: [{feature, shap, perm, rank_shap, rank_perm, combined}, ...],
       minimal_S, minimal_M, n_candidates, built_at
     }
  8. Refuse overwrite unless --force.
```

CLI: `.venv/bin/python -m training.path_a_blocks.build_minimal_ranks [--force]`

`run_wrap` for E1a/E1b/E4c **requires** this file; if missing, call builder or hard-fail with clear message.

---

## 4. Feature matrix construction (per exp)

Reuse loaders; **subset columns** after full merge so null/coverage asserts stay valid.

| Exp | Matrix construction |
|---|---|
| **E2a paid_only** | `load_watch_onboarding_mood(..., mood_cols=["paidscore"])` → features = watch + onboard + [paidscore] |
| **E2c ces_only** | same with `mood_cols=["cestl"]` |
| **E1a minimal_S** | load full C1 matrix (scores), then `feature_cols = ranks["minimal_S"]` (order preserved from ranks file) |
| **E1b minimal_M** | same with `minimal_M` |
| **E5 watch_mood** | load mood paidscore only; `feature_cols = watch_cols + ["paidscore"]` (drop onboard) |
| **E3a severity_addons** | load C1 scores + merge comorbidity `mhoccur_rnl`, `mhoccur_circ` only (**no** count feature) |
| **E3b clinical_upper** | C1 + `mhoccur_hbp, mhoccur_clsh, mhoccur_rnl, mhoccur_circ` (**no** count) |
| **E4a bin_watch** | watch GREEN only; y = (label>0) |
| **E4b bin_c1** | full C1 matrix; binary y |
| **E4c bin_min_s** | minimal_S cols; binary y |
| **E4d bin_severity** | E3a cols; binary y |

### Loader notes

- Prefer a small helper `load_c1_plus_comorb(binaries: list[str])` that:
  1. `load_watch_onboarding_mood` with scores `[cestl, paidscore]`
  2. left-merge selected comorb binaries (pid coverage assert like 1B)
  3. **does not** engineer `comorb_count_*` (plan O7)
- Assert parent feature list for matrices that extend C1: C1 feature_cols ⊆ new feature_cols for E3*.
- For E1*: every minimal feature must exist in C1 feature set; no silent drop.
- Tags for SHAP guardrail: watch / onboarding / mood / comorbidity as applicable.

---

## 5. Multiclass experiment protocol (E1*, E2*, E3*, E5)

Clone `run_1c.py` structure:

1. Assert parents as needed:
   - Always assert floor + A1 artifact numbers (config refs).
   - For comparisons vs C1 (E1 retention, E2, E5, E3): assert C1 metrics from artifact + optional recompute proba bit-match when parent model available and feature subset allows.
2. Build matrix / splits (`make_block_splits`).
3. HPO: `tune_lightgbm` + `tune_catboost` (50 trials; `--quick` → 2).
4. `pick_family` (AUC then AUPRC within eps=0.005).
5. **Freeze** `selected_model.json` **before** writing any test metrics.
6. Refuse overwrite if freeze or metrics_test exists.
7. Test: `full_report`; bootstrap CI; paired Δ vs relevant parent(s):
   - E2/E5: Δ vs A1 and vs C1 (report both; no decision_bar for wrap ablations unless useful).
   - E1: Δ vs C1 + **retention rule** evaluation.
   - E3: Δ vs C1 (severity narrative; not claim primary bar).
8. SHAP + perm (skip on `--quick` / `--skip-shap`).
9. Write metrics_val/test, models, REPORT.md, run_manifest, run.log.

### Parent C1 proba for subset matrices (critique O1 — blocker)

Most wrap matrices are **not** supersets of C1's 47 cols (E1/E2/E5). Do **not** call
`predict_proba(c1_model, splits.X_test[parent_feat])` on the experiment matrix.

Add `load_parent_c1_proba(repo, cfg, test_pids) -> (proba_4class, meta)`:

1. `assert_parent_c1(repo, cfg)` → metrics + `feature_cols` + `model_path` (must exist).
2. Independently `load_watch_onboarding_mood(mood_cols=[cestl, paidscore])`.
3. Build C1 feature frame; **`reindex(columns=c1_freeze["feature_cols"])`** (O3 — CatBoost is positional).
4. Align rows to experiment `test_pids` via `person_id` index (do not assume df order).
5. `predict_proba`; bit-match macro OVR AUC vs artifact with tol **1e-6** (O6).
6. Return proba for `paired_delta_bootstrap`.

E3a/E3b may short-circuit: C1 cols ⊆ current matrix → `X_test.reindex(columns=c1_feat)`.

### `assert_parent_c1` (O5)

Clone `assert_parent_1a`: read C1 `metrics_test.json` + `selected_model.json` + require
`models/selected.joblib` exists; assert config refs within 1e-9 on stored JSON values;
return feature_cols, model_path, metrics.

### Freeze guard timing

Check `selected_model.json` / `metrics_test.json` existence **before HPO**, not after
(so a re-run with same `--run-id` fails immediately).

### Retention rule (E1a/E1b only, on test)

```text
retain = (
  (auc_min - auc_C1) >= -0.01
  and (bin_min - bin_C1) >= -0.015
  and not (paired_boot ΔAUC CI entirely < 0)   # i.e. fail if hi < 0
)
```

Paper pick after both E1 runs: S if retains → else M if retains → else C1.
Write `artifacts/wrap_paper_pick.json` after E1a+E1b (and refresh after E4s) (O8).

Also report whether bootstrap CI includes 0 (informational; plan O3).

---

## 6. Binary HPO protocol (E4*)

### Labels / metrics

- `y_bin = (label > 0).astype(int)` on train/val/test (same person splits).
- Score = `P(y=1)` from binary model.
- Metrics: binary AUC, AUPRC, Brier; base rates train/val/test.
- Val rank: **binary AUC**; tie: **binary AUPRC** within same `auc_tie_eps`.

### Model builders (new)

| Family | Settings |
|---|---|
| LGBM | `objective="binary"`, `class_weight="balanced"` **literal** (O4 — do NOT read multiclass `cfg['class_weights']['lightgbm']` dict), early stop binary logloss |
| CatBoost | `loss_function="Logloss"`, `auto_class_weights="Balanced"` literal, Ordered→Plain fallback |

Reuse same HPO hyperparameter spaces as multiclass (learning_rate, depth/leaves, reg, etc.).

### Binary bootstrap / report (critique O2 — blocker)

Existing `bootstrap_ci` / `paired_delta_bootstrap` / `full_report` assert 4-class proba.
Binary path must use:

- `binary_report(y_true_multiclass_or_bin, score_1d)` → AUC, AUPRC, Brier, base rate
- `bootstrap_ci_binary(y_bin, score_1d, ...)`
- `paired_delta_bootstrap_binary(y_bin, score_new, score_parent, ...)`
- Optional: `permutation_on_val_binary` scoring binary AUC of `predict_proba[:,1]`

Score convention: `P(y=1)` from binary model; parent multiclass-derived score = `1 - P0`.
Skip single-class bootstrap resamples; report `n_boot_ok`.

### Selection

- `tune_lightgbm_binary` / `tune_catboost_binary` mirror multiclass packs but store `val_binary_auc`, `val_binary_auprc`.
- `pick_family_binary` uses `select_best(..., auc_key="binary_auc", auprc_key="binary_auprc")` or equivalent pack fields.
- Freeze before test.

### Paper binary-primary gate

Compare each E4 test AUC to multiclass-derived `1−P0` from the matching multiclass stack:

| Binary exp | Derived reference |
|---|---|
| E4a | W0 binary 0.6889 |
| E4b | C1 binary 0.8309 |
| E4c | E1a multiclass-derived binary (from E1a artifact after it runs) |
| E4d | E3a multiclass-derived binary |

Prefer binary-primary in paper table only if `test_bin_auc - derived >= +0.01`.

**Run order constraint:** E4c after E1a; E4d after E3a (already in plan order).

---

## 7. CLI design

```bash
export DRI_PRIME=1
.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --exp paid_only [--run-id ...] [--quick] [--n-trials N] [--skip-shap]
```

`--exp` choices (aliases match plan IDs):

```text
paid_only | ces_only | minimal_s | minimal_m | watch_mood
| severity | clinical_upper
| bin_watch | bin_c1 | bin_min_s | bin_severity
```

Default run_id pattern: `wrap_{exp}_%Y%m%d_%H%M%S` (local time ok; match prior style).

Optional: `--all` sequential runner for the fixed order (nice-to-have; can be shell loop).

---

## 8. Config.yaml additions

```yaml
paths:
  parent_c1_run_id: mood_scores_20260714_014415
  wrap_ranks: training/path_a_blocks/artifacts/wrap_feature_ranks.json

parent_c1_reference:
  test_macro_ovr_auc: 0.7377859791966677
  test_binary_auc: 0.8308943089430895
  test_macro_auprc: 0.4687470484302076
  family: catboost
```

No change to onboarding_keep / mood_scores lists.

---

## 9. Run order & wall time

```text
0 build ranks
1 E2a paid_only
2 E2c ces_only
3 E1a minimal_s
4 E1b minimal_m
5 E5  watch_mood
6 E3a severity
7 E3b clinical_upper
8 E4a bin_watch
9 E4b bin_c1
10 E4c bin_min_s   # needs E1a
11 E4d bin_severity # needs E3a
```

~11 × 10–15 min ≈ 2–3 h sequential. Smoke first: one multiclass (`paid_only --quick`) + one binary (`bin_watch --quick`).

Environment: `.venv`, `DRI_PRIME=1`, CatBoost CPU, LGBM OpenCL auto.

---

## 10. Artifacts per run

```text
artifacts/wrap_<exp>_<ts>/
  selected_model.json      # freeze first
  selected_model_post.json
  best_params_{lgbm,catboost}.json
  features.json
  metrics_val.json
  metrics_test.json
  models/{selected,lgbm,catboost}.joblib
  shap/…                   # multiclass primarily
  explain.json
  REPORT.md
  run_manifest.json
  run.log
```

Shared: `artifacts/wrap_feature_ranks.json`.

---

## 11. Reporting & freeze

### `REPORT_A_WRAP.md` sections

1. Ladder + wrap table (W0→A1→C1 + all wrap exps)
2. Minimal S/M vs C1 + retention + paper pick
3. PAID-only vs CES-only vs C1
4. watch+mood vs C1 (onboarding value)
5. Severity E3a vs clinical upper E3b
6. Binary-primary vs derived
7. SHAP notes (if run)
8. **Mandatory limitations:** val-importance selection bias; re-HPO confounds; complications = severity proxies
9. **Frozen Path A recommendation** (pre-registered rule applied to results)
10. Next = Path B handoff note

### Doc updates

- `DECISIONS.md`: wrap chronology + paper pick + freeze
- `PATH_AHEAD.md`: Phase A complete pointer → Path B
- `REPORT.md`: short wrap summary + link to REPORT_A_WRAP
- `PLAN_A_WRAP.md`: check implement/run boxes

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Binary HPO breaks multiclass | Separate functions; no flag spaghetti in existing tune_* |
| Minimal ranks file missing | Hard fail with builder instruction |
| Parent recompute bit-match fails on subset | Only recompute when full parent feature cols present in X_test |
| CatBoost long wall time | Sequential; smoke first; no unsafe GPU for CatBoost |
| Overwrite artifacts | Refuse if freeze/metrics exist |
| E4c/E4d before parents | Document order; assert E1a/E3a metrics files exist when needed for derived compare |
| Class imbalance binary | Keep balanced weights; report base rates (O10) |

---

## 13. Implementation steps (coding order)

1. `build_minimal_ranks.py` + run it → print S/M lists; commit ranks JSON (or leave under artifacts gitignored — still write file).
2. Config parent_c1 + paths.
3. `data_blocks.py` helper for C1+comorb subset (no count).
4. Binary model + HPO + metrics helpers.
5. `run_wrap.py` multiclass path (E2a first).
6. Smoke E2a `--quick`.
7. Binary path + smoke E4a `--quick`.
8. Wire E1 retention, E3, E5, remaining E4.
9. Code critique → fix criticals.
10. Full 50-trial batch in plan order.
11. REPORT_A_WRAP + doc freeze.

---

## 14. Test / verification checklist

- [ ] Ranks: 12/18 lists match recompute; cestl absent
- [ ] Smoke multiclass completes freeze-before-test
- [ ] Smoke binary ranks on val binary AUC
- [ ] Parent assert fails loudly if artifact moved
- [ ] E3 has no count column in features.json
- [ ] E5 has zero onboarding cols
- [ ] E1 retention fields present in metrics_test
- [ ] All 11 artifacts + consolidated report

---

## 15. Open questions (defaults locked — only flag if critique disagrees)

1. Put binary builders in `models.py`/`hpo.py` vs private in `run_wrap.py` → **prefer shared modules** (Path B may reuse binary).
2. SHAP on all multiclass wraps → **yes** full runs; skip binary SHAP unless cheap (optional perm only).
3. Decision-bar formal c1∧c2∧c3 for wrap ablations → **no** (only retention rule for E1; narrative Δ elsewhere).
