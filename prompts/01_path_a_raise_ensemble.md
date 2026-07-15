# Prompt — Path A raise: ensembling + multi-seed bagging

Get oriented with the project and its status (start at `AGENTS.md`).

Path A is **frozen** (see `training/path_a_blocks/REPORT_A_WRAP.md`, `DECISIONS.md`). The frozen
claim is unchanged:
- Watch-only headline **W0**: 4-AUC **0.666** / binary **0.689** (`training/path_a_watch/`).
- Secondary deployable **C1** (watch + onboarding + PAID mood): 4-AUC **0.738** / binary **0.831**
  (`training/path_a_blocks/`, run `mood_scores_20260714_014415`).

This is a **post-freeze raise attempt** under a **new** protocol, not a reopening of the frozen
claim. **Baseline to beat: C1 (0.738 / 0.831).** Do **not** re-run or alter W0/1A/1B/1C/wrap
artifacts. Do **not** claim Path A numbers changed unless this raise is actually run to completion
and the new numbers are labeled with new run ids — pre-raise / invalidated runs stay labeled as
such (see existing convention: `b1_grid_20260714` stays pre-fix-labelled).

**Scope of this raise (locked):** the current ladder discards the loser family at
`pick_family()` and uses a **single seed** per family. Standard small-n tabular practice is to
blend across families and across seeds. Specifically, within this package:
- **Multi-seed bagging** per family (e.g. 5–10 seeds; vary `seed` and/or val subsample) → mean
  predicted probabilities.
- **Cross-family ensembling**: arithmetic/geometric mean of LGBM and CatBoost probabilities; plus
  σ-stacking (logistic regression on out-of-fold val probabilities from each family).
- Optional: repeated paired-bootstrap on OOF across internal k-fold to get an
  effect-direction-consistency read alongside the single-test paired CI (does **not** move the
  point number; only tightens what you can claim and may rescue near-miss bars inside the test
  noise floor).

This is a **modelchoice** layer, not a new feature block and not a reopened ladder. Respect every
protocol lock in `training/path_a_blocks/config.yaml` and `Training.md` §6 (balanced weights,
val-select → immutable freeze → test once, paired person-bootstrap Δ CIs, SHAP guardrail,
pre-registered +0.01 AUC / CI-lo>0 / stable-perm decision bar).

**Isolation (required):** create a **new package** `training/path_a_raise_ensemble/` with its own
`artifacts/` root. Reuse `training.path_a_blocks.data_blocks` and `training.path_a_watch.*` by
import — do **not** edit them, do **not** write into their `artifacts/`. Keep all new run ids,
artifacts, and docs inside the new package. Path A frozen artifacts must remain byte-for-byte
untouched.

Keep all `.md` docs updated as we go: a new `PLAN_A_RAISE_ENSEMBLE.md` in the new package, a
`DECISIONS.md`, and a `REPORT` at the end with numbers and run ids. Mirror the existing doc style.

Explore cleaned data (`PROCESSED.md`, `data/processed/features/`, `data/processed/meta/`). Judge
whether the cleaning pipeline needs changes and whether the data we need is present:
- Ensembling does **not** require new features; it operates on the same C1 feature matrix. The
  relevant question is whether the existing feature parquets + `recommended_split` + frozen C1
  parent artifact are sufficient to reproduce C1's exact train/val/test rows and feature order
  for OOF generation (they should be — `parent_c1_run_id` + `selected_model.json` +
  `features.json` exist). Verify by reloading.
- If changes are needed: draft a plan, get a fresh critique (subagent `critiquer`,
  `opencode-go/glm-5.2:high`, fresh context — paste the plan in, child has no history), you
  decide what to address (drop false positives and nits that fight locked decisions; log
  disposition), then implement and verify. Present the plan before large edits.
- If the data is already sufficient: draft the raise plan, get a fresh critique (same critiquer
  rule), you decide what to address, then **present the plan** (do not implement yet unless I
  say so).

Use tools and subagents when intermediate work is better isolated (e.g. OOF-prediction
scaffolding, σ-stacker fitting); keep synthesis and false-positive filtering yourself. Prefer
`explorer.logic` for flow bugs, `web-researcher` for external methods. Do not run two writers on
the same files.

Keep the codebase working between edits. Run a smoke fit (2 trials, `--quick`-style) before any
full run. Re-read changed regions and run a relevant check after non-trivial edits; fix failures
before reporting completion — do not claim success on work with failing checks.