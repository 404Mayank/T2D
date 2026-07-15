# Prompt — Path A raise: CORN/CORAL ordinal head

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

**Scope of this raise (locked):** the 4-class label is **ordinal**, and the diagnosed ceiling on
macro-OVR is **class-2 (oral/non-insulin injectable) OVR ~0.62–0.63** across every wrap stack —
the mid-severity sandwich that independent OVR softmax handles poorly. `Training.md` §2/§7 and
`FEATURES.md` already list **CORN ordinal regression** as required/lean-CORN and it was never run.

This raise implements a **tabular CORN head** (Conditional Ordinal Regression for Neural
networks; `coral-pytorch` library, output dim = num_classes − 1, `corn_loss` /
`corn_label_from_logits`). Fit on the **same C1 feature matrix** (watch + onboarding + PAID) so
the result is directly comparable to the frozen C1. Lean **CORN over CORAL** (CORN drops the
proportional-odds assumption we suspect fails at the pre-diabetes ↔ oral-med boundary; both are
rank-consistent — see `Training.md` §9). Optional sibling: CORAL MLP as the ablation companion.

Respect every protocol lock in `training/path_a_blocks/config.yaml` and `Training.md` §6
(val-select → immutable freeze → test once, paired person-bootstrap Δ CIs vs the **C1 parent**
artifact, SHAP guardrail on the tabular MLP, pre-registered +0.01 AUC / CI-lo>0 / stable-perm
decision bar). Use a **regularized tabular MLP** (heavy dropout, L2, early stopping on val
macro-OVR AUC + AUPRC; small-n at n=1824 means GBMs usually win — that's fine; a documented tie
or null is a publishable result, a class-2 lift ≥0.03 would be the win). Do **not** bring new
feature blocks; feature matrix must equal C1's exactly (assert `feature_hash` match against the
frozen C1 `features.json`). Consider heavy regularization and an OOF/σ-stacked blend with the
frozen C1 GBM as an optional late step, but the primary claim cell is CORN MLP alone.

**Isolation (required):** create a **new package** `training/path_a_raise_corn/` with its own
`artifacts/` root. Reuse `training.path_a_blocks.data_blocks` and `training.path_a_watch.*` by
import — do **not** edit them, do **not** write into their `artifacts/`. Keep all new run ids,
artifacts, and docs inside the new package. Path A frozen artifacts must remain byte-for-byte
untouched.

Keep all `.md` docs updated as we go: a new `PLAN_A_RAISE_CORN.md` in the new package, a
`DECISIONS.md`, and a `REPORT` at the end with numbers and run ids. Mirror the existing doc style.

Explore cleaned data (`PROCESSED.md`, `data/processed/features/`, `data/processed/meta/`).
Judge whether the cleaning pipeline needs changes and whether the data we need is present:
- The 4-class ordinal label already exists (`label` column; verified in
  `training/path_a_blocks/config.yaml`). CORN MLP on a tabular matrix needs **NaN handling /
  imputation** (GBMs handle NaN natively; the MLP does not). Judge whether the current
  `watch_green` + onboarding + mood parquets' missingness pattern needs a dedicated imputation
  policy (median/zero/kNN for numeric; document the rule), or whether a sentinel-aware masking
  input layer is cleaner given the pipeline's sentinel-to-missing convention. This is the one
  real data/feature-judgement in this raise — decide deliberately, don't default to a zero-fill.
- If changes are needed: draft a plan, get a fresh critique (subagent `critiquer`,
  `opencode-go/glm-5.2:high`, fresh context — paste the plan in, child has no history), you
  decide what to address (drop false positives and nits that fight locked decisions; log
  disposition), then implement and verify. Present the plan before large edits.
- If the data is already sufficient: draft the raise plan, get a fresh critique (same critiquer
  rule), you decide what to address, then **present the plan** (do not implement yet unless I
  say so).

Use tools and subagents when intermediate work is better isolated (e.g. CORN loss integration,
MLP training loop, σ-stacker); keep synthesis and false-positive filtering yourself. Prefer
`explorer.logic` for flow bugs, `web-researcher` for external ordinal-loss methods. Do not run
two writers on the same files.

Keep the codebase working between edits. Run a smoke fit (tiny epochs, single seed) before any
full run. Re-read changed regions and run a relevant check after non-trivial edits; fix failures
before reporting completion — do not claim success on work with failing checks.