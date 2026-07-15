# Path A raise CORN — decisions log

> Package: `training/path_a_raise_corn/`.  
> Plan: `PLAN_A_RAISE_CORN.md`.  
> Parent: frozen C1 `mood_scores_20260714_014415` (4-AUC **0.7378** / binary **0.8309**).  
> Does **not** reopen Path A freeze (`REPORT_A_WRAP.md`).

## Locks (plan stage)

- Isolation: all artifacts under this package; no writes into `path_a_blocks` / `path_a_watch`.
- Feature matrix: exact C1 47 cols, `feature_hash=d63ec5713ada37bf`.
- Impute: train-median + train z-score (B3 recipe); no mask layer v1.
- Primary: CORN MLP; **required control:** CE-MLP same backbone.
- Imbalance: weighted CORN loss (no WeightedRandomSampler).
- Bar: Δ macro-OVR AUC > +0.01 vs C1 ∧ paired boot lo > 0 ∧ perm not noise — **CORN only**.
- Soft note: class-2 OVR Δ ≥ +0.03 publishable without replacing C1.
- Null is publishable.

## Chronology

| When | Event |
|---|---|
| 2026-07-15 | Package created; plan drafted |
| 2026-07-15 | Critiquer (`opencode-go/glm-5.2:high`) → **revise**; disposition in plan §10 |
| 2026-07-15 | **Implemented** package code |
| 2026-07-16 | Smoke `corn_smoke_20260715_211638` / `ce_mlp_smoke_20260715_211638` OK |
| 2026-07-16 | **Full concluded null:** `corn_full_20260715_211707` bar **FAIL** (test 4-AUC 0.706 vs C1 0.738, Δ−0.031, boot CI entirely &lt;0); CE control 0.713; CORN−CE Δ−0.0065 ~0. See `REPORT.md`. |

## Claim lock (post-full)

- **decision_bar_pass=False** for CORN vs C1.
- Soft class-2 win: False (CORN c2 0.631, Δ−0.007).
- Attribution: ordinal head does not beat CE-MLP; both lose to C1 GBM.
- C1 / Path A freeze **unchanged**. No CORAL/σ-stack (primary not competitive).
- Smoke runs are non-claim.

## Post-hoc null audit (critiquer, 2026-07-16)

Fresh `critiquer` / `opencode-go/glm-5.2:high` on completed raise.

| Question | Verdict |
|---|---|
| Is the **primary null justified**? | **JUSTIFIED** — Δ−0.031, boot CI entirely &lt;0; smoke≈full; margin too wide to flip |
| Was the experiment ideal? | **No** — see below; no blocker |

**Non-idealities (kept as caveats, not re-openers):**
1. **High:** C1 GBM native-NaN vs MLP median-impute — vs-C1 confounds arch+impute; CE control shows most gap is MLP/impute not CORN loss.
2. **High/plausible:** weighted CORN may over-mass severe classes (class-3 in all tasks); no unweighted ablation — could move soft class-2, not macro bar.
3. **High/plausible:** single seed — CORN≈CE direction sensitive; bar not.
4. **Med:** val→test ~3pp gap; CE HPO winner=trial 0 (plateau); weak perm “stable” rule (mean drop 0.0029).

**Wording lock:** prefer “CORN MLP + median-impute + weighted recipe does not raise C1 GBM” over bare “ordinal does not raise C1.”

**Follow-ups that could change science (optional, need new go):** unweighted CORN ablation (soft c2); impute sensitivity for attribution; multi-seed for CORN−CE sign. **Not recommended:** more HPO alone; multi-seed only to re-litigate primary bar.

## Implementation notes

- `coral-pytorch==1.4.0` installed; hand-port fallback in `losses_proba.py`.
- Feature hash `d63ec5713ada37bf`; weighted CORN loss (no sampler); train-median+z impute.
- Commands: `--self-check` · `--quick --arm both` · `--arm both`.
