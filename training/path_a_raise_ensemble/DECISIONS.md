# Path A raise ensemble — decisions log

> Package: `training/path_a_raise_ensemble/`.  
> Plan: `PLAN_A_RAISE_ENSEMBLE.md` (post-critique revised).  
> Does **not** reopen frozen Path A claim numbers.

## Locks

- Isolation: own `artifacts/`; import-only from `path_a_blocks` / `path_a_watch`
- Parent = C1 `mood_scores_20260714_014415` (4-AUC 0.7378 / binary 0.8309)
- No re-HPO primary; frozen C1 params; LGBM **device=gpu** asserted
- Seeds `[42,43,44,45,46]`; dual primary **A=`Bag_cat`**, **B=`E_arith`**
- Stacker OOF = bag-mean ES-on-fold; siblings exploratory if only they clear
- Bootstrap seeds 42 / paired 53; `n_boot_ok ≥ 950` (smoke relaxed)
- Decision bar vs C1: ΔAUC>+0.01 ∧ CI lo>0 ∧ arm-specific c3
- S=10 mandatory if near-bar on S=5 (Δ∈(0,+0.01] or c1∧¬c2)

## Chronology

| When | Event |
|---|---|
| 2026-07-15 | Plan drafted; data readiness verified (C1 bit-match) |
| 2026-07-15 | Critique (glm-5.2): revise — dual primary, c3, stacker OOF, GPU pin |
| 2026-07-15 | Critique dispositions accepted; plan revised |
| 2026-07-15 | Package implemented |
| 2026-07-16 | Smoke `ens_smoke_20260715_impl` OK (non-claim) |
| 2026-07-16 | Full S=5 `ens_full_20260715_impl` — **both primaries bar-fail** |
| 2026-07-16 | S=10 `ens_s10_20260715_impl` (mandatory) — **both primaries bar-fail** |
| 2026-07-16 | **Raise concluded null**; C1 unchanged (`REPORT.md`) |

## Runs

| Run | Seeds | claim | A bar | B bar | Best ΔAUC (E_arith) |
|---|---|---|---|---|---:|
| `ens_smoke_20260715_impl` | 2 | no | — | — | +0.0038 (non-claim) |
| `ens_full_20260715_impl` | 5 | **yes** | **F** | **F** | **+0.0062** |
| `ens_s10_20260715_impl` | 10 | **yes** | **F** | **F** | **+0.0032** |

## Disposition

- **Ensemble/bag null** under locked protocol.  
- Secondary deployable remains **C1**.  
- No soft-promote of E_arith point lift (under +0.01; CI includes 0; binary slightly down).  
- Close raise; further modelchoice needs a **new PLAN_*** (not silent re-grid).
