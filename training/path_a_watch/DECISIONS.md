# Path A watch-only GBM floor — decisions log

> Living research log for `training/path_a_watch/`.  
> Methodology authority remains `Training.md` / `PROCESSED.md`; this file records
> **implementation decisions and deviations** for this package only.

## 0. Scope (locked 2026-07-13)

- **Title:** Path A GBM floor (watch-only GREEN)
- **Models:** LightGBM + CatBoost on `watch_green` ⋈ `pool_masks`
- **Cohort:** wearable_core n=1824; train/val/test = 1277/270/277; train insulin = 80
- **User locks (session):**
  1. **Val-select** primary family by val macro-OVR AUC (not fixed-LGBM, not test-select)
  2. **50 Optuna trials** per family
  3. **Ordinal:** metrics + thin ordinal logistic only; full CORN neural deferred
- **Out:** survey blocks, Path B, SSL, dedicated binary HPO, RF/SVM

## 1. Protocol locks (post dual-critique revise)

| Lock | Choice | Why |
|---|---|---|
| Split language | Fixed `recommended_split` + val HPO/ES/cal — **not** nested CV | Honest held-out test; nested-CV CIs deferred to block ablations via person bootstrap |
| Primary metric | 4-class macro-OVR AUC | Training.md |
| Companion metric | macro AUPRC | Imbalance; always logged; AUC-tie → higher AUPRC within `auc_tie_eps=0.005` |
| Family selection | **Val** macro-OVR AUC among LGBM/CatBoost (+ baseline) | User choice; freeze before test |
| Final scorer | Train @ `best_iteration` (Policy A) | No silent train+val refit for headline |
| Early stopping | **multi_logloss** on val | Native/stable at n_val=270; HPO still ranks by AUC |
| Class weights | LGBM `balanced`; CatBoost `Balanced` | Locked before importance/HPO narrative |
| Binary metrics | Derived `1 − P(y=0)` from multiclass | No dedicated binary model this pass |
| Calibration default | Per-class **sigmoid** + renormalize on val | Stable at class3_val=49; isotonic secondary |
| Calibration claim | **Test** Brier + curves only | Val cal is post-HPO diagnostic |
| Feature sets | `full_green` primary; `physio_only` same-HP ablation | Wear-time confound (`hr_n` ρ≈−0.16) |
| Freeze | `selected_model.json` before any test write | Reviewer defense |

## 2. Dual-critique summary

Both critics: **revise**. Incorporated:

- Multiclass cal protocol + renormalize
- Freeze / test-once contract
- ES vs HPO alignment documented
- Coverage ablation
- Nested-CV rename
- Env: 3.14 wheels worked (no pyenv needed this machine)
- Ordinal evaluation without full CORN
- AUPRC as companion + tie-break (not opaque 0.5/0.5 composite)

## 3. Compute / GPU (this machine)

- CPU: Ryzen 5 4600H; GPU: Radeon RX 5600M 6 GB (`gfx1010`), plus iGPU `gfx90c`
- User note: export `DRI_PRIME=1` so processes bind to dGPU not iGPU
- **Probe results (2026-07-13):**
  - OpenCL: AMD APP platform, Device0 `gfx1010`, Device1 `gfx90c`
  - **LightGBM `device='gpu'`:** works (OpenCL)
  - **CatBoost `task_type='GPU'`:** fails — CUDA-only runtime (`CUDA error 35`)
- **Policy:** LGBM `lgbm_device: auto` (try GPU, fall back CPU); CatBoost **CPU only**
- At n≈1.3k × 30 features, Path A is cheap on CPU; GPU is optional acceleration for LGBM HPO only

## 4. Environment

- Project `.venv` Python **3.14.6**
- Installed (pinned in `requirements.txt` here): see that file
- Smoke: LGBM + CatBoost 20-tree multiclass fit OK on 3.14 wheels

## 5. Chronology

| When | Event |
|---|---|
| 2026-07-13 | Plan drafted from PROCESSED/Training; dual critiquer (default + opencode-go/glm-5.2) → revise |
| 2026-07-13 | User locks: val-select, 50 trials, ordinal logistic; research docs inside this dir |
| 2026-07-13 | Deps install OK on 3.14; LGBM OpenCL GPU OK; CatBoost GPU unavailable |
| 2026-07-13 | Package scaffold + implementation started |
| 2026-07-13 | Smoke OK (LGBM OpenCL GPU; freeze-before-test) |
| 2026-07-13 | Code critique #1 (default): revise — global select_best, trial try/except, appendix dual-family test, immutable freeze |
| 2026-07-13 | Code critique #2 (opencode-go/glm-5.2:xhigh): revise — freeze selected_device, ordinal convergence flag |
| 2026-07-13 | Fixes applied; launching full 50-trial run |
| 2026-07-13 | **Full run complete** `artifacts/full_20260713_221240/` (~2.8 min wall) |

## 8. Full-run results (`full_20260713_221240`)

| Item | Value |
|---|---|
| Val LGBM | macro-OVR AUC **0.6750**, AUPRC 0.4257 |
| Val CatBoost | macro-OVR AUC **0.6763**, AUPRC 0.4341 (Ordered, trial 45) |
| **Val-selected** | **CatBoost** |
| **Test claim (raw)** | 4-AUC **0.6662**, AUPRC **0.3916**, binary AUC **0.6889**, QWK 0.417 |
| Test per-class OVR AUC | 0: 0.689 · 1: 0.648 · 2: 0.565 · 3: **0.763** |
| physio_only Δ test AUC | **+0.0006** (coverage drop ≈ null) |
| Ordinal logistic test | AUC 0.649, QWK 0.329, converged=True (HessianInversionWarning) |
| SHAP top | hr_cv, rar_amplitude, rhr, mvpa_min_per_day, sri; **hr_n rank 6** |
| Cal Brier | raw 0.697 → sigmoid 0.728 (worse; keep raw for ranking claim) |
| Failures | LGBM 0, CatBoost 0 |

**Interpretation (research):** Watch-only floor is **below** the pre-registered honest band (4-class ~0.72–0.75, binary ~0.78–0.82). Class 2 (oral/non-insulin) is the weak link; insulin OVR is relatively strong. Coverage features are not carrying the model (physio ablation flat; hr_n mid-rank). This is a usable Path A floor for Path B deltas — not a paper-ready performance claim without further FE/blocks.

## 6. Open / deferred

- Full CORN neural ordinal (Training.md evaluation variant) — optional leftover
- Survey block ablation hierarchy — **done in `path_a_blocks/`** (1A/1B/1C + wrap freeze 2026-07-14);
  see `../path_a_blocks/REPORT_A_WRAP.md`. This package remains watch-only floor only.
- **Path B** — main next work (privileged CGM)
- Person-bootstrap CIs on floor test — optional polish (diagnostics already ran some of this)

## 7. How to run

```bash
# Prefer dGPU for OpenCL LightGBM
export DRI_PRIME=1
.venv/bin/python -m training.path_a_watch.run --run-id $(date +%Y%m%d_%H%M%S)
```

Artifacts: `training/path_a_watch/artifacts/<run_id>/`.
