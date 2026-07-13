# Path A — watch-only GBM floor

Research package: **LightGBM + CatBoost** on `data/processed/features/watch_green.parquet`
under fixed `recommended_split`.

| Doc | Role |
|---|---|
| `DECISIONS.md` | Living decision / protocol log for this package |
| `config.yaml` | Knobs (trials, ES, paths, HPO spaces) |
| `requirements.txt` | Pinned train deps |
| `artifacts/<run_id>/` | All outputs for a run (gitignored if desired) |

Methodology authority: repo root `Training.md` / `PROCESSED.md`.

## Run

```bash
export DRI_PRIME=1   # bind OpenCL LightGBM to dGPU (RX 5600M), not iGPU
cd /path/to/T2D
.venv/bin/python -m training.path_a_watch --run-id $(date +%Y%m%d_%H%M%S)

# smoke (2 trials, no shap/ordinal/ablation)
.venv/bin/python -m training.path_a_watch --quick --run-id smoke
```

## Protocol (summary)

1. Load `watch_green` ⋈ `pool_masks` (n=1824 assert)
2. Optuna 50 trials LGBM + 50 CatBoost on val multi_logloss ES; rank by val macro-OVR AUC
3. **Val-select** family → write `selected_model.json` (**before** test)
4. Calibrate on val (sigmoid default); score test once
5. `physio_only` ablation (drop coverage counts, same HPs)
6. Ordinal logistic baseline; SHAP + permutation on val

CatBoost GPU is CUDA-only → **CPU**. LightGBM may use OpenCL GPU (`lgbm_device: auto`).

## Results (locked floor)

Run `full_20260713_221240` (local `artifacts/`, gitignored):

| Metric | Test |
|---|---:|
| 4-class macro-OVR AUC | **0.666** |
| Binary AUC | **0.689** |
| Family | CatBoost Ordered |

Block ladder / onboarding: see `../path_a_blocks/REPORT.md`.
