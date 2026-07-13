# Compute, Storage & Access — Project Environment

> Stable reference for any agent/session working on this project, so the access
> info doesn't need to be re-typed. Verify GPU prices/quotas before long runs —
> these are as-provided by the user (2026-07). Related: `DATA_STRUCTURE.md`
> (dataset layout), `FEATURES.md` §9 / `Training.md` §7 (what to run where),
> `DATA_AUDIT.md` (cleaning is the current CPU-bound bottleneck).

## Local machine (dev / prototyping only)

- **CPU:** AMD Ryzen 5 4600H. **GPU:** AMD Radeon RX 5600M, **6 GB VRAM** (mobile, not CUDA → useless for mainstream ML stacks). **RAM:** 16 GB. **Network:** ~40 Mbps.
- **Filesystem:** btrfs root. User is fine putting up to ~5 GiB on the normal drive (no subvol needed); in practice the relevant pull is ~780 MiB so this is moot.
- **Use for:** mini-variant prototyping, cleaning-logic dev, small parquet reshaping. **Not** for full-scale training.

## Dataset location (canonical, on Google Drive)

- **rclone remote:** `gdrive_zyrus:` (configured on this machine).
- **Canonical path:** `gdrive_zyrus:AI_READI/{mini|full}/AI_READI/` — identical internal layout for both variants; swap only `mini`/`full`.
  - `mini` = 305 MiB (100 participants) · `full` = 6.36 GiB (2,280 participants).
  - Full breakdown: environment 5.0 GiB · garmin 741 MiB · ecg 568 MiB · dexcom 32 MiB · clinical 4.8 MiB · metadata 5.9 MiB.
  - **Relevant subset** (garmin + dexcom + clinical + metadata; env & ecg dropped) ≈ **780–784 MiB**.
- **Local copy on this machine:** `data/full/AI_READI/` (relevant subset present; use this for
  cleaning/FE/audit). Full tree including environment/ECG still lives on Drive if needed.
- **⚠ Remote-name mismatch:** `convert_pipeline.py` defaults `GDRIVE_REMOTE` to `gdrive:…` but the actual configured remote is `gdrive_zyrus:`. Set `AI_READI_GDRIVE=gdrive_zyrus:AI_READI/{full|mini}/AI_READI` (or rename the remote) before archiving.
- **Rule:** copy canonical to VM local disk once before training; never train over a Drive mount (stalls on random access).

## Lightning.ai — interruptible GPU (primary training target)

SSH + VSCode/notebook interfaces; even the free 4-core CPU VM has 400 GB+ storage. "Time on $15" column omitted for brevity.

| GPU | 1× $/hr | 2× | 4× | 8× |
|---|---|---|---|---|
| T4 | 0.51 | — | 1.11 | — |
| L4 | 1.22 | 1.78 | 3.23 | 6.41 |
| L40S | 1.94 | — | 6.64 | 10.49 |
| RTX 6000 | 2.06 | 3.74 | 6.73 | 10.55 |
| A100 | 3.32 | 6.60 | 13.15 | 26.25 |
| H100 | 3.82 | 7.64 | 15.28 | 30.56 |
| H200 | — | — | — | 27.11 (1× unavailable) |
| B200 | 9.86 | — | — | 41.57 |
| TPU | 1.46 | — | 5.74 | 11.47 |

CPU VMs: 1/4/8/16/32/64/96 cores, interruptible (exact $/hr not captured; cheap, ~400 GB+ storage). Good for parquet cleaning/feature engineering.

## Modal — serverless, per-second ($30 credit)

Pay-per-second, good for short sweeps; worse per-hour than Lightning-interruptible for long runs.

| GPU | $/sec | $30 buys |
|---|---|---|
| B200 | 0.001736 | ~4.8 h |
| H200 | 0.001261 | ~6.6 h |
| H100 | 0.001097 | ~7.6 h |
| RTX PRO 6000 | 0.000842 | ~9.9 h |
| A100 80 GB | 0.000694 | ~12.0 h |
| A100 40 GB | 0.000583 | ~14.3 h |
| L40S | 0.000542 | ~15.4 h |
| A10 | 0.000306 | ~27.3 h |
| L4 | 0.000222 | ~37.5 h |
| T4 | 0.000164 | ~50.8 h |

CPU $0.0000131/core/sec (min 0.125 cores/container) · Memory $0.00000222/GiB/sec · Volumes $0.09/GiB/mo (incl. 1 TiB/mo free).

## Colab (free) & Kaggle

- **Colab free:** terminal now accessible; T4, ~12 h sessions. Good for mini prototyping and the small ~2-week-window subset.
- **Kaggle:** free-tier GPU (T4/P100) + CPU; ~9 h max session. User is less familiar with it — **verify current quotas before relying.**

## Placement recommendation (per task)

Aligned with `Training.md` §7 build order (cleaning → Path A → B1 → B4 → gated SSL):

| Task | Where |
|---|---|
| Cleaning / feature engineering (parquet, CPU) — **current bottleneck** | Local (16 GB, row-group streaming) **or** Lightning 16–32 core CPU VM (faster) |
| Path A LightGBM / CatBoost / Optuna (tabular) | Local or Lightning CPU; GPU optional/unused |
| Dev neural training (B1 / small seq) | Lightning L4 1× ($1.22/hr) or Colab/Kaggle |
| Full B4 seq2seq / SSL pretrain | Lightning A100-80 ($3.32/hr) or L40S; Modal for short hyperparam sweeps |
| MOMENT embedding extract (side bet) | Lightning L4 / Colab — one-afternoon job |
| ECG model (if upper-bound arm) | GPU, L4 minimum (array memmaps → memory not the constraint) |
| Mini prototyping | Local / Colab / Kaggle |

**Lean path:** clean + GREEN FE on local full subset (`data/full/AI_READI/`) → Path A tabular on local/Lightning CPU → B1 cheap ablation on L4 → B4/SSL on A100/L40S; Modal for short sweeps; Colab/Kaggle ad-hoc. Drop the environment sensor unless adding an exposure-covariate ablation. Do not open GPU-heavy Path B before the Path A floor exists.

## Gotchas

- The `gdrive` vs `gdrive_zyrus:` remote-name mismatch (above) — the only thing that bites if you re-run archive.
- 6 GB AMD GPU locally is non-CUDA; don't plan any local GPU training.
- Drive mounts stall on random access → always copy canonical to local disk first.
- Clinical tables are OMOP **long-format** (source_value rows) — FE is a pivot job, not column select (`DATA_STRUCTURE.md`, `DATA_AUDIT.md` A.2).
- Garmin/dexcom timestamps are tz-aware UTC; clinical dates are **strings** (3 formats). Parse explicitly.
- Effective aux n (≤~1.9k) and train insulin n=105 dominate compute *and* statistical power more than GPU choice.
