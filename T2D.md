# T2D — ML Project Notes

Working notes for the ML side. Canonical detail lives in the other project docs — this is the
1-page "why."

| Doc | Role |
|---|---|
| `DATA_STRUCTURE.md` | Layout, schemas, access |
| `DATA_AUDIT.md` | Empirical audit + cleaning checklist (source of truth for data facts) |
| `FEATURES.md` | Feature inventory, leakage rules, GREEN core |
| `Training.md` | Methodology / Path A–B / build order (source of truth for ML) |
| `COMPUTE.md` | Machines, Drive, GPU placement |
| `training/path_a_watch/` | Watch-only GBM floor implementation |
| `training/path_a_blocks/REPORT.md` | Latest Path A diagnostics + onboarding results |

## Objective

Predict Type 2 Diabetes **severity** (risk stratification, not early diagnosis) from non-invasive
Garmin Vivosmart 5 wearables.

- **Label** (`metadata/participants.parquet` → `label`): `0` healthy → `1` pre-diabetes →
  `2` oral/non-insulin injectable → `3` insulin-dependent.
- **Paper claim:** **watch-only.** Onboarding/self-report is the deployable config (deployment
  section), not the scientific claim.
- **CGM (Dexcom)** is invasive → **training-time privileged supervision only** (LUPI). At inference
  there is no CGM.
- **Post-diagnosis caveat:** wearables were recorded after participants knew their status →
  behavior may reflect lifestyle change. Limits screening external validity; discuss honestly.

### Two-stage / aux idea (Path B)

Historical framing that still motivates deployment:

1. **Glucose auxiliary:** learn glycemic structure from wearables under CGM supervision (train only).
2. **T2D predictor:** use wearables (+ glucose-shaped representation at train time) for the 4-class label.

**Headline formulation** (see `Training.md`): not plain "regress glucose then feed," and not
Diasense logit-KD. Primary novelty candidate is **B4 — seq2seq full-CGM-trajectory teacher → T2D
head**, with **representation distillation under LUPI**. Path A (direct LightGBM+CatBoost on
summary features) is built first and is the floor every aux result is measured against.

**Path A status (2026-07-13):** Watch-only CatBoost floor test 4-AUC **0.666** / binary **0.689**
(`training/path_a_watch/`). Deployable **watch+onboarding** 4-AUC **0.699** / binary **0.749**
with decision_bar_pass (`training/path_a_blocks/REPORT.md`). Next block: comorbidity (1B).

## Diasense — teammate baseline to beat / stay distinct from

- Repo: `github.com/mannangrover/Diasense`
- **Approach:** knowledge distillation — CGM teacher (wearable+CGM LSTM) distills **soft class
  logits** into a wearable+survey student. Inference: no CGM. Out-of-fold teacher predictions
  handle leakage. This is **B3** in our inventory — strong baseline, **not** our contribution.
- **Features:** 41 wearable/day + 27 survey (from `clinical/observation.parquet` long-format
  source_values).
- **Models:** LSTM+Attention + Optuna LightGBM ensemble.
- **Results (n=1,586 after coverage filter; random 5-fold — not `recommended_split`):**
  - Ensemble 2-AUC **0.7937**, 4-AUC **0.7412**
  - Wearable+KD-only 2-AUC **0.6846** — the 0.6846→0.7937 jump is overwhelmingly the survey block
  - Survey-only LightGBM 4-AUC **0.6963** beat wearable-only LSTM **0.6725**
- **Key findings:** survey ≈ wearable; HRV/circadian-HR-amplitude/sleep-efficiency strongest
  wearable predictors; KD transfers glucose-correlated signal; multi-task on scalar CGM summaries
  was tried and abandoned (+0.003 4-AUC, confounded backbone).
- **Our delta must be a different formulation** (B4 trajectory teacher, rep-distill, SSL+aux,
  attention fusion) or the deployment angle — not a redo of logit-KD + ensemble.

## Data / training pools

Full AI-READI v3.0.0, n=2,280. Empirical numbers from `DATA_AUDIT.md` (not ceilings):

| Pool | n (approx) | Notes |
|---|---|---|
| All labeled (direct T2D ceiling) | 2,280 | label 0/1/2/3 = 776/560/686/258 |
| Wearable core HR∩stress∩sleep | 2,052 raw → **≤1,983** | after post-sentinel zero-valid removal |
| Aux HR∩stress∩RR∩sleep∩CGM | 2,034 raw → ≤1,963 → **≤1,921** | after sentinel + ≥24h CGM↔HR overlap |
| CGM-haves | 2,245 | CGM∩HR raw 2,085; 51 have ≤0h temporal overlap |
| Diasense-style filtered | ~1,586 | their dead-sensor/coverage cut; re-lock ours post-clean |

- **Train insulin n=105** is the binding 4-class constraint (`recommended_split` train/val/test =
  1576/352/352).
- **Do not silently restrict the T2D pool to CGM-haves.** Aux is a non-random (CGM-tolerant) subset.
- **No sex/race** in this release (`person.parquet` demographics blank). Hard onboarding =
  age/BMI/waist/FH/smoking/BP only.
- Stress scale is **0–100** (not 0–17). ~63 pids have true year-long wear (truncate policy TBD).
- Site×label confound (UAB enriched for insulin) → **UTC→local mandatory** before circadian features.

Canonical on Drive: `gdrive_zyrus:AI_READI/{mini|full}/AI_READI/`. Relevant subset also local at
`data/full/AI_READI/` (~784 MiB). Layout/schemas: `DATA_STRUCTURE.md`. Cleaning plan: `DATA_AUDIT.md` §B.

## Honest performance target

Wearable-only literature ceilings: binary ~0.75–0.86, **4-class ~0.72–0.75**. For this
mixed-control held-out split expect binary ~0.78–0.82, 4-class ~0.72–0.75. Do **not** anchor on
~0.92 lab-biomarker (HbA1c/lipids) figures. Real bar: beat matched tabular/sequence baselines
watch-only, with calibration (Brier + curves), under the SHAP survey-dominance guardrail.

## Training setup note

Copy canonical to local disk before training (Colab `/content` or a VM). Drive mounts stall on
random access. This machine already has the relevant full subset under `data/full/AI_READI/`.
Local GPU is non-CUDA (useless for mainstream stacks) — train on Lightning/Modal/Colab; clean/FE
can stay local. Placement detail: `COMPUTE.md`.
