# T2D — ML Project Notes

Working notes for the ML side. For the data layout/schemas/access, see `DATA_STRUCTURE.md`.

## Objective & two-stage idea

Predict Type 2 Diabetes (T2D) from non-invasive wearable data (Garmin Vivosmart 5).

- **Stage 1 — glucose emulator:** learn to predict CGM blood-glucose from wearable signals (heart_rate, physical_activity, sleep, stress, oxygen_saturation, respiratory_rate). Dexcom CGM is the supervision target, used **only at training**.
- **Stage 2 — T2D predictor:** predict the T2D label from wearable data + the emulated glucose signal.
- **Deployment motivation:** a smartwatch has wearables but no CGM — the emulator supplies a glucose proxy at inference, so the T2D predictor runs watch-side without invasive sensors.
- **Label** (in `metadata/participants.parquet`): `0`=healthy, `1`=pre-diabetes, `2`=oral-medication, `3`=insulin-dependent.

## Diasense — teammate's baseline to beat

- Repo: `github.com/mannangrover/Diasense`
- **Approach:** knowledge distillation — a CGM teacher (wearable+CGM LSTM) distills soft predictions into a wearable+survey student. At inference, no CGM. This is a more principled variant of the explicit emulator above (leakage handled via out-of-fold teacher predictions). Worth considering over a plain "regress glucose then feed" design.
- **Features:** 41 wearable features/day (HR stats, HRV/RMSSD, circadian HR amplitude, sleep efficiency/WASO, activity bouts, SpO₂ desaturation events) + 27 survey features (demographics, comorbidities, CES-D/PAID, family history, lifestyle — extractable from `clinical/observation.parquet`).
- **Models:** LSTM+Attention + Optuna-tuned LightGBM ensemble.
- **Results:** 2-AUC 0.7937, 4-AUC 0.7412 (n=1,586 after dead-sensor/coverage filtering).
- **Key findings:** survey-only features are very predictive (≈ wearable LSTM); HRV, circadian HR amplitude, and sleep efficiency are the strongest wearable predictors; KD successfully transfers glucose-correlated signal.
- Your contribution should be a **distinct delta** (e.g., a better distillation target, end-to-end sequence modeling that beats hand features, or the watch-deployment angle) — not a redo of the above.

## Data / training pool

- 2,280 participants total (full variant).
- 2,245 have Dexcom CGM; **~1,975–2,085 have wearable ∩ dexcom** (the glucose-emulator supervision pool — the intersection, varies per modality).
- Diasense filtered 2,280 → 1,586 on dead-sensor/coverage; expect to filter similarly after masking sentinels.
- Canonical is on Google Drive at `AI_READI/full/AI_READI/` — layout/schemas/access in `DATA_STRUCTURE.md`.

## Training setup note (Drive)

Copy the canonical to local disk before training (Colab `/content` or a VM). Drive mounts stall on random access and can hang training loops. The layout is few large files (Drive-friendly), but local disk is still faster — copy once, read locally.
