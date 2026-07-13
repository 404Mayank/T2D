# Training — Methodology & Approach

> Research/decision log for the ML approach. **Methodology source of truth.** Data facts:
> `DATA_AUDIT.md` (empirical) + `DATA_STRUCTURE.md` (layout); features: `FEATURES.md` (aligned
> with this doc); compute: `COMPUTE.md`. Diasense (teammate's parallel attempt) is referenced
> where it's a useful data point, not as an antagonist — the end product is one combined effort.

## 1. Objective & framing

Predict Type 2 Diabetes (T2D) severity (4-class: 0 healthy → 1 pre-diabetes → 2 oral/non-insulin
injectable → 3 insulin-dependent) from non-invasive Garmin Vivosmart 5 wearables. **Watch-only is
the paper headline.** Onboarding/self-report is the deployable config, reported in a deployment
section — not the claim. CGM (Dexcom) is invasive and is **training-time supervision only**; at
inference there is no CGM.

Reframe the scientific claim as **risk stratification, not early diagnosis** — AI-READI wearables
were recorded *after* participants knew their status, so T2D-positive behavior may reflect
post-diagnosis lifestyle change. This limits external validity to true screening and belongs in
the discussion, but it doesn't kill the watch-only claim.

**Central risk (named upfront):** survey/comorbidity self-report may dominate the wearable signal.
Diasense's survey-only LightGBM (27 features, 4-AUC 0.6963) beat its wearable-only LSTM (0.6725).
If the top-10 SHAP features are all survey items after the final fit, the "from wearables" claim
is strained regardless of AUC. The **SHAP guardrail** (§6) is the response — surface it in the
paper, don't hide it.

## 2. Label, evaluation, pools

- 4-class ordinal label from `metadata/participants.parquet` → `label`. Imbalanced (mini
  31/30/33/6; full distribution in `DATA_AUDIT.md`).
- **Split by `person_id` using `recommended_split`**, never by row. This is a stricter, more
  honest comparison than k-fold CV averages — report a held-out test number.
- **Evaluation variants (all required):**
  - 4-class macro-OVR AUC + AUPRC (AUPRC mandatory under imbalance).
  - Binary healthy-vs-not (screening).
  - Ordinal formulation — prefer **CORN/CORAL** (neural, rank-consistent, drops the
    proportional-odds assumption we suspect fails between pre-diabetes and oral-med) over
    classical proportional-odds regression.
  - Per-class one-vs-rest AUC.
- **Calibration is a required step, not a model.** Risk stratification only means something if
  probabilities are calibrated; AUC is rank-only and hides miscalibration. Under the class-4
  imbalance raw softmax will be badly miscalibrated. Apply isotonic/Platt/Beta on the val fold,
  report calibration curves + Brier score alongside every AUC.
- **Pools** (use **post-clean** numbers for training; raw ceilings in `DATA_AUDIT.md` A.6):
  - **Processed contract:** `PROCESSED.md`. Features: `data/processed/features/watch_green.parquet`
    (n=**1824** = `wearable_core`). Labels/splits: join `meta/pool_masks.parquet`.
  - Direct T2D raw ceiling = 2,280 labeled; after pipeline gates (HR density + stress + sleep≥7):
    **wearable_core = 1824** (Path A default). Raw wearable core was ~2,052 → ≤1,983 post-sentinel.
  - Aux after clean: **aux_eligible = 1685** (raw aux ~2,034 → ≤1,921 at ≥24h before shared window).
  - SpO₂ is only 71.8% coverage — do not gate the primary pool on it.
  - Aux is a non-random subset (CGM-tolerant). **Sex/race unavailable** (`person.parquet` blank).
    Do **not** silently restrict the T2D pool to CGM-haves beyond the published pool flags.
  - **Train insulin n=80** on wearable_core ∩ train (full-split train insulin was 105 before
    coverage filter) — binding 4-class constraint under `recommended_split`.

## 3. The honest performance target

Wearable-only ceilings from literature (FEATURES.md §5): **binary 0.75–0.86, 4-class 0.72–0.75**.
For our 2,280 mixed-control cohort on a held-out split, expect **binary 0.78–0.82, 4-class
0.72–0.75**. Do **not** anchor on the ~0.92 AUC figures from UK-Biobank-style structured data —
those include lab biomarkers (HbA1c, lipids) and are not reachable from a watch. Anchoring on 0.92
manufactures a disappointment that reads as failure. The real bar is "beat the parallel
tabular/sequence baselines on a held-out split, watch-only, with honest calibration."

When citing the teammate's 0.7937 2-AUC, be precise: that is the **ensemble including the 27 survey
features**; the wearable+KD-only number is 2-AUC 0.6846. The jump 0.6846 → 0.7937 is overwhelmingly
the survey block, not the glucose signal. That figure reinforces the survey-dominance warning, it
does not set a bar the glucose emulator clears.

## 4. Method inventory (two paths)

### Path A — direct data → T2D (participant-level summary features)

**LightGBM (baseline, build first).** Native NaN handling (critical given sentinels→missing),
handles mixed tabular types, fast Optuna sweep, SHAP built-in. Use `is_unbalance=True` or class
weights for the class-4 imbalance (**train insulin n=80** on current watch_green cohort). Optuna
found heavy regularization needed at this n (`min_split_gain` 4.0–4.8, `colsample` 0.30–0.37) —
expect the same; don't default to shallow regularization. Input matrix: see `PROCESSED.md`.

**CatBoost (run alongside LightGBM — addition over the prior plan).** Two real reasons here:
(1) the OMOP comorbidity block (`mhoccur_*`) and other categoricals are handled natively without
one-hot blow-up (note: **sex/race are not in this dataset** — onboarding categoricals are thinner
than planned); (2) CatBoost's **ordered boosting** is explicitly designed to reduce target leakage
/ overfitting on small-n — exactly the diagnosed problem at n≈1,586–2,280. Non-redundant with
LightGBM; run both.

**Logistic Regression / linear SVM.** Secondary interpretability floor — honest calibration,
directly interpretable coefficients, near-zero overfitting risk. Not competitive; a sanity check.
Needs imputation (no native NaN).

**Random Forest.** Low-variance cross-check with built-in importance; more robust to outliers
than GBM. Run it as a *secondary* model — but **do not** justify it with the unsourced claim that
"RF outperformed XGBoost and CNN in one wearable study." That claim couldn't be verified and the
local data point contradicts it (RF was the weakest of the teammate's models). RF is worth running
for variance/robustness reasons, not because of that citation.

**Ordinal regression.** The 4-class label is ordered. Prefer CORN/CORAL (above) over classical
proportional-odds, since proportional odds likely fails at the pre-diabetes / oral-med boundary.

**1D-CNN / LSTM on raw 1-min series (optional, later, high-risk).** Captures intra-day dynamics
the tabular models can't, but has a **negative local prior**: hourly resolution (336 steps) was
tried and failed (4-AUC 0.6454 with 28.8% NaN, worse than 14-day daily summaries at 0.6725). Going
finer to 1-min (20,160 steps) worsens sparsity and data-hunger. A cold-start raw model at this n
is likely to underperform hand features. **The path to making end-to-end viable is self-supervised
pretraining first (§5), not a cold-start CNN.** Build only after the direct baseline is solid and
only behind a pretrained encoder.

### Path B — data → glucose auxiliary → T2D (the contribution angle)

CGM is privileged information: available at training, absent at inference. This is textbook
**Learning Using Privileged Information (LUPI, Vapnik 2015)**. Naming the framework gives the paper
a principled justification instead of ad-hoc "we regressed glucose then fed it," and opens
non-KD formulations.

**B1 — Multi-task joint (shared backbone + glucose head + T2D head).** *Ablation, not the
headline.* The teammate already built and abandoned this (BiLSTM hidden=128, glucose regression
on 8 daily CGM summaries, `L = L_class + λ·L_glucose`, λ ∈ {0,0.3,0.5,1.0}): glucose helped only
+0.003 4-AUC, abandoned for KD. **Caveat that rescues it as an ablation:** their multi-task used a
*different, larger backbone* than their proven model — a confound. Run a **controlled** version
(identical 64-hidden attention backbone, ± glucose head) to settle whether the failure was their
architecture or the idea. Cheap; belongs in the ablation table either way.

Do **not** claim B1 > two-stage as established. Research confirms **no universal evidence that
multi-task beats two-stage** — it's dataset/task-dependent and task-gradient conflict is a real
failure mode (which is why GradNorm / uncertainty weighting exist as fixes, not guarantees). Use
uncertainty weighting or GradNorm for task balance; run both B1 and B2 and let the ablation decide.

**B2 — Two-stage (glucose emulator → T2D predictor).** *Ablation.* Modular, debuggable, clean
deployment story (swap stage 1 for watch inference). Risk: error compounding — stage-1 regression
errors propagate into stage 2, and the point-estimate handoff is lossy vs a shared embedding.
Defensible as a rigorous ablation to justify the joint/seq2seq representation; not the method.

**B3 — Knowledge distillation (CGM teacher → wearable student, soft logits).** *Strong baseline to
beat, not a contribution.* This is the teammate's method (T=2, α=0.3 best; recovers ~20% of the
teacher's gap). Principled leakage handling via out-of-fold teacher predictions. Your delta over
it must be a *different* formulation, not a redo.

**B4 — Seq2Seq CGM-trajectory teacher → T2D head (headline candidate).** A seq2seq model predicts
the **full CGM trajectory** from wearables (not a scalar / 8 summaries), and the encoder
representation feeds the T2D head. The teammate regressed 8 scalar summaries, never the curve —
this is the one aux formulation they didn't touch. Predicting the full glucose curve forces the
encoder to model glycemic dynamics, a richer inductive bias than regression-to-mean or soft-label
KD. **This is the real headline candidate.** Complexity is higher (sequence decoder, CGM↔wearable
timestamp alignment/windowing); needs ablation to confirm the richer decoder actually improves
downstream T2D AUC.

**Representation distillation under LUPI (the cleanest novelty delta).** Instead of distilling
soft class logits (B3 / teammate's KD), distill the **glucose-shaped intermediate representation**
from the teacher into the student. The privileged signal in the features is arguably stronger than
in the logits. This is a principled, distinct delta and pairs naturally with B4's encoder.

## 5. Additions to the menu (researched this session)

**Self-supervised / contrastive pretraining on the unlabeled wearable pool (first-class option,
biggest omission in the prior plan).** ~2,280 participants of unlabeled wearable time series vs
~≤1,921 effective aux-labeled (post-sentinel + overlap) / ~1,586-class Diasense-style filtered
T2D pool. SSL (SimCLR/BYOL-style augment-and-contrast, or masked-value reconstruction) on *all*
participants pretrains an encoder the small labeled pool then fine-tunes. Active 2024–2025
direction for ECG/PPG in few-label clinical settings; architecturally orthogonal to the
glucose-auxiliary idea (stack them: SSL backbone + glucose head + T2D head). This is what makes
the raw 1-min end-to-end model *learnable* at this n — the direct answer to the cold-start-CNN
negative prior. **Gate:** open only if Path A watch-only leaves dynamics on the table (see
`FEATURES.md` §0 exploration ladder).

**Time-series foundation models (MOMENT; TimesFM/Chronos/Moirai secondary).** MOMENT is pretrained
on public time series and supports classification directly — fine-tune a head, or extract frozen
embeddings → LightGBM (a one-afternoon experiment). Honest caveat: these are trained on generic
time series and transfer to physiological signals is partial and unproven — treat as a
high-variance side bet, not a default. TimesFM/Chronos are forecasting-first, less directly useful
for classification than MOMENT.

**Per-modality encoders + cross-attention late fusion.** Modalities have wildly different sampling
and missingness (HR 1-min, SpO₂ **71.8%** coverage, sleep/activity interval-based; RR ~2× denser
than HR). Separate lightweight encoders per modality with a learned cross-attention fusion layer
is more robust to per-modality dropout than forcing everything through one shared backbone. The
teammate fuses via concat → Linear; an attention-fusion ablation is an unfilled architectural cell.

**Generative / time-series augmentation for the minority class.** Class weights / focal loss shape
the *loss* but add no information. For insulin (full n=258, **train n=80** on wearable_core), especially for
sequence models where data hunger is binding, **time-series augmentation**
(jitter/scaling/magnitude-warping/permutation) or **SMOTE on summary features** / conditional
generative on sequences grows the rare-class signal. Controlled experiment specifically for the
sequence/aux models — not a substitute for locking class weights before feature selection.

## 6. SHAP guardrail & decision bar

- **SHAP guardrail:** after the final fit, if the top-10 SHAP features are all survey items → the
  "from wearables" claim is dead; revisit watch feature engineering or framing. Report SHAP
  alongside permutation importance.
- **Decision bar (pre-registered):** a feature block stays deployable iff ΔAUC **>+1.0** with
  non-overlapping nested-CV CIs **and** stable permutation importance. Below that, dropped
  regardless of SHAP.
- **Block-ablation hierarchy** (fixed split + person-bootstrap CIs; package `path_a_blocks/`):
  watch-only → +hard onboarding (**age, BMI/waist/height, family history, BP** — no sex/race;
  smoking not in current onboarding file) → +comorbidities (HTN+dyslipidemia first) → +mood
  (CES-D-10; PAID) → +diet. Report ΔAUC + ΔAUPRC per block with CIs.
  **Status:** watch-only + 1A onboarding **done** (see `training/path_a_blocks/REPORT.md`).
- Decide the class-imbalance strategy (weights / focal loss) **before** feature selection so
  importance isn't confounded by the loss.

## 7. Build order (revised)

0. **Cleaning + GREEN feature matrix** — **done for v1 defaults** (`CLEANING.md` / `PROCESSED.md`):
   `watch_green` n=1824, join labels from `pool_masks`. Re-run pipeline only if config/policy
   changes (`DATA_AUDIT.md` §B / `CLEANING.md`).
1. **Direct LightGBM + CatBoost** on GREEN summary features, nested CV on `recommended_split`,
   all label formulations (multiclass / binary / **CORN ordinal**). Add **isotonic calibration** +
   Brier. Floor *and* honest reference. Then block-ablate survey add-ons under the §6 hierarchy.
   - **Status (2026-07-13):** Watch-only floor **done** — package `training/path_a_watch/`
     (CatBoost test 4-AUC **0.666**, binary **0.689**; fixed `recommended_split`, freeze-before-test).
     Block ladder started in `training/path_a_blocks/`: diagnostics + **1A watch+onboarding**
     (test 4-AUC **0.699**, binary **0.749**, decision_bar_pass). See
     `training/path_a_blocks/REPORT.md`. CORN neural ordinal still deferred; cal remains diagnostic.
     **Next:** 1B comorbidity (HTN-first) under same decision bar.
2. **Controlled B1 ablation** — same 64-hidden attention backbone, ± glucose head, summary-CGM
   target. Not the headline; settles whether the teammate's multi-task failure was architecture
   or idea.
3. **B4 (seq2seq full-CGM-trajectory teacher → T2D head)** as the **headline candidate**, paired
   with **representation distillation** under a LUPI framing — the cleanest novelty delta and the
   one aux formulation untouched by the parallel attempt.
4. **B2 (two-stage) ablation** — proves the joint/seq2seq representation does work the
   point-estimate handoff can't.
5. **SSL-pretrained backbone** (gated) as the lever to make the raw 1-min end-to-end model viable
   (instead of a cold-start CNN the hourly-failure prior says will underperform). MOMENT-embedding
   + GBM as a one-afternoon high-variance side bet.
6. **Generative augmentation + calibration** wherever the sequence/aux models show minority-class
   data hunger or miscalibration.

## 8. What is explicitly not the contribution

- Plain logit-KD (B3) — that's the teammate's method.
- Multi-task on scalar CGM summaries (B1) — already tried and abandoned; only a controlled
  ablation here.
- A redo of the teammate's ensemble — the delta must be a *different* formulation (B4,
  representation distillation, SSL + aux, attention fusion) or the deployment angle.

## 9. Open methodology questions

- CORN vs CORAL: pick one ordinal loss; CORN drops proportional-odds (safer given suspected
  boundary failure), CORAL assumes it. **Lean CORN.**
- Task-weighting scheme for B1/B4: uncertainty weighting vs GradNorm — small sweep.
- Whether to train the T2D head on all wearable-eligible participants while the glucose head
  trains on the effective aux subset with a shared backbone (avoid silently restricting T2D to
  CGM-haves).
- SSL augmentations appropriate for PPG HR + interval sleep/activity (not all image-style
  augmentations transfer cleanly).
- Whether MOMENT embeddings are worth keeping given generic-pretraining transfer caveat.
- Min CGM↔wearable overlap threshold for aux (≥24h vs ≥72h) and year-long-wear truncation policy
  — lock with cleaning, before B4 alignment work (`DATA_AUDIT.md` B5.2 / B4.3 / B10).

**Closed (data/feature side; see `FEATURES.md` §12):** sex/race unavailable; stress 0–100;
headline = B4+rep-distill not multi-task; `condition_occurrence` is self-report not ICD;
`clinical_site`→TZ map is feasible and mandatory; SpO₂ stays Tier-3 at 71.8% coverage.
