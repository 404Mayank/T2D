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
**Status (2026-07-16):** lean-CORN tabular MLP on exact C1 **ran / null vs C1** — package
`training/path_a_raise_corn/` (test 4-AUC 0.706; CE control 0.713; bar fail; C1 unchanged).
Post-freeze multi-seed bag + cross-family ensemble raise also **null** — package
`training/path_a_raise_ensemble/` (S=5 best ΔAUC +0.006; S=10 +0.003; bar fail; C1 unchanged).

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
headline — **frozen 2026-07-15** (plain-λ); **GS sibling frozen 2026-07-16**.* Controlled
`attn_lstm_64` + day-level 8-vector glu head: post FE/scale fix, pure-seq test 4-AUC **0.652**;
λ=0.5 multi-task **null** (paired boot CI lo≯0); GREEN late-fuse **no raise**. Gradient-balanced
retry (PCGrad + uncertainty weighting, `b1gs_grid_20260716`) also **null** (best Δ +0.0006,
CI lo≯0) despite moderate conflict (~20% steps); glu head stays sub-constant (val z-MSE~1.32).
Pre-fix ~0.51 grid was broken sleep FE + unscaled inputs — not a ceiling. See
`training/path_b/REPORT_B1.md` + `REPORT_B1_GS.md`. Do **not** reopen B1 λ/GS grids without a new
`PLAN_*`; ladder **B2 (frozen) → B4 (concluded) → B3 (frozen)**; siblings B1-GS / B2-V2 frozen.

Do **not** claim B1 > two-stage as established. Research confirms **no universal evidence that
multi-task beats two-stage** — dataset/task-dependent; task-gradient conflict is real. Run B2
next and let the ablation decide against the frozen B1 floor.

**B2 — Two-stage (glucose emulator → T2D predictor).** *Ablation — **frozen 2026-07-15**.*
Person GREEN → 8 daymean CGM → C1/W0 Stage-2: predicted handoff **null** (T1 test 4-AUC **0.7345**
vs matched D1/C1 **0.7378**; Δ CI lo≯0). **No deployable B2 arm beats C1** (best deployable = D1 ≡ C1).
Oracle true-CGM O1 **0.823** vs matched D1a (**+0.094**) proves headroom but is privileged + aux-only;
Stage-1 R²~0.05 is the bottleneck. See `training/path_b/REPORT_B2.md`.

**B2-V2 — sibling retry (daily MSE mid + variance pack) — frozen 2026-07-16** (`b2v2_grid_20260716`).
Daily watch→CGM reduced-Y emulator + mid/spread/daysd handoff: T1v **0.727** / T1p **0.730** still
**≯** matched D1 **0.7378** (all Δ CI lo≯0); O1−D1a **+0.096**. Stage-1 val mean R² ~0.09 / test ~0.03.
Adversarial post-hoc audit: null **authentic** (no blocker bug; T1v still uses ~18% yhat importance).
See `training/path_b/REPORT_B2_V2.md`. Do **not** reopen B2/B2-V2 claim HPO without a new plan.
Ladder complete with **B3 frozen**; siblings **B1-GS / B2-V2 / B4-V2** also frozen null. Further work only via new `PLAN_*` (e.g. SSL).

**B4 — Seq2Seq CGM-trajectory + rep-distill (headline cell) — concluded 2026-07-15; V2 sibling null 2026-07-16.**
5-min multi-modal grid FE + CNN patch encoder. **B4-A** traj multi-task: S0 test 4-AUC **0.646**;
Sλ−S0 **null**; hybrid z∥C1 best **0.714** &lt; matched D1 **0.736** (≈ freeze C1). Wear→curve Pearson
~0.14–0.26 (non-degenerate recon, weak transfer). **B4-B** rep-distill (easy X∥cgm + hard
cgm_only / wear→cgm teachers): student μ&gt;0 **null/hurts**; best hybrid **0.735** still ≯ D1.
Easy-teacher loophole closed by hard-teacher sensitivity. **B4-V2** (RKD/CRD + PCGrad + OOF fusion):
PCGrad hurts/null (no grad conflict); RKD/CRD μ&gt;0 null/hurts; best hybrid F1 OOF **0.726** ≯ D1 **0.736**;
H2 teacher Pearson **0.30** + probe GO. **No deployable B4/B4-V2 arm beats C1.**
Privilege remains real (B2 oracle; teacher/CGM probes). Authority: `REPORT_B4.md`,
`REPORT_B4_B.md`, `REPORT_B4_B_HARD.md`, `REPORT_B4_V2.md`. Do **not** reopen without a new `PLAN_*`.

**B3 — Knowledge distillation (CGM teacher → wearable student, soft logits).** *Strong baseline,
not a contribution — **frozen 2026-07-15**.* Diasense-style OOF soft logits (T=2, α=0.3 decision):
`G_α=0.3` test 4-AUC **0.7469** vs matched D1/C1 **0.7378** (Δ **+0.009** CI lo≯0 → **null**);
binary **worse** than C1; Hinton MLP `N_α=0.3`−N0 **null**; teacher Tch **0.823** vs D1a (+0.076).
See `training/path_b/REPORT_B3.md`. Do **not** reopen B3 claim grids without a new `PLAN_*`.

**Representation distillation under LUPI** was tested as B4-B (L2; easy + hard teachers) and
**B4-V2** (RKD/CRD + PCGrad MTL + OOF fusion): alignment/mechanism can work (or teacher GO) but
4-AUC does not raise; hybrids still ≤ C1/D1 (`REPORT_B4_V2.md`). Residual novelty may still live
in **SSL+aux**, attention fusion, or deployment framing — not “redo B4 λ/μ or RKD/CRD grids.”

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
  smoking was an FE gap: raw `susmk*`, now extractable via `build_smoking_features.py`) →
  +comorbidities (HTN+dyslipidemia first) → +mood (CES-D-10; PAID) → +diet. Report ΔAUC +
  ΔAUPRC per block with CIs.
  **Status (2026-07-14):** Path A tabular **frozen**. Watch floor 0.666; 1A onboarding bar-pass
  (0.699); 1B comorbidity bar-fail; 1C mood bar-pass (0.738 / binary 0.831, PAID-driven); wrap
  ablations keep **C1** as secondary (minimal retention fail; binary HPO no gain). Post-freeze
  C1 sensitivities (smoking / `mhoccur_obs` / `via1–3` / joint) **all bar-fail** — C1 unchanged.
  Diet not run. Post-freeze **CORN MLP raise** (2026-07-16) on exact C1 matrix: **bar-fail null**
  (test 4-AUC 0.706 vs C1 0.738, Δ−0.031, CI entirely &lt;0; CE control 0.713; CORN≈CE). Post-freeze
  **ensemble raise** (2026-07-16) multi-seed bag + LGBM/CatBoost blend/stack: **bar-fail null**
  (S=5 E_arith Δ+0.006; S=10 Δ+0.003; CIs include 0). C1 unchanged.
  See `training/path_a_blocks/REPORT_A_WRAP.md` + `training/path_a_raise_corn/REPORT.md` +
  `training/path_a_raise_ensemble/REPORT.md`.
- Decide the class-imbalance strategy (weights / focal loss) **before** feature selection so
  importance isn't confounded by the loss.

## 7. Build order (revised)

0. **Cleaning + GREEN feature matrix** — **done for v1 defaults** (`CLEANING.md` / `PROCESSED.md`):
   `watch_green` n=1824, join labels from `pool_masks`. Re-run pipeline only if config/policy
   changes (`DATA_AUDIT.md` §B / `CLEANING.md`).
1. **Direct LightGBM + CatBoost** on GREEN summary features, **fixed `recommended_split`**
   (person-bootstrap CIs for block Δ; not reshuffled nested k-fold), all label formulations
   (multiclass / binary / **CORN ordinal**). Add calibration + Brier. Floor *and* honest reference.
   Then block-ablate survey add-ons under the §6 hierarchy.
   - **Status (2026-07-14):** Path A tabular **frozen**. Watch-only floor **done**
     (`training/path_a_watch/`, CatBoost test 4-AUC **0.666**, binary **0.689**). Block ladder
     complete in `training/path_a_blocks/`: 1A **0.699** bar-pass; 1B core bar-fail; 1C mood
     **0.738 / 0.831** bar-pass; wrap (minimal/PAID/severity/binary) → secondary pick **C1**;
     C1 sensitivities smoke/obs/via **null**. Authority: `REPORT_A_WRAP.md` + `DECISIONS.md`.
     **CORN neural ordinal raise done / null (2026-07-16)** — package `training/path_a_raise_corn/`;
     CORN MLP 0.706 / CE-MLP 0.713 ≯ C1 0.738; primary bar fail justified (`REPORT.md`).
     **Ensemble raise done / null (2026-07-16)** — package `training/path_a_raise_ensemble/`;
     Bag_cat / E_arith / stack all bar-fail vs C1 (best S=5 Δ+0.006).
     Cal remains diagnostic. **Path B / B1 frozen (2026-07-15
     plain-λ; 2026-07-16 GS):** C1 sleep FE + C2 input z-score fixed; pure-seq test 4-AUC **0.652**
     (pre-fix 0.51 was broken inputs); multi-task λ=0.5 **null** (Δ≈0, CI lo≯0); GREEN late-fuse
     **no raise** (0.638); PCGrad/UW GS retry also **null** (`b1gs_grid_20260716`). Authority:
     `training/path_b/REPORT_B1.md` + `REPORT_B1_GS.md` + `REPORT_B2.md` + `REPORT_B4*.md` +
     `DECISIONS.md`. **Path B ladder complete (B1–B4+B3); no deployable raise vs C1.**
2. **Controlled B1 ablation** — **done / frozen** (plain-λ + gradient-balanced). Same 64-hidden
   attention backbone, day-level glucose head, summary-CGM target. Multi-task is a clean null after
   FE/scale repair *and* after PCGrad/UW; pure spine does not beat Path A watch GBM (0.666). Not
   the headline. See `REPORT_B1_GS.md`.
3. **B2 (two-stage) ablation** — **done / frozen.** Predicted person-CGM handoff null; oracle headroom real.
4. **B4 (seq2seq traj + rep-distill)** — **done / concluded null** for deployable raise (A + B easy +
   hard teachers). Grid FE + package `training/path_b/b4/`. See `REPORT_B4*.md`.
5. **B3 logit-KD baseline** — **done / frozen** (Diasense-style; G_α=0.3 null vs C1; `REPORT_B3.md`).
6. **SSL-pretrained backbone** (gated) as the lever to make the raw 1-min end-to-end model viable
   (instead of a cold-start CNN the hourly-failure prior says will underperform). MOMENT-embedding
   + GBM as a one-afternoon high-variance side bet.
7. **Generative augmentation + calibration** wherever the sequence/aux models show minority-class
   data hunger or miscalibration.

## 8. What is explicitly not the contribution

- Plain logit-KD (B3) — that's the teammate's method.
- Multi-task on scalar CGM summaries (B1) — already tried and abandoned; only a controlled
  ablation here.
- A redo of the teammate's ensemble — the delta must be a *different* formulation (B4,
  representation distillation, SSL + aux, attention fusion) or the deployment angle.

## 9. Open methodology questions

- CORN vs CORAL: **lean-CORN primary run done / null vs C1** (`path_a_raise_corn/`, 2026-07-16).
  CORAL sibling and unweighted-CORN ablation **not** required to close the raise; optional only.
  Wording: null is “CORN MLP + median-impute + weighted recipe ≯ C1 GBM,” not “ordinality disproven.”
- Multi-seed bagging + cross-family ensemble vs C1: **done / null** (`path_a_raise_ensemble/`,
  2026-07-16; S=5 + mandatory S=10). Do not re-open without a new `PLAN_*`.
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
