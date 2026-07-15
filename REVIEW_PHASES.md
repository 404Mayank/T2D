# Retrospective review — Path A / B1 / B2 / B4 (why nothing beat C1)

**Scope:** Audit whether each completed phase was executed *correctly* and whether
*scope for improvement* was left on the table, given the central finding: **no Path B arm
and no Path A raise beat the C1 deployable baseline (test 4-AUC 0.738 / binary 0.831).**
Sources: project reports (`REPORT_A_WRAP.md`, `REPORT_B1/B2/B4*.md`, `AUDIT_B1_UNDERPERF.md`,
`REPORT.md`, `PROCESSED.md`, `Training.md`, `FEATURES.md`) + targeted web research on every
technique used (LightGBM/CatBoost, LUPI, MTL gradient conflict, CORN/CORAL, logit/feature/CRD
distillation, SSL/MOMENT for wearables, calibration).

**Authority note:** This is an *analytical review*, not a re-derivation. It does not change
any frozen number. All run claims remain as logged in their `REPORT_*.md` / `DECISIONS.md`.

---

## 0. Verdict (up front)

The phases were executed **honestly and (mostly) correctly** under the pre-registered
protocols. The nulls are **real for the formulations run**, not artifacts of bugs or
leakage. **But** most Path B cells used the *simplest possible* instantiation of their
technique (plain-weighted multi-task, point-estimate two-stage, L2/MSE representation
distillation, cold-start sequence encoders). Some stronger variants have since been
executed: **B1 GS** (PCGrad + UW, 2026-07-16 — null with moderate conflict measured;
`REPORT_B1_GS.md`), **B3 logit-KD** (concluded null; `REPORT_B3.md`), and **B4-V2**
(RKD/CRD + PCGrad traj MTL + OOF fusion, 2026-07-16 — null; `REPORT_B4_V2.md`; H2 Pearson
~0.30; no encoder grad conflict; best hybrid 0.726 ≯ D1 0.736). Still open from the original
list: SSL-pretrained backbones, CORN, threshold-calibrated screening (and only via new
`PLAN_*`). "Nothing beat C1" is best read as: *the tested LUPI recipes (including B1 GS, B3,
and B4-V2 stronger KD/MTL/fusion) did not beat a strong tabular baseline on this cohort* —
not as: *every representation path is exhausted*. Residual scope concentrates in untried
levers (chiefly **SSL**).

| Phase | Done correctly? | Null authentic? | Scope left? |
|---|---|---|---|
| **Path A** (watch floor + block ladder + CORN + ensemble raises) | Yes; minor limitations | Floor 0.666 real; C1 0.738 real; CORN MLP null real; ensemble bag/blend/stack null real (`path_a_raise_*/`) | Real — cal/op-point, nested-CV companion, GREEN v2 FE, diet; CORN + ensemble primaries closed |
| **B1** (multi-task) | Protocol correct after C1+C2 fix; **pre-fix was buggy** | Post-fix plain-λ null real; **GS (PCGrad/UW) also null** (`REPORT_B1_GS.md`) | GS family on day spine **closed**; residual — SSL backbone, ordinal loss (separate plans) |
| **B2** (two-stage) | Correct, protocol-matched, parity-verified | Null real; oracle proves privilege is real | **Attacked in B2-V2** (daily + variance pack) — still null; SSL/deep Stage-1 remains open only via new plan |
| **B4-A** (traj multi-task) | Correct; overfit gate clean | Null real for λ-weighted traj MTL + frozen-z hybrid | **B4-V2 PCGrad** also null (no conflict; hurts class); residual SSL / learned FiLM only if new plan |
| **B4-B** (rep-distill, easy+hard) | Correct; easy-teacher loophole properly closed | Null real for L2-z / MSE distillation | **B4-V2 RKD/CRD** also null (`REPORT_B4_V2.md`); residual SSL not more KD objectives |
| **B4-V2** (RKD/CRD + PCGrad + OOF) | Correct; teacher GO; S0 parity; post-claim **approve-with-caveats** | Null authentic for stronger recipes | Family closed for cold PatchCNN; F3 FiLM trigger fired-not-run (residual disclosure only); **SSL** remains the coherent unbuilt lever |

**One-line synthesis:** most nulls were nulls of *one recipe*; B1 GS and B4-V2 closed the
"we only tried the naive member" gap on day-spine MTL and B4 KD/fusion without a class lift.
Residual high-value untried lever is chiefly **SSL** (plus Path A polish).

---

## 1. What each phase actually did (recap, frozen numbers)

| Phase | Method | Deployable result | vs C1 (0.738/0.831) |
|---|---|---|---|
| **A floor (W0)** | CatBoost Ordered, 30 GREEN watch features | 4-AUC **0.666** / bin **0.689** | paper headline |
| **A 1A** | + hard onboarding (age/BMI/FH/BP…) | 0.699 / 0.749 | bar-pass vs floor |
| **A 1B** | + comorbidity core | 0.709 / 0.778 | **bar-fail** vs 1A |
| **A 1C = C1** | + mood (PAID carries it; CES near-null) | **0.738 / 0.831** | **the bar to beat** |
| **A wrap** | minimal/paid_only/ces/severity/binary HPO | best ≈C1; binary HPO *worse* than derived | no raise |
| **B1** | BiLSTM+h=64 + day-8 CGM multi-task, λ∈{0,.5}; **GS** PCGrad/UW | pure seq 0.652; plain-λ + GS null (GS best Δ +0.0006, CI lo≯0); GREEN late-fuse 0.638 | no raise |
| **B2** | Stage-1 LGBM person-CGM emulator → Stage-2 CatBoost | T1 0.7345 / 0.814 (slightly *harms* binary); oracle O1 **0.823**/**0.877** (privileged, non-deployable) | no raise; **+0.094 oracle headroom** |
| **B4-A** | PatchCNN 5-min grid + per-bin traj MTL, λ∈{0,.3,1} + frozen-z∥C1 hybrid | S0 0.646; MTL null; hybrid **0.713–0.703** < D1 0.736 | no raise |
| **B4-B** | CGM-AE teacher → wear student, L2 z-MSE distill, μ∈{0,.3,1}; easy+hard teachers | student μ>0 *hurts* (μ=.3 CI<0); hybrid recovers to 0.735 at μ=1, still ≯ D1 | no raise |

Class-2 (oral med) is the persistent 4-class bottleneck (~0.60–0.63 OVR across every stack);
ends of spectrum drive macro-OVR. This is a *label-geometry* problem, which matters for the
CORN recommendation below.

---

## 2. Correctness assessment (per phase)

### 2.1 Path A — correct, with documented limitations

**Done right:**
- Fixed `recommended_split` (by `person_id`), not row shuffling — strictly honest vs the
  teammate's random 5-fold. Class weights locked (`balanced`) **before** feature/importance
  analysis, exactly as `Training.md` mandates.
- Pre-registered decision bar (ΔAUC > +0.01 AND paired bootstrap CI lo > 0 AND stable perm
  importance), applied uniformly. C1 retention rules applied even when minimal_M "almost"
  held (no soft-promotion past the rule).
- Leakage exclusion is the most thorough part of the project (`FEATURES.md` §3): HbA1c /
  insulin / T2D meds / retinopathy / `via4–6` etc. all hard-excluded; retinal metadata leak
  into clinical tables filtered; `condition_occurrence` correctly identified as self-report
  dup (not ICD).
- SHAP guardrail applied (top-10 not all-survey; watch autonomic features co-equal with
  anthropometrics/FH).
- Site×label confound handled correctly: UTC→local mandatory before circadian FE, site
  never used as a feature. SpO₂ correctly demoted to Tier-3 (71.8% coverage).

**Limitations the team already owns (REPORT_A_WRAP.md §6):** features chosen from C1 val
importances → optimistic bias on the *same* test split (didn't invent a win, but worth
noting); re-HPO per subset confounds feature-count with HPO noise; fixed split → bootstrap
CIs are within-split, not external validity.

**Scope left (genuine):**
1. **CORN/CORAL neural primary — done / null (2026-07-16).** Package `training/path_a_raise_corn/`:
   CORN MLP test 4-AUC **0.706** vs C1 **0.738** (Δ−0.031, boot CI entirely &lt;0); CE-MLP control
   **0.713**; CORN≈CE. Class-2 soft win no. Primary bar null **justified** (`REPORT.md`).
   Historical note: only `statsmodels OrderedModel` existed as a diagnostic before this raise.
   Optional leftovers only: unweighted CORN ablation, CORAL sibling, impute sensitivity — not a
   re-open of the primary bar without a new `PLAN_*`.
2. **Calibration is "diagnostic only."** Brier and curves are reported, but raw ranking is
   the claim and no threshold-tuned screening point is selected. For a *screening* binary
   (healthy-vs-not, which is where the project's honest band — binary ~0.78–0.82 — is nearly
   hit at 0.831), a fixed-sensitivity operating point (e.g. 80% sensitivity at minimal
   specificity) is the deployable metric, and isotonic/Beta calibration + threshold moving
   can change the *usable* performance story without changing AUC at all.
3. **GBM stacker / ensemble — done / null (2026-07-16).** Package `training/path_a_raise_ensemble/`:
   multi-seed bags (S=5 + mandatory S=10) + arith/geom mean + σ-stacker on exact C1 matrix.
   Dual primary A=`Bag_cat` / B=`E_arith` both bar-fail vs C1 (best S=5 ΔAUC **+0.006**, CI includes 0;
   binary slightly down). Authority: `path_a_raise_ensemble/REPORT.md`. Do not re-open without a new
   `PLAN_*`.
4. **Fixed split, not nested CV.** Defensible ("stricter than k-fold averages"), but it
   suppresses the external-validity claim the paper wants. A nested-CV *companion* number
   (not replacing the fixed-split headline) would strengthen the "watch-only works" claim.

### 2.2 B1 — correct *after* the fix; pre-fix was genuinely buggy (and acknowledged)

**The pre-fix near-chance result was a real bug, not a method failure** — and the team
caught and fixed it (`AUDIT_B1_UNDERPERF.md`): (C1) sleep duration computed as int64/1e9 on
a `datetime64[ms]` dtype → ~1e6× too small → all nights collapsed to one day, ~8% non-null
sleep days, dead sleep channel; (C2) no input z-score for the BiLSTM → steps/sedentary
dominated at std-ratio ~1e8, CE flat near ln(4). Post-fix, the overfit gate passed
(CE 1.38→0.79 on 50 pids), confirming the backbone *can* learn. This is model methodology
hygiene done right — bugs found, fixed, invalid run kept labeled, valid run re-issued with a
new id. Good process.

**Post-fix null is authentic for the recipe run:** λ-weighted sum of CE + glu-MSE, λ∈{0,.5},
paired bootstrap Δ≈−0.0003 (CI lo≯0). GREEN late-fusion does not raise it (0.638 < 0.652).
Pure-seq 0.652 < tabular W0 0.666 — consistent with the literature prior (trees beat
cold-start RNNs at n≲2k on tabular/physio; Liao 2022, Kinfu 2021, TabZilla 2023).

**Scope left (genuine) — post B1 GS (2026-07-16):**
1. **Naive loss weighting — CLOSED on day spine.** `PLAN_B1_GS.md` / `REPORT_B1_GS.md` ran
   PCGrad + Kendall/Gal-style UW on the same backbone. Moderate conflict (~20% glu-active
   steps) was measured; class Δ still null (best +0.0006, CI lo≯0); glu head stayed
   sub-constant (val z-MSE~1.32). **Do not cite "GS never tried" for B1.** Residual MTL gap
   is B4-lane traj GS only (separate plan), not more B1 λ/GS grids.
2. **Cold-start sequence encoder with the weak-hand features.** B1 uses 18 daily watch
   dims and **omits** the person-level constructs that drive Path A (SRI, RAR amplitude/mesor/
   acrophase, onset SD, multi-day aggregates). The audit (C5) flags this as a design gap.
   Even after the late-fusion attempt, the *sequence* itself is information-poor relative to
   GREEN. The project's own identified lever — **SSL pretraining on the ~2,280 unlabeled
   wearable pool** — was never built.
3. **No ordinal loss.** Same CORN gap as Path A; a distance-aware / EMD loss is a documented
   lift for severity labels once the floor is learnable (which post-fix it is).
4. **Class weights** (C6) inverse-freq giving insulin ~0.5 mass — pinned, never re-tuned
   after the fix. Focal loss / class-balanced effective-sample weighting never tried on the
   neural side.

### 2.3 B2 — correct, clean parity, and the most instructive null

**Done right:** Stage-2 plumbing is *bit-exact* with frozen Path A (D0 = W0 = 0.6662, D1 =
C1 = 0.7378 to within 1e-5). OOF Stage-1 with non-aux train imputed as fold-mean
(leakage-safe). Decision bars pre-registered. Paired bootstrap. The matched-oracle (O1 vs
D1a on the *same aux pool*) is the right experimental design — **+0.094 4-AUC CI
[+0.058, +0.130]** is real privilege, not a pool artifact.

**The null is authentic for the recipe:** Stage-1 R²≈0.05 (val) / 0.015 (test) means the
emulator is near-null; predicted person-CGM is approximately *noise collinear with watch*,
and adding it slightly *harms* binary (Δbin −0.017, CI < 0). That's the expected signature
of predicted-feature error propagation — weak Ŷ injects collinear noise, not signal.

**Scope left at freeze time (genuine then):** residual knobs in `REPORT_B2` §7 (daily Stage-1,
variance propagation, reduced Y) were recommended "skip / defer."

**Update 2026-07-16 — B2-V2 attacked those knobs** (`b2v2_grid_20260716`, `REPORT_B2_V2.md`):
1. **Daily watch→CGM MSE mid** (val mean R² ~0.13 vs person GREEN ~0.05; test still ~0.03).
2. **Variance pack** mid+spread+daysd; Stage-2 still used ~18% yhat importance and **lost** to D1.
3. **Reduced collinear Y** {mean,sd,tir,tar}. T1v 0.727 / T1p 0.730 ≯ D1 0.738; O1−D1a +0.096.
Adversarial audit: null **authentic** (no blocker bug). Remaining open cells are **not** more
tabular two-stage HPO — SSL/deep Stage-1 or other new `PLAN_*` only if reopened.

### 2.4 B4-A — correct, but the *same* naive MTL recipe, on a richer target

**Done right:** Critiqued FE (`grid_5min` 6.88M bins × 1824, site-local ToD, concurrent-aux
mask, CGM-free subwindow selection = wear-density only). Overfit50 gate clean. Traj head is
*non-degenerate* (Pearson rises 0.145→0.255 with λ) so the teacher signal exists.
Pre-registered bars. D1 re-fitted on the exact surviving `pid_allow` (no freeze-fallback
cheat). This is the best-controlled cell in Path B.

**The null is authentic for the recipe:** Sλ ≯ S0 (Δ −0.010 / −0.008, lo≯0); hybrid
Sλ+C1 **loses** to D1 by 0.022–0.033, with S1.0+C1 CI entirely ≤0. So frozen-z from a
sequence encoder *dilutes* a strong tabular stack — the consistent pattern that an
information-poor neural embedding is collinear noise to GBM, not additive signal.

**Scope left (genuine):**
1. **Same gradient-conflict gap as B1.** λ-weighted traj MSE + CE, λ∈{0,.3,1}. No task
   balancing. Same critique applies, arguably more so because the two tasks are at very
   different SNR (glucose recon Pearson ~0.25 vs class AUC ~0.65) — exactly the regime
   where PCGrad/uncertainty weighting matters most.
2. **Frozen-z hybrid is the wrong fusion test.** Freezing the embedding and concatenenating
   to C1 for a GBM Stage-2 is known to underperform *fine-tuned* or *attention-fused*
   variants; the project's own methodology menu lists "cross-attention late fusion" as an
   unfilled cell. A learned-fusion (gated attention / FiLM / learned residual) over z and
   C1 where the gradient can *shape* z for the class objective was never run. Frozen z
   forces the GBM to find signal in an embedding optimized for trajectory recon, not for
   T2D ranking — a representational mismatch.
3. **Information-poor encoder without SSL.** PatchCNN h=64, patch12, cold start. The
   wear→curve Pearson ceiling (~0.26) is itself a statement that the encoder is
   data-hungry. SSL pretraining on the unlabeled wearable 5-min grid (in-mask
   reconstruction / masked-value MAE / contrastive over time-windows) is the documented
   way to raise this, and it is what the project identified as the gated lever and never
   built.
4. **No time-series augmentation** for the minority class (insulin train n=80). Listed as
   an open option; never run. Jitter/scaling/magnitude-warping/permutation is a cheap
   sequence-data lever specifically for the rare class.

### 2.5 B4-B — correct, easy-teacher loophole properly closed, but the *objective* is the weakest variant

**Done right:** The easy-teacher critique (X∥cgm input makes the teacher a near-copy of
CGM, Pearson ~0.99) was taken seriously and answered with **H1 (cgm-only) and H2
(wear→cgm hard map)** teachers to close the loophole — good adversarial process. Distill
scope (train∩aux only) and z_T not feeding val/test student loss are leakage-safe.
Cosine alignment *rises* with μ (0.09→0.48→0.64 easy; up to 0.75 hard) — so the
*mechanism* of distillation works; the *classification benefit* does not.

**The null is authentic for the recipe:** μ-MSE distillation. μ=0.3 *hurts* (CI<0 or
borderline) across easy/H1/H2; μ=1 neutral; hybrid recovers toward D1 as μ rises but
never beats it. This is a *real null for L2 representation distillation*.

**Scope left (genuine — and this is the most citable gap):**
1. **L2 / MSE on raw embeddings is the *weakest* distillation objective in the literature.**
   The KD literature is explicit and consistent: logit-KD (KL on soft targets), CRD
   (contrastive over multiple views + memory bank), RKD (relational — pair-wise distances
   + angles), AT (attention transfer), FitNet (hint-based intermediate supervision), and
   PKT / variational KD all *routinely outperform* naïve L2 feature regression on small
   datasets, because L2 optimizes for *reconstruction* of z, not for preserving the
   *task-discriminative* structure. The cosine metric in the report already *measures*
   alignment that L2 isn't directly optimizing for; switching the distill term to a
   contrastive (CRD/InfoNCE over augmented time-windows) or relational (RKD) objective is
   the single best-evidenced way to convert "student matches teacher z" into "student
   classifies better."
2. **B3 (logit-KD) is still unrun.** The teammate's own method (T=2, α=0.3, recovered ~20%
   of the teacher gap) is the natural complement and was deliberately deferred to "last."
   Logit-KD often beats feature-KD when the teacher's *soft class probabilities* carry
   dark-knowledge structure that point-estimate regression (the B2 failure) and L2-z
   (B4-B) can't capture. Running B3 is the cheapest way to *complete* the LUPI-KD
   comparison the paper needs; not running it leaves the headline "no LUPI formulation
   beats C1" technically unsupported (B4-B tested feature-distill; B3 *is* logit-distill).
3. **No CRD / contrastive distill means no use of the unlabeled non-aux participants as a
   "negative view" bank** — the cheapest large-n resource in the dataset, sitting unused.

---

## 3. Web research synthesis — best practices for each technique used

| Technique (as used) | Best-practice finding (2024–25 literature) | Implication for this project |
|---|---|---|
| **LightGBM/CatBoost at n≈1,800, multiclass, imbalance** | Native NaN + native categoricals are correct; `is_unbalance`/class-weight `balanced` is the *minimum* — class-balanced / focal / effective-sample variants often beat it; **50 Optuna trials is light** for a 47-dim space — aggressive search (200–500 + multi-fidelity Optuna/HEBO) frequently finds +0.5–1.5pp on this n; **nested CV** is the honest small-n companion to a fixed split. | Path A HPO is undersized; a stacker + 5–10× trials + nested-CV companion is a low-cost, real lift. |
| **Calibration** | For screening, the operating point (sensitivity at fixed specificity) is the deployable metric; isotonic/Beta on val + threshold-moving routinely produces usable gains AUC-blindly. Brier as-is underestimates the value of temperature scaling on multiclass. | Apply calibration + threshold selection to the *binary screening* claim; reframe binary as "at 80% sensitivity, specificity = …". |
| **MTL using plain λ-weighted losses** | **Plain-λ is the weakest MTL formulation.** Conflicting / imbalanced-gradient regimes (one head much weaker, as here: glucose recon R²≈0.05 vs class AUC 0.65) are exactly where PCGrad / CAGrad / GradNorm / uncertainty weighting / gradient-vaccine flip nulls to positives. | **B1 GS done** (`REPORT_B1_GS.md`) — still null; residual is **B4-A/B4-lane GS** only. |
| **CORN/CORAL ordinal** | CORN drops the proportional-odds assumption (suspected to fail here at the pre-diabetes/oral boundary); rank-consistent; integrates with neural backbones; works with imbalanced labels when paired with weighted CORN. The class-2 bottleneck is precisely the boundary CORN is designed for. | Train CORN on the post-fix B1/B4 encoder (and on Path A's neural ordinal variant). Genuine untried lever. |
| **Two-stage predicted-feature stacking** | Predicted features without variance propagation inject collinear noise (textbook). Fixes: pass predictive intervals (quantile LGBM / NGBoost / conformal) or full Bayesian model averaging; or use Stage-1 *residuals* + OOF features as the Stage-2 input. | B2's "Ŷ adds noise" is the expected failure mode — fixable, not fundamental. |
| **Representation distillation (L2/MSE)** | L2 feature regression is the *weakest* KD objective. CRD (contrastive), RKD (relational), FitNet, PKT, and logit-KD (B3) all *routinely* beat it on small data because they preserve task-discriminative structure, not reconstruction. | B4-B tested L2; **B3 logit-KD since concluded null** (`REPORT_B3.md`). Residual stronger variants: **CRD/RKD** (B4-lane), not more L2. |
| **Self-supervised pretraining (SimCLR/BYOL/MAE) on unlabeled wearables** | 2023–25 wearable/PPG/ECG literature: SSL on the unlabeled pool *materially* raises few-label downstream performance and is the standard answer to cold-start sequence underperformance at clinical n. In-mask/masked-value reconstruction is particularly suited to the modality-dropout structure of this dataset. | The biggest *unbuilt* lever the project itself identified. ~2,280 unlabeled participants; SSL backbone → B1/B4/Stage-1 is the coherent retry path. |
| **MOMENT / time-series foundation models** | MOMENT supports classification via frozen embedding → GBM, or fine-tuned head; transfer to physiological signals is partial and high-variance but cheap to probe ("one-afternoon side bet," per project). Chronos/TimesFM/Moirai are forecasting-first, less useful here. | Probing MOMENT frozen-embedding → GBM is a single afternoon and a genuine side bet the project deferred. |
| **Cross-attention late fusion** | Per-modality encoders + attention fusion > concat-then-Linear when modalities have unequal missingness (HR dense, SpO₂ 71.8%, sleep interval-based, RR ~2× HR) — which is exactly this dataset. | "Attention-fusion ablation" is listed as an unfilled architectural cell in `Training.md` and was never run. |

---

## 4. Was "nothing beat C1" the expected outcome?

Partly. Three independent priors stacked against Path B:
1. **Trees beat cold-start DL at n≲2k** on tabular/physio (Tabzilla, Liao 2022, Kinfu 2021).
   B1's own audit cites this. A cold-start 5-min CNN at n≈1.8k *beating* a 47-feature
   CatBoost tuned to 0.738 was always the low-probability outcome. The honest target was
   "beat the matched tabular baseline *with the LUPI signal*," not "beat it cold."
2. **Watch→glucose SNR is low.** B2 Stage-1 R²≈0.05 and B4-A wear→curve Pearson ~0.25
   independently confirm: the wearable signal carries only weak glycemic structure. When
   the auxiliary target is near-unlearnable, *any* LUPI handoff that depends on predicting
   that target will struggle — the oracle's +9pp is gated behind a Stage-1 the data can't
   support from GREEN summaries.
3. **Several formulations started as the weakest in their family** (plain-λ MTL, point-estimate
   stack, L2 distill, cold-start encoder). **B1 GS** and **B3 logit-KD** have since closed two
   of those gaps without a deployable raise — so the null is no longer "only naive λ / only
   feature-distill."

So: the nulls are **not surprising** and are **trustworthy**. Stronger B4 recipes (RKD/CRD,
PCGrad, OOF fusion) have now also been tried and null (`REPORT_B4_V2.md`). The legitimate claim
today: *"plain-λ and gradient-balanced day-level multi-task, point-estimate two-stage, L2 and
RKD/CRD rep-distill, PCGrad traj MTL, OOF fusion, and logit-KD do not beat a tuned
watch+onboarding+mood CatBoost on this cohort — privilege is real (oracle), the bottleneck is
the wear→glucose SNR / cold representation, and SSL (plus Path A polish) remain open."*

---

## 5. Prioritized recommendations (scope for improvement, best-evidenced first)

Ranked by (expected lift) × (effort) × (evidence strength), and honest about which would
re-open a frozen cell (requires a new `PLAN_*` per the project's own locks).

### Tier 1 — strongest evidence, modest effort, opens a `PLAN_*`

1. **B3 logit-KD — DONE / null** (`REPORT_B3.md`, 2026-07-15). Completes the feature- vs
   logit-distill comparison under the tested recipes; do not re-list as unrun.
2. **CRD / RKD + PCGrad + OOF (B4-V2) — DONE / null** (`REPORT_B4_V2.md`, 2026-07-16).
   Teacher H2 GO (Pearson ~0.30); student μ&gt;0 null/hurts; PCGrad no-conflict + hurts class;
   best hybrid F1 OOF 0.726 ≯ D1 0.736. Do **not** re-list as unrun or reopen without a new plan.
3. **PCGrad / UW MTL — B1 DONE / null** (`REPORT_B1_GS.md`); **B4-lane PCGrad also DONE / null**
   inside B4-V2 (no conflict on PatchCNN). Do not reopen B1 day-spine λ/GS or B4-V2 grids.

### Tier 2 — real but bigger lift; the project's own gated-never-built backbone

4. **SSL-pretrained sequence backbone** (masked-value MAE + SimCLR/BYOL over time-windows)
   on the ~2,280 unlabeled wearable 5-min grid, then fine-tune for (a) the B4 sequence head,
   (b) a better B2 Stage-1 emulator, (c) MOMENT-embedding→GBM as the one-afternoon probe.
   This is the coherent direction that attacks all three Path-B failure modes at once:
   cold-start underperformance, weak Stage-1, and the wear→glucose SNR ceiling. *Biggest
   unbuilt lever the project already named.* New `PLAN_SSL.md`.

### Tier 3 — Path A polish (does not re-open B cells; complements the paper)

5. **CORN neural ordinal on Path A tabular — executed / null (2026-07-16).** See
   `training/path_a_raise_corn/REPORT.md`. Does not raise C1; most macro gap shared with CE-MLP
   (MLP+impute story). Optional: unweighted weighting ablation / CORAL sibling only if a new go.
6. **GBM multi-seed bag + cross-family ensemble — executed / null (2026-07-16).** See
   `training/path_a_raise_ensemble/REPORT.md`. S=5 best ΔAUC +0.006 (CI includes 0); S=10 no help.
   Seed bag ≈ C1; blend/stack do not clear +0.01 bar. Closed without a new `PLAN_*`.
7. **Calibration + fixed-sensitivity operating point for the binary screening claim.**
   AUC-blind but deployable-meaningful. Reframes binary 0.831 as "at X% sensitivity,
   Y% specificity" — the actual screening metric. Low effort.
8. **Nested-CV companion number** (and optional re-HPO / trial-count sensitivity) for external
   validity of the fixed-split headline. Low effort; does not re-open ensemble primary.

### Tier 4 — lower-priority / higher-risk

9. **Cross-attention late fusion** over per-modality encoders (unfilled `Training.md` cell).
   Architecturally motivated by unequal missingness; real but the SSL backbone probably
   subsumes it.
9. **Stage-1 with predictive variance / Bayesian stacking** for B2 — principled but ~null
   prior to beat, given Stage-1 R²≈0.05; only worth it *after* the Stage-1 emulator is
   improved (Tier 2 item 2).
10. **Time-series augmentation** for insulin (train n=80) — cheap, targets the rare class
    specifically; best paired with the SSL backbone.

### Explicitly *not* recommended

- Do **not** reopen B1 λ grids, B2 HPO, or B4-A trajectory λ grids at the *same* recipe —
  the nulls are trustworthy for those recipes and the project's own locks forbid it. Any
  retry must be a *different formulation* under a new `PLAN_*`.
- Do **not** expand Path A's feature set on the same test split without nested CV — the
  within-split optimization bias is already a stated limitation.
- Do **not** chase the binary 0.831 → 0.84 delta by HPO; the dedicated binary HPO already
  lost to multiclass-derived `1−P0`. The win there is calibration/thresholds, not AUC.

---

## 6. Bottom line

- **The phases were done correctly** under pre-registered protocols; the post-fix B1 bug-fix
  in particular is good methodology (found, fixed, re-issued, invalid run kept labeled).
  Path A's leakage discipline and parity control are the strongest parts.
- **The nulls are real for the formulations run** (plain-λ MTL, point-estimate two-stage,
  L2-MSE distillation, cold-start sequence encoders on a weak-watch→glucose channel). They
  are not artifacts of bugs, leakage, or HPO noise — verified by parity checks (D1≡C1),
  overfit gates, paired bootstraps, and easy+hard teacher sensitivities.
- **But "nothing beat C1" is not proof every representation path is exhausted.** B1 GS, B3,
  and **B4-V2** (RKD/CRD + PCGrad + OOF) closed the main "naive-only" LUPI gaps without a
  deployable raise. Still open (new `PLAN_*` only): **SSL-pretrained backbones**, CORN,
  calibration-thresholded screening, optional Path A polish. The supportable claim is a
  **recipe-scoped** negative across the tested LUPI/KD/MTL family; "SSL exhausted" is not
  justified.
- The **most defensible next steps**, in order: **SSL backbone** (project's own gated lever);
  Path-A CORN + calibration + optional GBM stacker. Do **not** reopen B4-V2 KD/MTL grids.