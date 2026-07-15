# Path B3 — logit knowledge distillation (Diasense-style baseline)

**Date:** 2026-07-15  
**Status:** **IMPLEMENTED & CONCLUDED** 2026-07-15 — claim run `b3_grid_20260715`; see `REPORT_B3.md`.  
**Role:** Strong **baseline to beat / complete**, not paper novelty. Completes the LUPI-KD comparison after B4-B **feature**-distill null (logit-KD ≠ L2-z).  
**Authority:** `Training.md` §2 metrics / §4 B3 / §7; Path A freeze `REPORT_A_WRAP.md`; B2 freeze `REPORT_B2.md` (oracle + OOF pattern); B4-B `REPORT_B4_B*.md` (rep-distill null); `FEATURES.md` §5 Diasense; `DECISIONS.md`.  
**User ambition bar:** beat Path A **C1** (test 4-AUC **0.7378** / binary **0.8309** / macro AUPRC **0.4687**) via the **pre-registered** arm `G_α=0.3` only (not max-over-grid).  
**Ladder:** B1 frozen → B2 frozen → B4 A+B+hard concluded → **B3 last**.

**Critique:** fresh `critiquer` (`opencode-go/glm-5.2:high`) → **revise** → disposition applied (§8).

---

## 0. Data readiness verdict

### Verdict
**No re-clean. No new FE required for B3 v1.** Existing processed assets are sufficient (same as B2).

| Asset | Status | B3 use |
|---|---|---|
| `features/watch_green.parquet` | ready 1824×31; 0 nulls | W0 block; C1 base |
| `features/onboarding.parquet` / `mood.parquet` | ready; Path A C1 stack | student + teacher deployable base |
| `features/cgm_person.parquet` | ready 1924×12; **aux 1685/1685** with valid daymeans | **teacher-only** true CGM 8-vector |
| `meta/pool_masks.parquet` | ready; core 1277/270/277; aux 1184/247/254 | labels, splits, pools |
| `features/cgm_daily.parquet` / `watch_daily.parquet` / `grid_5min*` | ready | **not primary for B3 v1** (logit-KD is person-tabular; sequence KD = different cell) |

### Why no pipeline change
- Soft class logits need **person-level X + label + privileged CGM features** — all present.
- Teacher privilege = **C1 + true person CGM 8-vector** (same feature cell as B2 O1). B2 already showed oracle headroom **O1−D1a ≈ +0.094** 4-AUC on aux — teacher signal is real.
- Sleep-unit FE bug was watch_daily only (fixed); `watch_green` / C1 path untouched — Path A numbers stay frozen.
- B4 5-min grid is for trajectory/rep-distill (done); **not** required for person-level logit-KD.
- Weak watch→glucose R² (B2 Stage-1) is **not** a B3 blocker: teacher sees **true** CGM, not predicted glucose. Student never sees CGM.

### Empirical anchors (prior runs; not re-claimed)
| Probe | test 4-AUC | notes |
|---|---:|---|
| Frozen C1 / B2 D1 | **0.7378** | ambition external + matched target |
| B2 O1 (C1 + true CGM, **aux-matched**) | **0.8227** | privileged ceiling; O1−**D1a** ≈ **+0.094** (not vs full-core D1) |
| B2 T1 (C1 + predicted CGM) | 0.7345 | point-estimate handoff null |
| B4-B best hybrid | ≤0.735 | L2 rep-distill null / hurts |

**Implication:** Logit-KD can still work where B2/B4-B failed if **dark knowledge in teacher class soft labels** transfers better than CGM point estimates or z-MSE. Prior is mixed (Diasense claimed ~20% gap recovery; our trees+C1 already strong). A clean null is still an informative baseline close-out.

---

## 1. Goal

### Scientific question
Does **logit knowledge distillation** from a CGM-privileged teacher (soft class probabilities) improve a **deployable** student (no CGM at train features / infer) over a **matched hard-label** baseline on the same student feature stack?

### Claims B3 may make
1. **User-ambition KD:** pre-registered `G_α=0.3` (LightGBM soft-expansion, T=2) vs **matched D1** (ΔAUC + paired boot CI) on full `wearable_core`.
2. **KD science (exact Hinton):** `N_α=0.3` vs **N0** on full core (architecture-controlled; not the beat-C1 bar).
3. **Teacher ceiling:** privileged teacher on **aux-matched** pool vs D1a (sanity; should echo B2 O1−D1a).
4. **Family completeness:** logit-KD cell closed relative to B4-B feature-distill (different objective).

### Claims B3 is not
- Multi-task (B1), two-stage Ŷ-glu (B2), traj multi-task / rep-distill (B4) — do not reopen without new `PLAN_*`.
- A novelty claim for plain Diasense redo (document as baseline).
- Re-open of Path A survey blocks / C1 sensitivities / re-clean.
- License to change Path A frozen numbers.
- “LUPI works” from teacher AUC alone.
- Sequence / 5-min logit-KD (out of scope v1).
- Max-over-α / max-over-family “beat C1” without multiplicity control.

---

## 2. Design locks

### 2.1 Pipeline shape

```
Teacher (privileged; fit train ∩ aux only):
  X_T = C1 ∪ Y_glu_true          # 47 + 8; never at student infer
  Y   = label ∈ {0,1,2,3}
  → f_T → soft probs p_T (temperature T)

OOF soft labels (leakage lock):
  K=5 stratified (by label) on train ∩ aux
  → p_T_oof[i] for each aux-train person
  Full teacher fit on all train∩aux → p_T for val/test diagnostics only
  (student train never uses val/test teacher fit on that person's split leakage)

Student (deployable; full wearable_core):
  X_S = C1                       # primary (no W0 student arms in v1)
  Neural: L = (1-α) * CE(y, p_S) + α * T² * KL( p_T^(T) || p_S^(T) )
      # soft KL only on rows with OOF teacher soft labels (train∩aux)
      # non-aux train: hard CE only
  GBM: soft-row expansion (§2.5) approximating soft CE (no T² term)

Deployable inference: p_S = f_S(C1). Never CGM. Never teacher network.
```

**Temperature:** for tree teachers that expose only probabilities, form  
\(p^{(T)}_k = p_k^{1/T} / \sum_j p_j^{1/T}\)  
(equivalently `softmax(log p / T)`). If a neural teacher exposes logits, use `softmax(z/T)`.

### 2.2 Feature blocks

| Block | Columns | Role |
|---|---|---|
| **W0** | 30 GREEN numeric | part of C1 only in v1 (no W0-only student grid) |
| **C1** | W0 + Path A `onboarding_keep` (15) + `cestl` + `paidscore` | **primary** student X; teacher base |
| **Y_glu** | 8 `cgm_*_daymean` from `cgm_person` | **teacher only** |
| **Ŷ_glu** | predicted CGM | **out of scope** (B2 cell) |

**C1 manifest lock:** load keep-lists from `training/path_a_blocks/config.yaml` at runtime **and** snapshot resolved column list into `b3/artifacts/<run_id>/c1_feature_manifest.json`. Assert `n_feat=47` and names match frozen wrap (`feature_hash` optional cross-check vs `mood_scores_20260714_014415/features.json`).

**Forbidden in any X:** `label`, `recommended_split`, `clinical_site`, pool flags, `n_valid_days`, `cgm_n_total`, `n_days`, site, study_group.

### 2.3 Leakage rules (hard)

1. **Teacher never fit on val/test persons.**
2. **Student train soft targets** = **OOF** teacher proba on train∩aux only (K=5, seed 42, stratify by label).
3. **Non-aux train persons** (core \ aux, n≈93 train): **hard CE only** — no teacher soft labels. Do **not** median-fill CGM into non-aux to manufacture soft labels.
4. Val/test evaluation of student: **no teacher inputs**. Teacher metrics on val/test aux are diagnostic only.
5. Outer claim split = fixed `recommended_split` only.
6. Do not restrict deployable student train/eval pool to aux-only (full core 1824). Teacher fit pool = aux.

### 2.4 Models

| Role | Primary | Notes |
|---|---|---|
| **Teacher** | Path A family **CatBoost + LightGBM** multiclass on C1+Y_glu | Same HPO spaces / class-weight spelling as `path_a_blocks` / B2 Stage-2. Val-select macro-OVR AUC → AUPRC tie. |
| **Student neural (KD science only)** | **MLP** on C1 (train-only median impute + z-score) | Exact Hinton KL. 2 hidden layers; **h=64 primary**; dropout 0.1; Adam; early stop on val hard macro-AUC. **Not** in user-ambition bar. |
| **Student GBM (user-ambition primary)** | **LightGBM** soft-expansion on C1 | Pre-registered primary Gα family. Soft labels via §2.5. α=0 → G0 protocol twin of D1-LGBM. |
| **Student GBM (sensitivity)** | CatBoost soft-expansion | Footnote only if cheap; **not** ambition decision arm (avoids weight × `Balanced` ambiguity). |
| **D1 matched baseline** | Path A family on C1 hard labels (CatBoost+LGBM val-select as Path A) | Must reproduce freeze within drift tol (same as B2). |

**Class weights (exact Path A spelling):**
- LightGBM hard CE: `class_weight="balanced"`.
- LightGBM soft-expansion: per-row `weight` = expansion mass × inverse-freq weight of the person's **hard** label; class weights computed on train-core hard labels only (lock).
- CatBoost hard fits (D1/Tch/D1a): `auto_class_weights="Balanced"`.
- CatBoost soft-expansion: sensitivity only; use per-row `weight` **without** stacking another Balanced transform if the library double-counts — document in run log.

**HPO:**
- Teacher: ~50 trials/family, seed 42 (smoke: 5).
- Student LightGBM Gα: ~50 trials under soft-expansion at each α (smoke: 5). **Exception:** G0 does **not** re-HPO — inherits D1 LightGBM selected params (§2.9).
- Student MLP: light grid / small Optuna on lr / weight_decay; smoke fixed; wall-time cap → freeze smoke-selected params (B2-style fallback).
- Reuse HPO search spaces from `training/path_a_blocks/config.yaml`.

**Do not** use B1 BiLSTM / B4 CNN as primary B3 student (different cells; out of v1).

### 2.5 Soft-label training for GBM (lock)

Tree models lack native KL soft-target multiclass in our stack. **Locked approximation** (standard soft-label expansion):

For each train person \(i\) with hard label \(y_i\) and (if aux) OOF teacher soft \(p_{T,i}\):

\[
w_{i,k} =
\begin{cases}
(1-\alpha)\, \mathbf{1}[k=y_i] + \alpha\, p^{(T)}_{T,i,k} & i \in \text{train∩aux} \\
\mathbf{1}[k=y_i] & i \in \text{train non-aux}
\end{cases}
\]

- Emit up to 4 virtual rows per person with `label=k` and `weight=w_{i,k}` for \(w_{i,k} > \epsilon\) (ε=1e-6).
- Train multiclass GBM with those sample weights (hard CE on expanded set ≈ soft CE; **no T²** term — incommensurate with Hinton α).
- **α=0:** single hard row per person, weight 1 on \(y_i\) — G0 protocol with **pinned** D1-LGBM params.
- **α=1:** pure soft targets on aux; non-aux still hard.

**Neural student** uses exact Hinton form with KL on temperature-scaled distributions; no row expansion.

### 2.6 Hyperparameters (pre-registered)

| Knob | Primary | Grid / sensitivity |
|---|---|---|
| Temperature **T** | **2** (Diasense) | T∈{1,4} **only if** §3.4 trigger fires |
| Mix **α** | **0.3** (Diasense) | Science table **{0, 0.3, 0.5, 1.0}** at T=2 |
| OOF K | **5** | fixed |
| Seed | **42** | fixed |
| Bootstrap | n=**2000**, seed 42 | fixed |

**Pre-registered Diasense decision point:** (T=2, **α=0.3**).
- **User-ambition decision arm:** `G_α=0.3` (LightGBM) vs **D1** only — **not** max-over-grid.
- **KD-science decision arm:** `N_α=0.3` vs **N0** only.
- Other α cells: report point + CI as **sensitivity**; descriptive “best deployable” table row is **non-claim** (no beat-C1 from post-hoc best without multiplicity note).
- α is **not commensurate** across Gα (row-expansion soft CE, no T²) and Nα (Hinton with T²) — never pool them into one “α effect.”

### 2.7 Arms (pre-registered)

| ID | Pool | Features | Train signal | Deployable? | Role |
|---|---|---|---|---|---|
| **D1** | full core | C1 | hard Path A family | yes | **matched C1 direct** (ambition baseline) |
| **Tch** | **aux-only** | C1+Y_glu | hard Path A family | **no** | teacher ceiling (expect ~O1 on aux) |
| **D1a** | **aux-only** | C1 | hard Path A family | yes* | matched teacher baseline (Tch−D1a) |
| **G0** | full core | C1 | LGBM soft-exp α=0, **D1-LGBM params pinned** | yes | expansion plumbing protocol check |
| **G_α=0.3** | full core | C1 | LGBM soft-exp α=0.3, T=2 | yes | **user-ambition decision arm** |
| **G_α∈{0.5,1.0}** | full core | C1 | LGBM soft-exp | yes | sensitivity only (non-claim max) |
| **N0** | full core | C1 | MLP hard α=0 | yes | neural control (science only) |
| **N_α=0.3** | full core | C1 | MLP Hinton α=0.3, T=2 | yes | **KD-science decision arm** |
| **N_α∈{0.5,1.0}** | full core | C1 | MLP Hinton | yes | sensitivity only |

\*D1a is protocol baseline for teacher matching, not the user-bar deployable story.

**No W0 student arms in v1** (baseline close-out scope). Optional single-point footnote only if user requests after primary null — not in claim grid.

**Primary comparisons (pre-registered)**
1. **User ambition:** `G_α=0.3` vs **D1** (full core); frozen C1 external anchor.  
2. **KD science (exact Hinton):** `N_α=0.3` vs **N0**.  
3. **Teacher headroom:** **Tch vs D1a** (aux-matched; sanity vs B2 O1−D1a).  
4. **Protocol:** **G0 vs D1-LGBM** (pinned params).  
5. Report teacher calibration + soft-label sharpness (entropy, max-prob, Brier/ECE on val aux).

**Narrative lock if outcome cells disagree**

| G_α=0.3 vs D1 | N_α=0.3 vs N0 | Read as |
|---|---|---|
| raise | raise | logit-KD helps (both approx + exact) |
| raise | null | soft-expansion regularisation / family effect — **not** pure logit-KD; report approximation caveat |
| null | raise | Hinton works on weak MLP but no deployable tree raise |
| null | null | logit-KD cell closed null under this recipe |

**Dropped from primary:** W0-KD grids; sequence teacher/student; true-CGM on student; median-filled non-aux soft labels; Ŷ_glu stacking (B2); max-over-α ambition claims.

### 2.8 Metrics (Training.md §2 aligned)

| Metric | Use |
|---|---|
| Macro-OVR **4-AUC** | primary rank |
| Macro **AUPRC** | co-required |
| Binary AUC (`1−P0`) + binary AUPRC | screening report |
| Per-class OVR AUC | diagnostic |
| **Brier** multiclass/binary | required with AUC |
| **Calibration** | student: val **sigmoid** primary; isotonic if min_pos≥30; ranking claim on **raw**. Teacher: val-aux reliability + Brier/ECE on **raw** proba (pre-temperature) |
| Teacher val/test 4-AUC (aux) | Tch ceiling |
| **OOF teacher** mean±sd val 4-AUC across K folds | KD-signal quality (student sees OOF, not full Tch) |
| Soft-label diagnostics | entropy, max-prob by split; OOF fold class counts |
| Paired person bootstrap Δ (n=2000, seed 42) | **G_α=0.3−D1**, **N_α=0.3−N0**, Tch−D1a (sensitivities reported, non-claim) |

### 2.9 Decision bars

| Question | Pass rule |
|---|---|
| **User ambition: beat C1** | **Decision arm only:** `G_α=0.3` (T=2, LGBM) vs **D1**. Pass if test Δ4-AUC **> 0** and paired boot CI **lo > 0**. Soft note if point > +0.005 but CI includes 0. **External anchor:** frozen C1 0.7378 / 0.8309. **Fallback:** if `D1 < frozen_C1 − 0.01`, fair bar = vs D1 only; diagnose parity. **Nα never enters this bar.** |
| KD science (Hinton) | `N_α=0.3`−N0: same CI lo>0 rule. Absolute Nα vs C1 is **informational only**. |
| Teacher headroom | Tch−D1a point Δ4-AUC **≥ +0.02** (expect pass; aligns B2). |
| Kill / weak full teacher | Tch−D1a **&lt; +0.01** → ceiling dead; still write REPORT. |
| **OOF teacher usable** | mean OOF-fold val 4-AUC **> D1a val + 0.01**. If fail: report “OOF teacher signal too weak at this n” — do not read student null as pure logit-KD failure. |
| **G0 protocol** | G0 uses **pinned D1 LightGBM hyperparams** (no re-HPO). Assert expansion emits exactly `n_train` persons with per-person weight sum **= 1** and only hard class non-zero at α=0. Metric: \|G0 − D1_LGBM\| test 4-AUC **≤ 1e-3** (numerical noise). Fail = plumbing bug → stop before Gα claims. |
| Smoke gate | Tch val > D1a val; OOF folds non-empty insulin; G0 row-count assert; student forward OK. |

B3 does **not** reopen B1/B2/B4 claim grids regardless of pass/fail.

---

## 3. Protocol details

### 3.1 Cohorts (verified 2026-07-15)

| Pool | train | val | test | notes |
|---|---:|---:|---:|---|
| wearable_core | 1277 | 270 | 277 | student deployable |
| core ∩ aux_eligible | 1184 | 247 | 254 | teacher fit / OOF / Tch |
| core \ aux | 93 | 23 | 23 | student hard-only rows |

Train insulin core **80**; aux-train insulin ~69 (OOF fold ~13–14).

### 3.2 Diagnostics (required)
1. C1 manifest assert (47 cols) + optional feature_hash vs Path A freeze.  
2. D1 vs frozen C1 absolute Δ4-AUC (drift tol 0.01).  
3. G0 pinned-params protocol (§2.9) + expansion row/weight asserts.  
4. OOF fold label counts (insulin non-empty).  
5. **OOF teacher** mean±sd val 4-AUC vs full Tch val (gate §2.9).  
6. Teacher soft-label entropy / max-prob by split; teacher val-aux Brier/ECE (raw).  
7. Non-aux train: confirm **zero** soft-weight mass from teacher.  
8. Leakage scan: student feature cols ∩ forbid = ∅; no CGM cols in student X.  
9. Report limitation (one line): Gα vs G0 does not separate dark knowledge from generic soft-label regularisation; shuffled-soft control only if headline raise and user requests.

### 3.3 Package layout (implement when approved)

```
training/path_b/b3/
  config.yaml
  data.py          # C1 load, glu teacher cols, OOF soft labels, expansion
  teacher.py       # Path A family teacher fit + predict_proba + temperature
  student_gbm.py   # soft-expansion LightGBM student (+ optional CatBoost sens)
  student_mlp.py   # Hinton MLP student
  evaluate.py      # metrics, bars, paired bootstrap (reuse path_a / b2 helpers)
  run.py / __main__.py
  artifacts/<run_id>/
```

Reuse `path_a_watch` / `path_a_blocks` / `path_b/b2` metrics, HPO, `load_watch_onboarding_mood`, `paired_delta_bootstrap` — **no drive-by refactors**.

### 3.4 Run ladder
1. **Smoke** `b3_smoke_<date>`: max_train subset, 5 HPO trials, T=2; arms D1, D1a, Tch, G0, `G_α=0.3`, N0, `N_α=0.3`.  
2. **Teacher full** `b3_teacher_<date>`: OOF soft labels + full teacher + Tch/D1a + OOF-quality gate.  
3. **Claim grid** `b3_grid_<date>`: D1, D1a, Tch, G0, Gα∈{0.3,0.5,1.0}, N0, Nα∈{0.3,0.5,1.0} at T=2. Decision claims only from `G_α=0.3` and `N_α=0.3`.  
4. **T sensitivity** only if: (`G_α=0.3` point Δ4-AUC > 0 but CI includes 0) **OR** (Tch−D1a > +0.04 **and** `G_α=0.3` null at T=2). Then T∈{1,4} at α=0.3 only.  
5. `REPORT_B3.md` + `DECISIONS.md` + `path_b/README.md` (+ root README/AGENTS status lines).

### 3.5 Seeds & compute
- Global seed 42; torch/numpy/cuda deterministic flags where cheap.  
- GBM CPU OK (Path A practice); MLP CPU or ROCm CUDA if available (`HIP_VISIBLE_DEVICES`).  
- No Path A re-run required for claim; D1 is internal re-fit.

---

## 4. Out of scope (B3 v1)

- Re-clean / re-FE GREEN, daily, or 5-min grid  
- B1 λ grids, B2 Ŷ-glu Stage-1, B4 traj/rep-distill reopen  
- CRD / RKD / FitNet / attention-transfer (possible later `PLAN_B4_CRD` — not B3)  
- SSL pretraining, MOMENT, CORN  
- Survey blocks beyond C1  
- Sequence teacher/student as primary  
- W0-only student KD grid  
- Changing Path A frozen numbers  
- Deployable claims from teacher or aux-only student  
- Beat-C1 claims from max-over-α without multiplicity control

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Soft expansion ≠ true KL | Nα = exact science cell; Gα = ambition approximation; narrative lock §2.7 |
| α=0 GBM ≠ D1 | **Pinned** D1-LGBM params + 1e-3 bar + row asserts |
| Multi-arm max-over-α false raise | User bar **only** `G_α=0.3`; other α non-claim |
| D1 ≠ frozen C1 | Manifest + HPO pin; §2.9 fallback |
| Full Tch strong, OOF weak | OOF mean val gate §2.9 |
| MLP ≪ trees | Nα never in beat-C1 bar; science only |
| Non-aux without soft labels | Hard CE only (honest full-core) |
| CatBoost × sample weights | **LGBM primary** for Gα; CatBoost soft = sensitivity |
| Over-claim Diasense match | Different n/split/features; controlled baseline |
| User bar too high | Ambition ≠ science pass; null still closes cell |
| Soft reg vs dark knowledge | Report limitation; shuffled-soft only if raise + user ask |

---

## 6. Relation to prior Path B nulls

| Cell | Result | B3 difference |
|---|---|---|
| B1 multi-task day-CGM | null | different objective (joint heads ≠ soft logits) |
| B2 predicted Ŷ_glu | null | distills **class soft labels**, not glucose point estimates |
| B4-A traj multi-task | null | person tabular KD, not 5-min recon |
| B4-B L2 z-distill | null / hurt | **logit** KL / soft CE, not representation MSE |

If B3 is also null, the honest paper claim becomes: *under this cohort and C1 stack, naive LUPI handoffs (scalar MTL, Ŷ-glu, traj MTL, L2-z, logit-KD) do not beat tuned C1; privilege is real (oracle/teacher).* Stronger recipes (CRD/SSL/PCGrad) stay explicit future work — not silent reopen.

---

## 7. Implementation gate

**Done.** Smoke `b3_smoke_20260715` + full `b3_grid_20260715` + T-sens + N-fix. Outcome: deployable logit-KD **null** vs C1; teacher headroom **pass**; Path B ladder complete.

---

## 8. Critique disposition (2026-07-15)

Source: fresh `critiquer` (`opencode-go/glm-5.2:high`) on pre-implement `PLAN_B3.md`. Verdict was **revise**.

| # | Critique item | Disposition | Action in plan |
|---|---|---|---|
| B1 | User bar = max-over-grid without multiplicity | **Accept (blocker)** | Pin ambition decision to **`G_α=0.3` vs D1** only; other α descriptive/non-claim |
| B2 | G0 \|Δ\|≤0.01 hides expansion bugs under HPO noise | **Accept (blocker)** | G0 **pins D1-LGBM hyperparams**; \|Δ\|≤**1e-3** + row/weight asserts |
| H1 | OOF teacher quality ungated | **Accept (high)** | Mean OOF val AUC gate vs D1a+0.01; report mean±sd |
| H2 | Dual “primary” Gα/Nα + incommensurate α | **Accept (high)** | Gα = ambition; Nα = Hinton science; narrative 2×2 lock; α not pooled |
| H3 | Best-deployable multi-arm | **Accept** | Collapses with B1 |
| M1 | Soft reg vs dark knowledge | **Accept (report)** | One-line limitation; shuffled-soft only if raise + user ask |
| M2 | CatBoost weight interaction unresolved | **Accept** | **LightGBM primary** for Gα; CatBoost soft = sensitivity |
| M3 | Teacher calibration missing | **Accept** | Teacher val-aux Brier/ECE raw |
| M4 | W0 arms scope bloat | **Accept** | **Drop W0-KD from v1** claim grid |
| M5 | Nα in user-bar pool dishonest | **Accept** | Nα **excluded** from beat-C1 bar |
| nit | T-sweep trigger ambiguous | **Accept** | §3.4 explicit OR trigger |
| — | Re-clean / reopen B1–B4 / CRD-mandatory / Diasense exact repro | **Reject** | False positives; out of scope |

**Verdict after disposition:** plan **ready for implement** pending **user go**.
