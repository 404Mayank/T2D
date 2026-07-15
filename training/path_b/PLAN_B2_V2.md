# Path B2-V2 — better Stage-1 emulator + variance-propagated stacking

**Date:** 2026-07-16  
**Status:** **IMPLEMENTED & CONCLUDED** 2026-07-16 — claim `b2v2_grid_20260716`; see `REPORT_B2_V2.md`.  
**Role:** **Sibling retry** of the B2 *tabular handoff cell* under a **new recipe**. Does **not** reopen frozen B2 HPO (`b2_grid_20260715` / `REPORT_B2.md`).  
**Authority:** `Training.md` §4 B2; frozen B2 `REPORT_B2.md` + `PLAN_B2.md` + `DECISIONS.md`; data `PROCESSED.md`; residual knobs `REVIEW_PHASES.md` §2.3.  
**User ambition bar:** beat Path A **C1** (test 4-AUC **0.7378** / binary **0.8309** / macro AUPRC **0.4687**) via a **deployable** arm on full `wearable_core` — **honest prior: low** given SNR probes; fair science is the ablation.  
**Fair internal bar:** re-fit **D1 ≡ C1** and **D0 ≡ W0** under this package (bit-exact parity was achieved in frozen B2 — preserve it).

---

## 0. Why this is a retry, not a reopen

Frozen B2 closed **one recipe**:

> Stage-1 = 8× LightGBM **point** regressors on **person GREEN (30) → person CGM 8-vector daymean**, Ŷ handed to Stage-2 as **noise-free** features.

Claim result (`b2_grid_20260715`): best deployable = **D1 ≡ C1**; T1 null / slightly harms binary; oracle O1−D1a **+0.094**. Stage-1 val mean R² **~0.05**.

Per project locks: **do not** re-HPO that recipe. This plan is a **user-opened** residual-knob retry (prompt `PROMPT_B2_RETRY.md`) of the **modular tabular handoff** cell. Primary scientific cell is **variance-propagated stacking** of a daily-grain quantile Stage-1 (joint with reduced Y); “better emulator” is a **co-investigated** knob isolated by the required `P_point` arm — not a silent claim that daily R² will close the oracle gap.

Attacks residual knobs named in `REPORT_B2` §7 / `REVIEW_PHASES` §2.3:

1. **Variance-propagated stacking** (Stage-2 sees Ŷ uncertainty — load-bearing hypothesis for why T1 harmed binary).  
2. **Daily-grain Stage-1 + reduced collinear Y** (better-conditioned regression than person GREEN → 8-vec; co-investigated).  
3. Quantile LGBM emitter (no new deps); SSL / deep nets **deferred**.

### vs `REPORT_B2` §7 “Defer to B4 lane”

Frozen §7 said daily `watch_daily` → person glu Stage-1 should **“Defer to B4 lane (dynamics), not B2 reopen.”** That sentence steered *against reopening the frozen point-estimate recipe* and toward dynamics work, which B4 then ran at **5-min sequence** grain and concluded null. **B2-V2 is not that B4 cell and not a B2 HPO reopen:**

| | B2 frozen | **B2-V2 (this plan)** | B4 (concluded) |
|---|---|---|---|
| Grain | person tabular | **daily tabular → person agg** | 5-min sequence |
| Handoff | point Ŷ only | **quantiles + spread/daysd** | z / traj multi-task / distill |
| Package | `b2/` frozen | **`b2v2/` new** | `b4/` frozen |

Daily **scalar** handoff with explicit **error propagation** was never run. Sequence dynamics remain B4’s closed cell.

### Ladder precedence vs B3

Default Path B claim ladder remains **B1 → B2 → B4 → B3 last** (`Training.md` / `AGENTS.md`). **B2-V2 does not reorder B3 off that ladder.** It is an optional residual-knob branch on the B2 *tabular* cell, opened by user prompt. Full `b2v2_grid_*` claim run only on **explicit user go**. B3 (`PLAN_B3.md`) keeps default precedence for the next *ladder* claim package unless the user prioritizes b2v2. Smoke / Stage-1 R² sanity for b2v2 may run without claiming ladder completion.

**Package isolation:** implement under `training/path_b/b2v2/` (new). Frozen `training/path_b/b2/` stays read-only reference. Artifacts / run ids: `b2v2_*` only. Sibling report: `REPORT_B2_V2.md` (never overwrite `REPORT_B2.md`).

---

## 1. Data readiness verdict

### Verdict
**No re-clean. No new FE required for B2-V2.** Existing processed assets are sufficient. Stage-1 will **consume** daily matrices already shipped for B1; any “target reduction” is a **train-time column subset**, not a pipeline rebuild.

| Asset | Status | B2-V2 use |
|---|---|---|
| `features/watch_daily.parquet` | ready 22844×24; sleep units fixed | Stage-1 **primary X** (day grain) |
| `features/cgm_daily.parquet` | ready 19805×12; 0 nulls on stats | Stage-1 **primary Y** (day grain) |
| `features/cgm_person.parquet` | ready 1924×12 | Stage-1 **person-agg targets** + oracle Y; reduced subset |
| `features/watch_green.parquet` | ready 1824×31 | Stage-2 W0; optional Stage-1 **static fuse** |
| onboarding / mood | ready | Stage-2 C1 (match freeze) |
| `meta/pool_masks.parquet` | ready | pools + labels + split |
| `grid_5min*` | ready | **out of scope** (B4 cell) |

### Empirical anchors (session probes 2026-07-16 — **not claim numbers**)

| Check | Result |
|---|---|
| Aux both-valid days (watch∧cgm) | **1685/1685** pids; min **4** / p10 **9** / med **11** / max **14** |
| train∩aux both-valid days | **12610** days / **1184** pids; insulin pids **69** |
| Person 8-vec collinearity (aux) | TIR+TBR+TAR **=1**; cv≈sd/mean (r≈0.998); SVD: top-3 PCs ≈ **68%+19%+10%**; last SV **≈0** |
| Person GREEN → daymean HistGB R² (val) | mean **0.05**, tir **0.08**, tar **0.07**, tbr **−0.19**, cv **−0.06** — matches frozen B2 near-floor |
| Daily watch → daily CGM HistGB R² (val day) | mean **0.08**, tir **0.07**, tar **0.07**; person-agg of daily Ŷ: mean **~0.10** |
| Daily + GREEN fuse | not fully re-probed after name-collision; expect modest, not leap |
| Sleep null rate on watch_daily | `sleep_duration_hours` **~28%** null; nocturnal HR **~14%** — impute policy required (below) |
| Outer day coverage | watch-only days 4286; cgm-only 1247; concurrent inner 18558 — fine for aux training |

### FE / cleaning changes?
| Question | Answer |
|---|---|
| Re-clean? | **No** |
| New FE columns? | **No** for primary plan |
| Train-time only transforms? | **Yes** — day join, median impute (train-fit), target subset, person aggregation of day Ŷ |
| Aux pool policy change? | **No** — keep `aux_eligible` 1685 / span-overlap gate |

### Caveats surfaced (not already load-bearing in frozen docs)

1. **Daily grain does not free-lunch the SNR wall.** Session probe day-level R² is only ~+0.02–0.05 over person GREEN → person daymean. A daily emulator that still lands **person-level val R² < 0.10** on the primary reduced targets is an **early kill** for deployable raise hopes (concern #1 in the retry prompt). Report honestly; do not chase SSL in this plan.  
2. **Short concurrent series.** Some aux pids have only **4** both-valid days (p10=9). Day-level models get more *rows* but not more *persons*; person-agg Ŷ variance is high for short series — another reason to pass **interval / residual features**, not point means alone.  
3. **Rank-1 collinearity is exact on time-in-range triple.** Using all 8 as independent Stage-2 features is actively harmful under weak Ŷ (frozen T1 binary harm). Reduced Y is **primary**, not optional footnote.  
4. **Non-aux still need day→person Ŷ without CGM labels.** Deployable full-core T\* requires predicting on watch-valid days for non-aux and aggregating — same scientific cell as frozen non-aux rule, different grain.  
5. **Watch-daily nulls / counts.** Do **not** put `hr_n` / `stress_n` / `rr_n` / `cgm_n` / validity flags into Stage-1 X (coverage leakage cousins). Impute continuous channels with **train-aux medians** fit once; keep trees’ native missing handling as secondary path only if documented.  
6. **Insulin rarity unchanged.** train∩aux insulin **69** → ~13–14 / OOF fold. Rich nets deferred; stick to regularized LGBM quantiles.

---

## 2. Goal

### Scientific question
Does **variance-propagated** stacking of a **daily-grain quantile** Stage-1 (reduced CGM targets) improve deployable T2D discrimination over the **matched** Stage-2 base without predicted CGM — and does the daily point emulator alone help, or only with uncertainty features?

**Prior (from probes):** person-agg val R² likely ~0.05–0.10; mid Ŷ is partly a nonlinear map of GREEN already inside C1. A deployable raise is **possible but not expected**. The pre-registered value is a clean joint ablation of (daily mid, variance pack) vs D1, not a forced ambition win.

### Claims B2-V2 may make
1. **Primary ablation:** **T1v − D1** on full `wearable_core` (Δ4-AUC + paired person boot CI) — variance pack on C1.  
2. **Variance vs point:** **T1v − T1p** — is uncertainty load-bearing given the same daily emulator?  
3. **Point daily alone:** **T1p − D1** — better-conditioned mid without variance.  
4. **Watch-only handoff:** **T0v − D0** (and T0p diagnostic).  
5. **Oracle ceiling (sanity, matched aux):** O1 vs D1a — expect still ~+0.09 if Stage-2 parity holds; not a deployable claim.  
6. **Stage-1 quality:** person-level R² / pinball / interval coverage on reduced targets (val/test aux).  
7. **Ambition bar (secondary):** best deployable T\* vs D1 and frozen C1 anchor — report; do not soft-pass on ambition alone.

### Claims B2-V2 is not
- Reopen of frozen point-estimate B2 HPO.  
- Multi-task (B1), traj/rep-distill (B4), logit-KD (B3).  
- Path A number changes.  
- “LUPI works” from oracle alone.  
- SSL sequence encoder Stage-1 (deferred; only if (a)+(b) leave clear headroom **and** user opens a follow-on plan).

---

## 3. Design locks

### 3.1 Pipeline shape

```
Stage-1 (fit only on train ∩ aux_eligible days / persons):
  Day model g_day:
    X_day = watch_daily continuous channels (± optional GREEN static fuse)
    Y_day = reduced CGM daily stats
    → for each day: quantile preds (lo, mid, hi) per target
  Person aggregate A:
    ŷ_mid_person = mean of day mid over watch-valid days in window
    ŷ_spread_person = mean of (hi−lo)  [predictive interval width]
    ŷ_day_sd_person = sd of day mid   [within-person day volatility of Ŷ]
    (+ optional OOF residual scale features — §3.5)

Stage-2 (T2D; CatBoost + LightGBM, Path A family — same as frozen B2):
  X2 = base_block ∪ handoff_features   # or true reduced CGM for oracle
  Y2 = label ∈ {0,1,2,3}
```

**Deployable inference:** watch days only → `g_day` → aggregate → Stage-2. **Never** true CGM at infer.

### 3.2 Reduced Stage-1 targets (primary)

**Primary reduced set (4):**

| Short | Daily col | Person agg target |
|---|---|---|
| mean | `cgm_mean` | `cgm_mean_daymean` |
| sd | `cgm_sd` | `cgm_sd_daymean` |
| tir | `cgm_tir_70_180` | `cgm_tir_70_180_daymean` |
| tar | `cgm_tar_180` | `cgm_tar_180_daymean` |

**Dropped from primary Y (and from primary handoff):**  
`cgm_cv*` (≈ sd/mean), `cgm_min*`, `cgm_max*`, `cgm_tbr*` (near-null R² in frozen B2; TIR+TAR already carry hyperglycemia structure; TBR mass tiny).

**Sensitivity (optional, non-blocking):** 3-target `{mean, sd, tir}` only — if 4-target Stage-2 looks collinear-noisy.

**Forbidden** as targets or Stage-2 features: `n_valid_days`, `cgm_n*`, `n_days`, validity flags, `hr_n` / `stress_n` / `rr_n`.

### 3.3 Stage-1 X (day grain)

**Primary day channels** (from `watch_daily`, continuous only):

```
hr_mean, hr_sd, hr_min, hr_max, hr_nocturnal_mean, hr_day_mean,
stress_mean, stress_sd, stress_pct_medium_plus, stress_pct_high,
rr_mean, rr_sd,
sleep_duration_hours, sleep_n_bouts,
steps_sum, mvpa_min, light_min, sedentary_min
```

**Static fuse (primary on):** broadcast person GREEN **30** numerics onto each day row (same W0 as Path A). Rationale: frozen B2 Stage-1 *only* had GREEN; daily adds within-person variation without discarding the person signal.  
**Sensitivity off:** day channels only (no GREEN) — isolates pure daily SNR.

**Day row filter for Stage-1 *supervised* fit:**  
`watch_day_valid ∧ cgm_day_valid ∧ aux_eligible` (person in train for train folds).

**Day row filter for *inference* Ŷ:**  
`watch_day_valid` on the person (aux or not). If a person has **zero** watch-valid days in the observed table (should not happen on core), fall back to train-aux median person handoff vector and **flag** in diagnostics (expect 0).

**Impute:** fit medians on **train∩aux supervised days** only; apply to all splits. Do not use val/test for medians.

### 3.4 Stage-1 model (emitter)

| Choice | Lock |
|---|---|
| Family | **LightGBM** only for primary (installed; Path A native). **No NGBoost / MAPIE** dependency for claim (not installed). |
| Heads | Per target × **3**: mid = **MSE regression** (`objective=regression`); lo/hi = quantiles α∈`{0.1, 0.9}`. |
| Mid point | MSE mid is the point estimate for R² / aggregation. *(Impl note 2026-07-16: pure quantile α=0.5 mid gave negative R² while MSE mid / HistGB ~+0.06; switched mid to MSE, kept quantile tails for spread.)* |
| HPO | ≤ **20 trials per (target, α=0.5)** maximizing val person-agg R² on that target (cheaper than 8×30); pin α∈{0.1,0.9} to the **same** hyperparams as mid (or 10 trials shared). Seed 42. |
| Early stop | val day-level pinball or RMSE on mid; `stage1_es_rounds` ~50; `n_estimators` ≤500. |
| OOF | K=5 **person-stratified** by label on train∩aux **persons** (not days). Fit day model on days of persons in fold-train; predict days of fold-holdout persons + all non-aux train persons. |
| Non-aux train Ŷ | **Mean of K fold-models’ person-agg predictions** (frozen B2 rule, adapted to day grain). |
| Val/test Ŷ | Single final fit on **all train∩aux** supervised days (early-stop on val∩aux days); apply to all core val/test persons’ watch-valid days → aggregate. |
| Deep net / SSL | **Out of scope** for this plan. |

**Person aggregation (deterministic):**

For each person, over inferred watch-valid days with finite mid:

- `yhat_<t>_mid` = mean(day mid)  
- `yhat_<t>_spread` = mean(day hi − day lo)   # predictive interval width  
- `yhat_<t>_daysd` = sd(day mid) if n_days≥2 else 0  

Optional (primary **on** for variance pack):

- `yhat_<t>_resid_abs` = train-fold mean absolute residual scale: for OOF aux persons use |y_true_person − yhat_mid|; for non-aux / val / test use **global train-OOF MAE** of that target (constant feature — still lets Stage-2 learn a reliability prior).  
  *Simpler primary:* skip person-specific resid on val/test; only pass **spread + daysd** as uncertainty. **Lock primary uncertainty features = `{spread, daysd}` per target.** Resid-abs = sensitivity only.

### 3.5 Stage-2 handoff feature packs (pre-registered)

| Pack id | Columns | Role |
|---|---|---|
| **P0** | none | matched direct (D0/D1) |
| **P_point** | 4× `yhat_<t>_mid` only | daily-emulator point handoff (controls “better emulator alone”) |
| **P_var** | 4× mid + 4× spread + 4× daysd (**12**) | **primary variance-propagated handoff** |
| **P_true** | 4× true person daymeans | oracle only |

Primary deployable two-stage arm uses **P_var**.  
**P_point** is a **required diagnostic arm** (not optional): isolates whether variance features are load-bearing vs daily mid alone.

### 3.6 Feature blocks (Stage-2)

| Block | Columns | Role |
|---|---|---|
| **W0** | 30 GREEN | matched watch direct / T0 base |
| **C1** | W0 + onboarding_keep + paidscore + cestl | matched C1 / T1 base (**47** feats) |
| **P_*** | see §3.5 | handoff / oracle |

**C1 manifest lock:** resolve cols from `training/path_a_blocks/config.yaml` at runtime; snapshot to `b2v2/artifacts/<run_id>/c1_feature_manifest.json`. Assert n_feat=47 and names stable vs frozen B2 snapshot if available.

**HPO space lock:** snapshot `hpo.lightgbm` / `hpo.catboost` (and class-weight / calibration pins) to `b2v2/artifacts/<run_id>/hpo_space_snapshot.json`. Prefer **byte-stable** copy of spaces as used in frozen `b2/config.yaml` (already pinned from Path A at B2 freeze) rather than a live drifting `path_a_blocks` edit. Assert equality to frozen B2 HPO block when that file is present.

### 3.7 Leakage rules (hard)

1. Stage-1 never fit on val/test **persons**.  
2. Day rows from val/test persons never enter Stage-1 training.  
3. OOF + non-aux mean-K rule as §3.4.  
4. No `label` / `recommended_split` / `clinical_site` / pool flags / coverage counts in X.  
5. Oracle arms: true CGM only on **aux-only** pools; **no** median-fill true CGM onto non-aux as primary.  
6. Outer claim split = fixed `recommended_split` only.  
7. Impute statistics fit on train∩aux only.

### 3.8 Arms (pre-registered)

| ID | Pool | Stage-2 features | Handoff | Deployable? | Role |
|---|---|---|---|---|---|
| **D0** | core 1824 | W0 | — | yes | matched W0; **parity vs freeze 0.6662** |
| **D1** | core 1824 | C1 | — | yes | matched C1; **parity vs freeze 0.7378** |
| **T0p** | core | W0 + P_point | pred mid | yes | daily point on watch |
| **T1p** | core | C1 + P_point | pred mid | yes | daily point on C1 |
| **T0v** | core | W0 + P_var | pred+unc | yes | variance pack on watch |
| **T1v** | core | C1 + P_var | pred+unc | yes | **primary deployable two-stage** |
| **D1a** | aux 1685 | C1 | — | baseline | matched oracle baseline |
| **O1** | aux 1685 | C1 + P_true | true | **no** | oracle ceiling |

**Primary comparisons**
1. **T1v − D1** (full core) — primary ablation.  
2. **T0v − D0** — watch-only handoff.  
3. **T1v − T1p** — does variance pack beat point-only under same daily emulator?  
4. **T1p − D1** — does better (daily) point emulator alone help?  
5. **O1 − D1a** — ceiling sanity (expect pass ≥ +0.02).  
6. Ambition: T1v (and best T\*) vs D1 and frozen C1 anchor.

### 3.9 Stage-2 models / HPO / metrics

**Identical in spirit to frozen PLAN_B2:**

| Item | Lock |
|---|---|
| Families | CatBoost + LightGBM multiclass |
| Class weights | LGBM `class_weight="balanced"`; Cat `auto_class_weights="Balanced"` |
| HPO spaces | **Identical** to `path_a_blocks/config.yaml` `hpo.*` |
| Trials | ~50 / family / arm (smoke: 5); seed 42 |
| Select | val macro-OVR AUC; tie → macro AUPRC within `auc_tie_eps=0.005` |
| Calibration | val sigmoid primary; isotonic secondary if min_pos≥30; **ranking claim on raw** |
| Metrics | 4-AUC, macro AUPRC, binary AUC/AUPRC, Brier, per-class OVR |
| Bootstrap | paired person Δ, n=2000, seed 42 |
| CatBoost | Ordered with Plain fallback (Path A / B2 practice) |

### 3.10 Decision bars

| Question | Pass rule |
|---|---|
| Primary ablation | **T1v−D1** test Δ4-AUC **> 0** and boot CI **lo > 0** |
| Watch handoff | **T0v−D0** same rule |
| Variance helps vs point | **T1v−T1p** point Δ4-AUC > 0 (soft: CI may include 0; always report) |
| Point daily alone helps | **T1p−D1** same hard rule as primary (likely fail if R² wall) |
| User ambition | Best deployable T\* **> D1** under hard rule; frozen C1 external anchor. Fallback: if D1 < freeze−0.01, fair bar is T vs D1 only; still publish external-anchor Δ as non-primary |
| Oracle headroom | O1−D1a ≥ **+0.02** |
| Kill pivot (privilege gone) | O1−D1a < **+0.01** → write report; stop Stage-1 chasing |
| **Stage-1 early kill (R²)** | After Stage-1 full: if val **person-agg mean R² on {mean,sd,tar} < 0.10** **and** no target ≥ 0.12, **do not** expect deployable raise; still may run **cheap** Stage-2 smoke on T1v vs D1 to confirm null, but **full 50-trial multi-arm grid is optional** (user call). Always persist Stage-1 metrics. |
| **Stage-1 interval coverage gate** | On val∩aux person-agg, fraction of true target ∈ [ŷ_lo, ŷ_hi] (lo/hi = mean of day α=0.1/0.9) should fall in **[0.60, 0.95]** for **≥2 of 4** reduced targets before `b2v2_grid_*`. If violated: (i) re-HPO quantile tails (≤10 shared trials) **or** (ii) demote primary handoff to **P_point** and treat P_var as diagnostic only — log choice in DECISIONS. |
| Stage-1 smoke gate | val person-agg R² on mean **> 0** (non-degenerate). If ≤0, fix plumbing before grid. |
| Parity | \|D1 − freeze C1\| and \|D0 − freeze W0\| on 4-AUC within **1e-5** preferred; hard fail only if **> 0.01** low (then diagnose before claiming Δ) |

**B2-V2 does not block B3; B3 keeps default ladder precedence** (§0). Does not unfreeze B1/B4 claim grids.

---

## 4. Protocol details

### 4.1 Cohorts (verified; same as B2)

| Pool | train | val | test | train insulin |
|---|---:|---:|---:|---:|
| wearable_core | 1277 | 270 | 277 | 80 |
| core ∩ aux | 1184 | 247 | 254 | 69 |
| core \ aux | 93 | 23 | 23 | — |

### 4.2 Diagnostics (required)

1. Stage-1 **person-agg** R²/RMSE table (val/test aux) for reduced targets + mean R².  
2. Day-level R² table (secondary).  
3. Quantile **interval coverage** on val aux (person-agg true ∈ [lo,hi]); gate §3.10.  
4. Ŷ drift: percentiles of mid/spread on val/test by aux vs non-aux.  
5. OOF fold label counts (insulin ~13–14).  
6. C1 manifest + **HPO space** snapshots/asserts.  
7. D0/D1 vs frozen Path A scores.  
8. Correlation matrix of handoff features on train (flag if mid≈linear combo of C1 / GREEN).  
9. n_days used per person for aggregation (min/p50/max; aux vs non-aux) **and** spread/daysd percentiles **bucketed by n_days** (1 / 2–7 / 8+).  
10. Join-key assert: supervised Stage-1 rows require exact `(person_id, day_local)` match watch∧cgm (no ordinal day math).

### 4.3 Package layout

```
training/path_b/b2v2/
  config.yaml
  data.py          # day join, impute, person agg, OOF non-aux rule, C1 load
  stage1.py        # quantile LGBM multi-head, OOF, metrics
  stage2.py        # reuse B2/Path A patterns; arm feature packs
  evaluate.py      # metrics + bootstrap + bars
  run.py / __main__.py
  artifacts/<run_id>/
```

Reuse `path_a_watch` / `path_a_blocks` / frozen `b2` helpers where safe — **no drive-by refactors** of frozen packages.

### 4.4 Run ladder

| Step | Run id pattern | What | Claimable? |
|---|---|---|---|
| 0 | (dev) | unit checks: join keys, impute leakage, agg shapes | no |
| 1 | `b2v2_smoke_YYYYMMDD` | subset train (~200 pids), 5 Stage-2 trials; arms D1, T1p, T1v, D1a, O1 | no |
| 2 | `b2v2_s1_YYYYMMDD` | full Stage-1; persist Ŷ packs + metrics; **apply early-kill** | Stage-1 metrics yes; not T2D claim |
| 3 | `b2v2_grid_YYYYMMDD` | full arms D0,D1,T0p,T1p,T0v,T1v,D1a,O1 (user go) | **yes** if protocol clean |
| 4 | docs | `REPORT_B2_V2.md` + `DECISIONS.md` + README pointer | — |

**This planning turn:** present plan; **smoke + Stage-1 R² sanity allowed** after implement approval; **no full claim grid** unless user says so.

### 4.5 Seeds
Global seed **42**; bootstrap 2000× seed 42; OOF `StratifiedKFold` seed 42.

---

## 5. Out of scope

- Re-clean; GREEN re-FE; aux threshold changes  
- Mutating frozen `b2/` claim artifacts or `REPORT_B2.md`  
- NGBoost / MAPIE / deep Stage-1 / SSL encoder  
- B1 λ grids, B4 reopen, B3 implementation (B3 remains ladder-next for logit-KD baseline)  
- Survey blocks beyond C1  
- Median-filled full-core “oracle”  
- Stage-1 X = full C1 clinical (age/BMI) as **primary** (that’s not a watch emulator; optional footnote only if user asks)

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Daily R² still ~0.05–0.10 (SNR wall) | Early-kill §3.10; report Stage-1 as the finding; optional cheap T1v smoke only |
| Variance pack still collinear noise | Pre-registered T1v vs T1p vs D1; if all fail, recipe family still null — honest |
| D1 ≠ freeze | Same HPO/spaces/weights as B2; parity asserts before Δ claims |
| Day-model overfit aux | Person-level OOF; regularized LGBM; watch fold insulin counts |
| Spread miscalibrated | Coverage diagnostic; Stage-2 can down-weight via trees if useless |
| Impl complexity / time | Quantile LGBM only; no new deps; reduced 4 targets |
| Scope creep into SSL/B4 | Explicit out-of-scope; new PLAN if needed |

---

## 7. Success / freeze criteria

| Outcome | Interpretation |
|---|---|
| T1v−D1 CI lo>0 and T1v ≥ freeze C1 | **Deployable raise** — major Path B positive under new recipe |
| T1v−D1 pass but T1v < freeze slightly | Ablation pass; ambition soft — document |
| T1v fail, T1p fail, O1 headroom pass, Stage-1 R²&lt;0.10 | **Confirms SNR wall + handoff family null** even with daily+intervals — strong negative for modular CGM summary handoff |
| T1v fail but T1v > T1p with CI interest | Variance helps directionally but not enough — footnote |
| O1−D1a kill | Privilege disappeared under re-fit — stop; debug pool/features |

**Realized (`b2v2_grid_20260716`):** T1v fail, T1p fail, T1v ≯ T1p, O1 headroom pass, Stage-1 val mean R² ~0.09 / test ~0.03 → **handoff family null** under daily+variance recipe.

---

## 8. Critique disposition (2026-07-16, `critiquer` / glm-5.2:high)

| # | Item | Sev | Disposition | Action in plan |
|---|---|---|---|---|
| 1 | Ladder order vs B3 unresolved | high | **Accept** | §0 precedence: B3 keeps default ladder; b2v2 user-opened residual branch; full grid only on user go |
| 2 | REPORT_B2 §7 “Defer to B4 lane” elided | high | **Accept** | §0 table: daily tabular+variance ≠ B4 5-min sequence; not frozen-B2 HPO reopen |
| 3 | “Better emulator” overstates ambition; mid redundant w/ C1⊇GREEN | med | **Accept (framing)** | §0/§2: primary cell = variance-propagated stacking; daily mid co-investigated via P_point; ambition secondary |
| 4 | No gate on quantile mis-coverage | med | **Accept** | §3.10 coverage ∈[0.60,0.95] on ≥2/4 targets or demote P_var |
| 5 | Live HPO ref can drift vs freeze | med-low | **Accept** | §3.6 HPO snapshot + pin to frozen b2 spaces |
| 6 | Join key / n_days×spread diagnostics | med/low | **Accept** | §4.2 #9–10 |
| 7 | resid_abs train/val asymmetry | low | **Keep as-is** | already sensitivity-only; primary = spread+daysd |
| 8 | Early-kill threshold near probe R² | note | **Keep** | gate intentional; either side is informative |

**Verdict after disposition:** plan **ready for user approve → implement** (smoke / s1 first; claim grid gated).

---

## 9. Implementation gate

**Done:** smoke + s1 + full claim grid. Outcome: deployable **null** (T1v 0.727 &lt; D1 0.738); oracle **+0.096**. Package frozen unless new plan.

### Impl critique disposition (2026-07-16)
| # | Item | Disposition |
|---|---|---|
| OBJ-1 | smoke gate mean/sd/tar vs plan mean-only | **Fixed** — `gate_pass` = mean R²>0; 3-target diagnostic only |
| OBJ-2 | non-aux K-avg on day preds compresses daysd | **Fixed** — person-agg per fold, then mean of K person packs |
| OBJ-3 | mid↔C1/GREEN corr discarded | **Fixed** — full mid_vs_green + mid_vs_c1 + flags |
| OBJ-4 | coverage demotion flag-only | **Fixed** — primary_ablation_key / ambition keys off pack |
| — | n_days aux/non-aux + fallback counts | **Fixed** |
