# B2-V2 final report — daily MSE mid + variance-propagated stacking

**Status:** **B2-V2 CONCLUDED** (2026-07-16).  
**Protocol:** `PLAN_B2_V2.md` (critiqued → revised → implemented → mid-recipe fix).  
**Authority for claims:** this report + `DECISIONS.md` (2026-07-16 B2-V2 entries).  
**Claim run:** **`b2v2_grid_20260716`**  
**Stage-1 ref:** `b2v2_s1_20260716_msemid` (same recipe; grid re-fit Stage-1 in-process)  
**Smoke (non-claim):** `b2v2_smoke_20260716_msemid`  
**Invalidated:** `b2v2_s1_20260716` / first smoke (quantile α=0.5 mid — negative R²)

Path A numbers are **frozen** and **unchanged**. W0 **0.6662** / bin **0.6889**; C1 **0.7378** / **0.8309** / AUPRC **0.4687**.  
Frozen B2 (`b2_grid_20260715`) is **not** overwritten; this is a sibling recipe.

---

## 0. Executive conclusion

| Question | Answer |
|---|---|
| Does **any deployable B2-V2 arm beat C1 / D1**? | **No.** Best deployable = **D1 ≡ C1** (0.7378 / 0.8309). |
| Does **variance pack** help vs matched D1? | **No** — T1v−D1 Δ4-AUC **−0.011** CI **[−0.031, +0.010]** lo≯0 |
| Does **daily point** mid help vs D1? | **No** — T1p−D1 Δ **−0.008** CI **[−0.025, +0.008]** |
| Does variance beat point (same Stage-1)? | **No** — T1v−T1p Δ **−0.003** CI includes 0 |
| Watch-only variance pack? | **No** — T0v−D0 Δ **−0.001** |
| Oracle headroom (matched aux)? | **Yes** — O1−D1a **+0.096** CI **[+0.062, +0.129]** |
| Stage-1 better than frozen B2? | **Modestly on val** (mean R² **0.126** vs ~0.05); **test still ~0.03–0.04** |
| Protocol parity D0/D1 vs freeze? | **Yes** (within ~1e-5 on reported 4-dp) |
| Proceed ladder? | B3 already frozen; B2 tabular handoff family **closed null** under both recipes; optional only via new `PLAN_*` |

**Scientific takeaway:** Modular two-stage handoff of **daily-grain** CGM summaries — even with **MSE mid + quantile interval features** and reduced collinear Y — **does not** improve deployable T2D over matched C1. Oracle still proves ~**+10 pp** privilege on aux. The residual failure mode is **watch→glucose SNR + error propagation into a strong tabular Stage-2**, not “forgot variance columns.”

### Post-hoc adversarial audit (2026-07-16)
Fresh `critiquer` (assume-wrong stance): **null AUTHENTIC**. No blocker leakage/wiring bugs. Material probes:
- Quantile-crossing mid rewrite: rare (≪1% days), ΔR² ~0.001.
- Day-set HPO vs deploy mismatch: ΔR² ~0.002.
- Val early-stop peek: val→test drop **symmetric** across D1/T1v.
- T1v CatBoost **yhat importance share ~18%** (tir_spread rank 5) — Stage-2 uses Ŷ and still loses to D1.
Freeze **stands**; no re-run required for claim integrity.

---

## 1. Setup (locked)

| Item | Value |
|---|---|
| Stage-1 X | day watch continuous (18) + GREEN static fuse (30) = 48 |
| Stage-1 Y | reduced 4: mean, sd, tir, tar (daily → person agg) |
| Stage-1 mid | **MSE** LightGBM (`regression`) |
| Stage-1 tails | quantile α∈{0.1, 0.9} → spread |
| Handoff | P_point: 4× mid; P_var: mid+spread+daysd (12) |
| Stage-1 fit | train∩aux supervised days; person OOF K=5; non-aux = mean of K person-agg packs |
| Stage-2 | CatBoost + LightGBM, Path A HPO spaces (pinned frozen b2), val-select AUC→AUPRC |
| Class weights | LGBM `balanced` / Cat `Balanced` |
| Claim pool D/T | wearable_core 1824 (1277/270/277) |
| Claim pool O/D1a | aux 1685 (1184/247/254) |
| Seed | 42; bootstrap n=2000 |
| HPO | Stage-1 20 trials/target mid; Stage-2 50 trials/family |

**Package:** `training/path_b/b2v2/`. No re-clean; no new FE.

### Mid-recipe fix (pre-claim)
Quantile α=0.5 mid produced val person-agg R² **&lt; 0** (`b2v2_s1_20260716`, invalidated). Probe: HistGB / LGBM-MSE ~+0.06–0.12 on same days. **Lock:** mid = MSE; tails stay quantile. Documented in plan §3.4 + DECISIONS.

---

## 2. Runs

| Run id | Role | Claimable? |
|---|---|---|
| `b2v2_smoke_20260716` | first smoke | no — quantile mid + deny bug |
| `b2v2_s1_20260716` | full s1 quantile mid | **invalidated** |
| `b2v2_s1_20260716_msemid` | full s1 MSE mid | Stage-1 yes |
| `b2v2_smoke_20260716_msemid` | smoke post-fix | no |
| **`b2v2_grid_20260716`** | full arms claim | **yes** |

---

## 3. Stage-1 emulator quality (`b2v2_grid_20260716`)

Val mean R² **0.094**; test mean R² **0.029**.  
Smoke gate (mean R²&gt;0): **PASS**. Coverage all 4 ∈[0.79,0.84]: **PASS**. Early-kill: **not triggered** (mean target val R² 0.126 ≥ 0.12).

| Target | val R² | test R² | val coverage |
|---|---:|---:|---:|
| mean | **0.126** | 0.039 | 0.794 |
| sd | 0.074 | 0.041 | 0.814 |
| tir | 0.094 | 0.019 | 0.842 |
| tar | 0.083 | 0.019 | 0.814 |

vs frozen B2 person GREEN → 8-vec (val mean R² ~0.05; mean target ~0.085): daily MSE mid **improves val mean** but **does not** raise test R² above the ~0.03 floor. Fallbacks (0 watch-valid days): **0** train/val/test.

---

## 4. Stage-2 results (`b2v2_grid_20260716`)

### 4.1 Arm table (test, raw ranking)

| Arm | Pool | n_feat | Family | 4-AUC | Binary | Macro AUPRC |
|---|---|---:|---|---:|---:|---:|
| **D0** | core | 30 | CatBoost | **0.6662** | **0.6889** | 0.3916 |
| **D1** | core | 47 | CatBoost | **0.7378** | **0.8309** | **0.4687** |
| T0p | core | 34 | LightGBM | 0.6748 | 0.6976 | 0.4079 |
| T1p | core | 51 | CatBoost | 0.7296 | 0.8129 | 0.4755 |
| T0v | core | 42 | CatBoost | 0.6652 | 0.6862 | 0.3787 |
| **T1v** | core | 59 | CatBoost | **0.7271** | **0.8174** | 0.4701 |
| D1a | aux | 47 | LightGBM | 0.7420 | 0.8319 | 0.4805 |
| **O1** | aux | 51 | CatBoost | **0.8378** | **0.8796** | **0.6234** |

### 4.2 Protocol parity

| Compare | 4-AUC | Binary | AUPRC |
|---|---:|---:|---:|
| Frozen Path A W0 | 0.6662 | 0.6889 | 0.3916 |
| B2-V2 **D0** | 0.6662 | 0.6889 | 0.3916 |
| Frozen Path A C1 | 0.7378 | 0.8309 | 0.4687 |
| B2-V2 **D1** | 0.7378 | 0.8309 | 0.4687 |

D1/D0 match freeze to reported precision (abs Δ ~1e-5). Fair bar is trustworthy.

### 4.3 Pre-registered comparisons (paired person bootstrap, n=2000, seed 42)

| Contrast | Δ4-AUC point | 95% CI | lo&gt;0? | Verdict |
|---|---:|---|---|---|
| **T1v − D1** | −0.0107 | [−0.0311, +0.0103] | No | **primary fail** |
| **T0v − D0** | −0.0009 | [−0.0188, +0.0183] | No | fail |
| **T1v − T1p** | −0.0025 | [−0.0175, +0.0124] | No | variance ≯ point |
| **T1p − D1** | −0.0082 | [−0.0251, +0.0080] | No | daily point fail |
| **O1 − D1a** | **+0.0958** | **[+0.0620, +0.1295]** | **Yes** | **oracle headroom pass** |

Binary side (T1v−D1): Δbin **−0.014** CI includes 0 (not the frozen-B2 “CI entirely &lt;0” harm, but still no raise).

### 4.4 Decision bars

| Bar | Outcome |
|---|---|
| Stage-1 smoke / coverage | **Pass** |
| Early-kill | **Not triggered** |
| T1v vs D1 | **Fail** |
| T0v vs D0 | **Fail** |
| T1p vs D1 | **Fail** |
| T1v vs T1p | **Fail** (no variance win) |
| User ambition beat C1 | **Fail** |
| Oracle headroom ≥ +0.02 | **Pass** (+0.096) |
| Kill pivot | **Not triggered** |

---

## 5. Interpretation

1. **Variance-propagated stacking did not rescue the handoff.** With calibrated-ish intervals (coverage ~0.8) and better val R² than frozen B2, Stage-2 still prefers pure C1. T1v ≈ T1p &lt; D1.
2. **Daily grain is a weak Stage-1 upgrade, not a ceiling breaker.** Val mean R² 0.13 is real progress vs person GREEN ~0.05; test ~0.04 shows the same generalization wall seen in B2/B4 wear→glucose probes.
3. **Oracle remains large** (+0.096 on reduced 4 true daymeans) — LUPI privilege is real; representation/handoff of *predicted* summaries is not.
4. **D0/D1 bit-parity** reconfirms Stage-2 plumbing; null is not HPO drift.
5. Combined with frozen B2 (person point Ŷ null) and B4 (traj/rep-distill null): **scalar and daily tabular CGM handoffs are closed as deployable raises over C1** under these recipes.

---

## 6. What B2-V2 is / is not

**Is:**
- Controlled sibling of B2 under daily MSE mid + interval features  
- Evidence that variance pack alone does not flip the null  
- Confirmation of oracle ceiling with reduced 4-vector true CGM  

**Is not:**
- A deployable raise over C1  
- Reopen of frozen `b2_grid_20260715` HPO  
- SSL / deep Stage-1 / B4 sequence cell  
- License to change Path A numbers  

---

## 7. Implications for ladder

| Next | Implication |
|---|---|
| **B3** | Already **frozen** null (`REPORT_B3.md`) |
| B2 / B2-V2 further | **Frozen** unless a *new* plan (e.g. SSL encoder Stage-1) is opened |
| B4 / B4-V2 | Concluded null (`REPORT_B4*.md`, `REPORT_B4_V2.md`); do not reopen without new `PLAN_*` |
| Path B claim ladder | **Complete** under recipes run; residual = optional siblings / SSL |

---

## 8. Reproduce

```bash
# smoke (non-claim)
.venv/bin/python -m training.path_b.b2v2 --run-id b2v2_smoke_YYYYMMDD --quick

# Stage-1 only
.venv/bin/python -m training.path_b.b2v2 --run-id b2v2_s1_YYYYMMDD --stage1-only

# full claim grid
.venv/bin/python -m training.path_b.b2v2 --run-id b2v2_grid_YYYYMMDD \
  --n-trials 50 --stage1-n-trials 20
```

Artifacts: `training/path_b/b2v2/artifacts/b2v2_grid_20260716/`  
(`arm_summaries.json`, `decision_bars.json`, `stage1_metrics.json`, `compare_*.json`, `arms/*/`).
