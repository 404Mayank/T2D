# PLAN_B1_GS — gradient-balanced multi-task retry (not a B1 reopen)

**Date:** 2026-07-16  
**Status:** **concluded** — claim run `b1gs_grid_20260716`; see `REPORT_B1_GS.md`.  
**Role:** new formulation under a sibling plan — frozen plain-λ B1 stays closed.  
**Authority:** `REPORT_B1.md` (freeze), `AUDIT_B1_UNDERPERF.md`, `PLAN_B1_TRAIN.md` /
`PLAN_B1_FIX.md` (C1+C2 locks), `REVIEW_PHASES.md` (naive-λ gap), `Training.md` §4 B1.  
**Outputs (done):** implement → smoke/overfit → claim grid → `REPORT_B1_GS.md` +
`DECISIONS.md`. **Do not overwrite** `REPORT_B1.md`. Sections below retain pre-run protocol
language for auditability; execution results live only in the REPORT.

**Prior under weak watch→glucose SNR:** expected class Δ is null; the scientific deliverable
is still the **conflict + glu-head probe** that distinguishes data null from untried-formulation
excuses. GS effort is justified as closing the REVIEW_PHASES gap, not as a high-P(lift) bet.

---

## 0. Why this is a retry, not a reopen

Frozen B1 (`b1_grid_20260715_fix`) is a **null of one recipe**:

\[
L = L_{\mathrm{CE}} + \lambda L_{\mathrm{glu}},\quad \lambda \in \{0, 0.5\}
\]

with only `clip_grad_norm_`. Paired boot ΔAUC(λ0.5−λ0) ≈ −0.0003, CI lo ≯ 0.
`PLAN_B1_TRAIN` §3.3 deferred GradNorm / uncertainty “only if fixed-λ shows clear task
conflict” — but **conflict was never measured** before freeze. `REVIEW_PHASES` names this
as the single most defensible “we didn’t really try MTL” gap.

**This plan replaces plain-λ with gradient-conflict-aware / uncertainty-aware balancing.**
Same backbone, same C1+C2 data contract, new run ids. Pre-fix `b1_grid_20260714` stays
**invalid** (sleep ms/ns + no scale) — never cite as ceiling.

---

## 1. FE / cleaning judgment (step 1 of prompt loop)

| Question | Verdict |
|---|---|
| Re-clean? | **No** — `clean/*` + pools unchanged and sufficient |
| Rebuild `watch_daily` / `cgm_daily`? | **No** for primary GS recipe — post-fix sleep OK |
| C1 sleep units still hold? | **Yes** — mean sleep **6.64 h**, non-null frac **0.72**, days/pid median **12** |
| C2 train-only feat z-score? | **Keep enforce** in `b1/data.py` (do not reintroduce raw-scale LSTM) |
| Daily feature enrichment (C5)? | **Out of primary scope** — open only if GS MTL still null *and* diagnostics suggest encoder information-starved (see §8) |
| Missing-mask channels? | **Not primary** — optional sensitivity if G1/G2 regress vs freeze floor |

**Inspected (2026-07-16):** `watch_daily` 22844×24 / 1824 pids; `cgm_daily` 19805×12;
core split 1277/270/277; train labels {0:493,1:311,2:393,3:80}; aux concurrent both-valid
days median **11**. Contract matches `PROCESSED.md` Path B daily section.

**Conclusion:** data already sufficient → proceed to GS training plan (no FE plan / no FE
critique).

---

## 2. Goal & scientific claim

**Claim (only):** on identical `attn_lstm_64` + C1+C2 inputs, does **gradient-balanced**
day-level CGM multi-task beat **class-only (λ=0)** on test 4-class macro-OVR AUC?

Pre-registered comparator: **paired person bootstrap ΔAUC vs λ=0**, CI lo **> 0**
(n=2000, seed 42) — same rule as frozen B1.

**Not the claim:** beat Path A W0 (0.666) or C1 (0.738). Those are **informational**
context only (trees beat cold DL at n≈1.8k is a known prior; freeze pure-seq = 0.652).

**Secondary science question (must answer even if class Δ null):** did task-gradient
**conflict** exist under plain-λ, and did balancing change glu-head quality / conflict
stats? A null with **no conflict + dead glu head** is a **data null** (watch→glucose SNR
wall; B2 R²≈0.05 / B4 Pearson~0.25 priors). A null with **conflict resolved but class
still flat** is a **formulation-exhausted-for-this-backbone** null. Do not collapse these.

---

## 3. Locked constants (inherit freeze; do not re-litigate)

| Item | Lock |
|---|---|
| Backbone | `attn_lstm_64` (BiLSTM + mask attention, h=64, dropout 0.2) |
| Glu head | day-level Linear on \(h_t\) → 8 CGM stats; masked MSE |
| Features | 18 daily dims (no `hr_n`/`stress_n`/`rr_n`) — same `config.yaml` list |
| Pool / split | wearable_core 1824; train/val/test **1277/270/277**; fixed `recommended_split` |
| Glu pool | `wearable_core ∧ aux_eligible` ∧ watch_valid ∧ cgm_valid days only |
| Impute / scale | C1+C2: sleep duration **not** fill_zero; train-only feat z-score; glu target z-score train-aux |
| Class weights | **Inverse-freq on train core, sum-normalize** — **pinned before any GS arm** (same recipe as freeze). No focal / re-tune in claim grid (avoids confounding MTL Δ). Log exact weight vector in run meta. |
| Seed / ES | seed 42 per arm; ES val macro-OVR AUC patience 15; AdamW 1e-3, wd 1e-4, clip 1.0 |
| Pad / T | max_len 16; truncate prefer cgm-valid earliest (existing) |
| Device | CUDA/ROCm; `cudnn.enabled=False` on gfx1010 |
| GREEN fuse / surveys | **Off** for claim grid (freeze already showed GREEN late-fuse no raise; C1 stack is Path A’s claim) |
| CORN / ordinal head | **Not in claim grid** (would confound MTL ablation). Optional footnote arm only after GS claim written — see §8 |
| λ plain grid reopen | **No** — plain λ=0.5 is a **single reference arm** for co-located preds + conflict probe, not a re-sweep of {0.3,1.0} |

---

## 4. Formulation — balancing methods

### 4.1 Shared loss pieces

- \(L_{\mathrm{CE}}\): weighted cross-entropy (pinned class weights).
- \(L_{\mathrm{glu}}\): masked MSE on z-scored 8-vector (same `masked_mse` as freeze).
- If batch glu-mask mass = 0: skip glu term for that batch (log count); still step CE.

### 4.2 Arms (pre-registered)

| Arm id | Description | Role |
|---|---|---|
| **A0** | λ=0 class-only (glu forward OK, loss zeroed — capacity match cosmetic) | **control** |
| **A_plain** | \(L = L_{\mathrm{CE}} + 0.5\, L_{\mathrm{glu}}\) | plain-λ **reference** (repro freeze; conflict probe host) |
| **A_pcg** | PCGrad on task grads \(\{g_{\mathrm{CE}}, g_{\mathrm{glu}}\}\) over **shared** params; head-only grads unprojected | **primary GS** |
| **A_uw** | Kendall/Gal **homoscedastic uncertainty** weights: learn \(s_{\mathrm{ce}}, s_{\mathrm{glu}}\) (log-variance) | **primary GS** (scale-balance complementary to PCGrad) |
| **A_pcg_uw** | PCGrad **on top of** uncertainty-weighted task losses | **optional stack** — run only if (a) conflict probe shows frequent negative cosine **and** (b) A_uw alone leaves residual conflict; else skip to save GPU |
| **A_gn** | GradNorm on shared last layer | **conditional** — only if §5 conflict probe shows **magnitude imbalance** (‖g_glu‖ ≫ ‖g_ce‖ or vice versa by >10× median) **and** A_pcg / A_uw both fail class Δ. Not default. |

**Default claim set:** `{A0, A_plain, A_pcg, A_uw}`. Stack / GradNorm only under gates above.

### 4.3 PCGrad (Yu et al., NeurIPS 2020) — implementation locks

- Compute \(L_{\mathrm{CE}}\) and \(L_{\mathrm{glu}}\) separately; two `autograd.grad` passes
  (or retain_graph) on the **shared** parameter list only.
- **Shared params (locked after critique):** `self.input`, `self.lstm`, `self.proj`.
  - **Not shared:** `self.attn` (CE-only — scores → α → z → class_head; glu uses `h_t`
    directly), `self.class_head` (CE-exclusive), `self.glu_head` (glu-exclusive).
  - Rationale: including `attn` zeros out glu entries there and **inflates ‖g_CE‖**, which
    systematically **deflates** cos and can mis-fire the ≤5% “data null” conflict gate.
- For each pair with \(\langle g_i, g_j \rangle < 0\), replace
  \(g_i \leftarrow g_i - \frac{\langle g_i,g_j\rangle}{\|g_j\|^2} g_j\) (standard PCGrad).
- **|T|=2 convention (impl lock):** project each task onto the **original** peer
  gradient (not sequential mutate of the other). Sum is order-invariant; coin-flip
  order retained for API parity only. Document in REPORT.
- After surgery: write grads onto shared params; head (+attn) grads = plain task grads; then
  `clip_grad_norm_(all, 1.0)`; optimizer step.
- **Base scale:** use unweighted \(L_{\mathrm{CE}}\) and \(L_{\mathrm{glu}}\) (no extra λ) so
  PCGrad is the only interaction mechanism. Log mean cos and conflict rate.
- **Zero glu-mask mass batch:** if `L_glu` is zero / `g_glu` is all-zero → **skip PCGrad**;
  update shared params with `g_CE` only. Do **not** count these steps in the conflict-rate
  **denominator** (see §5).
- ~40–80 LOC hook in `training/path_b/b1/balance.py`; **no** third-party PCGrad package.
  Reference impls OK for unit check.

### 4.4 Uncertainty weighting (Kendall & Gal / Kendall et al. CVPR 2018) — locks

For classification + regression (homoscedastic form, **with explicit CE-primary prior**):

\[
L = e^{-s_{\mathrm{ce}}} L_{\mathrm{CE}} + s_{\mathrm{ce}}
  + \tfrac12 e^{-s_{\mathrm{glu}}} L_{\mathrm{glu}} + \tfrac12 s_{\mathrm{glu}}
\]

**Departure from fully symmetric Kendall/Gal (both tasks use \(\tfrac12\)):** CE log-var
regularizer coefficient is **1.0** (not 0.5) so CE stays anchored as primary; glu keeps
\(\tfrac12\) (Gaussian NLL shape). At init \(s=0\): \(L = L_{\mathrm{CE}} + 0.5 L_{\mathrm{glu}}\),
matching A_plain’s λ=0.5 initial point by design. Cite as “UW with CE-primary prior,” not
vanilla Kendall/Gal if reviewers ask.

Implement \(s = \log \sigma^2\) as unconstrained `nn.Parameter` (init 0.0).

- Clamp \(s\) to \([-5, 5]\) for numerical safety (log if clamp hits).
- **Must log** per-epoch mean `s_ce`, `s_glu`, and effective weights
  \(w_{\mathrm{ce}}=e^{-s_{\mathrm{ce}}}\), \(w_{\mathrm{glu}}=\tfrac12 e^{-s_{\mathrm{glu}}}\)
  in history.json — required to diagnose UW degeneration / clamp pegging.
- **Separate param-group LR for `s_*`:** `lr_s = 0.1 ×` backbone LR (default 1e-4 when
  backbone is 1e-3). Prevents premature `s_glu` saturation under constant regularizer
  grads before the glu head can learn (critical given freeze best-epoch=1 pattern).
- **Do not** also multiply by fixed λ=0.5 (double-counting). UW replaces λ.
- **Zero glu-mask mass batch:** still apply CE term + both `+s` regularizers; glu data term
  is 0. Note: repeated zero-glu batches push `s_glu` up (weight → 0) — this is *valid*
  UW (“glu uninformative”) but can make A_uw ≈ A0; REPORT must call out if `s_glu` pegs
  at clamp without ever seeing useful glu learning on glu-active batches.
- PCGrad+UW (optional arm): apply PCGrad to the **weighted** task losses
  \(e^{-s}L\) (without the `+s` regularizer in surgery inputs; add `+s` to total after).
  Conflict pattern on A_plain **does not transfer** to UW-weighted grads — re-log probes
  per arm.

### 4.5 GradNorm (conditional only)

If triggered: GradNorm on shared representation layer (`proj` or BiLSTM output linear),
target rate α=1.5 default, restore weights each epoch mean. Document if run; not part of
default claim table.

### 4.6 What is held fixed across arms

Same: features, split, backbone hyperparams, optimizer family, seed, ES, class weights,
data bundle (one `build_sequences` per run). Only **balancing formulation** changes
(plus UW’s extra `s_*` params / lower LR group when that arm is active).
Seed **reset before each arm** (same as λ loop).

**ES × best-epoch interaction (acknowledged):** freeze full runs often peak val AUC at
epoch 1. GS that needs >1 epoch for glu grads to shape shared `h_t` may still save ep1
if that remains the val peak. We **keep ES parity** with freeze (do not change patience
or metric). REPORT must table best_epoch per arm; if all arms still ep1, note that GS
had little wall-clock to act on the *selected* checkpoint even if later epochs moved glu.

---

## 5. Conflict & glu-head diagnostics (required, not optional)

Run on **A_plain** (and log for every multi-task arm):

| Probe | Definition | Why |
|---|---|---|
| **Conflict rate** | fraction of **glu-active** train steps with \(\cos(g_{\mathrm{CE}}, g_{\mathrm{glu}}) < 0\) on **shared** params (`input`/`lstm`/`proj` only) | Exists-conflict? |
| **Mean cos** | mean cosine over **glu-active** steps (epoch-end dump) | Severity |
| **Grad norms** | median ‖g_CE‖, ‖g_glu‖, ratio on glu-active steps | Magnitude imbalance → GradNorm gate |
| **n_glu_active_steps** | count of steps included in conflict denom | Denominator transparency |
| **Glu quality** | val/test masked MSE (z); optional Pearson on `cgm_mean` if cheap | Non-degeneracy |
| **CE dynamics** | train CE path vs freeze A0 | Sanity |
| **UW trajectory** | per-epoch `s_*` + effective weights (A_uw / stack only) | Degeneration check |

**Denominator lock:** conflict rate uses only steps with **non-zero glu-mask mass**.
Zero-glu batches are excluded (not counted as “no conflict”). Log both
`n_glu_active_steps` and `n_train_steps` so the fraction of excluded steps is visible.

**Glu-head quality references (three levels, not one “dead” number):**

| Reference | Approx z-MSE | Meaning |
|---|---:|---|
| Random untrained head (A0 freeze) | ~1.51 test | init noise |
| **Constant predictor** (predict train mean 0 after z-score) | **1.0** | “learned nothing useful” floor |
| Freeze A_plain | ~1.43 test | barely better than random, **worse than constant** |

A multi-task arm is **glu-alive** only if val z-MSE **≤ 0.95** (beats constant by a small
margin) **or** day-level Pearson(`cgm_mean`) **> 0.10** on val aux days. Soft flag only —
does not kill the class claim, but REPORT language must not call MSE 1.0–1.4 “learned.”

**Interpretive gates (report language, not kill switches):**

| Band | Condition | Lean |
|---|---|---|
| **Low conflict / dead aux** | conflict rate **≤ 5%** on glu-active steps **and** glu not alive (MSE ≳ 1.0) | **Data null** — no conflict to fix; watch→glucose SNR wall |
| **Moderate conflict** | conflict rate **(5%, 20%)** | **Ambiguous band** — report rate + whether GS reduced conflict; if class Δ null, say “intermittent conflict, transfer still null” (not pure data-null, not pure formulation-exhausted) |
| **High conflict / imbalance** | conflict **≥ 20%** **or** median ‖g‖ ratio >10×; after PCGrad/UW conflict drops but class Δ CI lo ≯ 0 | **Balancing-tried, transfer still null** on this backbone |
| **GS help** | glu-alive **and** class Δ CI lo > 0 | **Positive multi-task claim** |

Also log freeze prior: A_plain glu test z-MSE **1.43** vs A0 **1.51** — tiny / sub-constant
glu learning under plain-λ. GS must clear the **constant-predictor** bar to argue the glu
head had something to learn.

---

## 6. Metrics & decision rules

### 6.1 Metrics (same as B1 freeze)

- Test **raw** 4-class macro-OVR AUC + macro AUPRC (claim ranking)
- Binary healthy-vs-not AUC (`1−P0`)
- Per-class OVR AUC (watch class-2)
- Val isotonic calibration diagnostic (Brier); claim = raw
- Glu: val/test masked MSE/MAE (z); n_glu elements
- Conflict probes (§5)

### 6.2 Bootstrap

- Unit: **person**; n=2000; seed 42; paired vs **A0** for every multi-task arm
- Co-locate test proba under one run id so paired masks match

### 6.3 Decision rules (pre-register)

| Question | Rule |
|---|---|
| Backbone still learnable? | G1 overfit50: train CE drops hard (≲1.0); not flat near ln4 |
| A0 floor intact? | Full A0 val macro-AUC **≥ 0.60** (freeze ~0.68); if fail → stop, debug C1+C2 regression |
| Glu head non-degenerate (soft)? | **Glu-alive** per §5 (val z-MSE ≤ 0.95 **or** Pearson>0.10). Else flag “glu head still near-null / sub-constant” |
| **GS multi-task helps?** | For arm X ∈ {A_pcg, A_uw, …}: test paired ΔAUC(X−A0) boot **CI lo > 0** |
| Beat W0 / C1? | Informational only; do not gate claim |
| Open C5 enrichment? | Only if **all** default multi-task arms fail class Δ **and** (§5 low-conflict **or** moderate-conflict with glu not alive) **and** pure-seq A0 test 4-AUC ≲ freeze 0.652 + 0.01 — then **separate** plan. Operational trigger = “GS null + glu not alive + no high conflict” (information-starvation / SNR wall, not untried balancing). |
| Open GradNorm / stack? | Per §4.2 gates only |
| CORN later? | Separate footnote after REPORT; not in claim Δ |

**Multiple-arm note:** report **all** default arms; do not pick best-of-N as sole claim.
If both A_pcg and A_uw pass, say so; if only one, say which. Selection-biased “best arm”
checkpoint may be noted but must be labeled.

---

## 7. Package / code shape

Prefer **surgical extension of `training/path_b/b1/`** (reuse data/model/eval; avoid
copy-paste drift of C1+C2):

```
training/path_b/
  PLAN_B1_GS.md          # this file
  REPORT_B1_GS.md        # after claim run
  b1/
    balance.py           # NEW: pcgrad_step, UncertaintyWeights, conflict stats
    train.py             # extend: balance mode hook
    run.py               # CLI: --balance none|pcgrad|uncertainty|pcgrad_uw
    config.yaml          # optional balance: section; default none
    ...
```

CLI sketch:

```bash
# smoke plumbing
.venv/bin/python -m training.path_b.b1 --run-id b1gs_smoke --quick --device cuda \
  --balance pcgrad --arms 0,plain,pcgrad

# overfit gate (class-only, same as freeze G1)
.venv/bin/python -m training.path_b.b1 --run-id b1gs_overfit50 --max-participants 50 \
  --lambdas 0 --device cuda

# claim grid (user go)
.venv/bin/python -m training.path_b.b1 --run-id b1gs_grid_YYYYMMDD --device cuda \
  --balance-grid default   # expands to A0,A_plain,A_pcg,A_uw
```

Exact flag names free at impl time; contract is **one run id co-locates all claim arms**.

Artifacts:

```
b1/artifacts/<run_id>/
  summary.json
  data_diag.json
  arm_a0/  arm_plain/  arm_pcg/  arm_uw/   # history, best.pt, metrics, preds
  conflict/   # epoch cos / rate dumps
  bootstrap_vs_a0.json
```

---

## 8. Scope fences & deferred levers

**In scope:** PCGrad, uncertainty weighting, conflict diagnostics, plain-λ reference,
optional PCGrad+UW / GradNorm under gates, smoke + overfit + (user go) claim grid,
`REPORT_B1_GS.md` + DECISIONS.

**Out of scope (explicit):**
- Reopening plain λ∈{0.3,1.0} science table
- GREEN / C1 survey fusion as claim
- Daily feature enrichment (C5), SSL pretrain, h=128, 5-min grid, B2/B4/B3
- CORN / focal loss in claim grid (lock imbalance first; ordinal is orthogonal story)
- Claiming Path A numbers changed

**If GS claim is null — ordered next options (new plans, not silent):**
1. Document data-null vs formulation-null using §5 probes (required in REPORT).
2. Optional CORN head on **A0 only** (class objective; no MTL confound) — separate plan.
3. C5-lite: late-fuse a **small** GREEN subset (SRI/RAR/onset_sd) into \(z\) **with** best
   GS arm — only if encoder starvation suspected; freeze already tried full GREEN late-fuse
   on λ=0 and failed, so expect low prior unless combined with better MTL.
4. Ladder remains B3 last for logit-KD; do not reorder.

---

## 9. Named caveats to carry (verify in run, don’t assume)

From freeze + review + data inspect — surface in REPORT even if inconvenient:

1. **Watch→glucose SNR wall (external).** B2 Stage-1 R²≈0.05; B4 wear→curve Pearson
   ~0.25. If GS glu head stays near MSE~1.5 z, MTL cannot mint signal that isn’t there.
2. **C5 design gap.** 18 daily dims omit SRI/RAR/onset SD; short T may not rediscover them.
   Freeze GREEN fuse failed for pure seq — still a ceiling risk for *sequence* content.
3. **Best epoch = 1 pattern** on freeze full runs (val peaks early, mild train-fit drift).
   Watch whether GS changes ES dynamics; if still ep1 everywhere, capacity/signal issue
   dominates optimizer choice.
4. **Class-2 bottleneck** (~0.60 OVR). Balancing may not fix label-geometry; CORN is the
   orthogonal lever.
5. **Insulin weight mass ~0.5** under inverse-freq (train n_3=80). Locked for fair Δ; may
   still distort CE vs glu scale — UW partially addresses scale, not class prior.
6. **Truncate prefers earliest CGM-valid days** — small selection bias; keep for parity
   with freeze (changing it confounds GS vs freeze A0).
7. **`sedentary_min` can exceed 1440** (overlapping Garmin intervals). Trees tolerate;
   z-scored LSTM sees heavy tails — known residual, not a new bug.
8. **Site-local FE already more UAB-correct than frozen Path A GREEN** — A0 vs W0 is still
   not architecture-matched; do not over-read informational floor gaps.

Open-ended: if impl/smoke surfaces **new** leakage, mask, or coverage issues not listed,
add them to DECISIONS + REPORT rather than silent workarounds.

---

## 10. Run ladder

| Step | Action | Gate | Claim? |
|---|---|---|---|
| R0 | Critique this plan → address real blockers | disposition logged §12 | no |
| R1 | Implement `balance.py` + train/run hooks; unit test PCGrad on 2-toy grads (conflict projects; aligned unchanged); unit zero-glu skip | unit pass | no |
| R2 | `--quick` smoke all default arms end-to-end; assert conflict denom >0 on fullish smoke if aux present; dump `s_*` for A_uw | finishes; finite losses | no |
| R3a | Overfit50 **A0** (no `--quick`) | CE drop ≲1.0 (G1) | no |
| R3b | Overfit50 **A_pcg** (no `--quick`) | CE still drops ≲1.0 — PCGrad must not break CE convergence | no |
| R4 | **Present results of R1–R3; stop for user go** | — | no |
| R5 | Full claim grid `b1gs_grid_YYYYMMDD` arms {A0,A_plain,A_pcg,A_uw} | A0 val≥0.60; boot table; conflict + glu-alive table | **yes** |
| R6 | Conditional A_pcg_uw / A_gn | only if §4.2 gates | yes if run |
| R7 | `REPORT_B1_GS.md` + DECISIONS + pointer in Training.md status line | docs | — |

**User lock from prompt:** do **not** run R5 claim grid until user says so. Smoke + overfit
(R2–R3) OK to validate plumbing after critique.

**Optional cheap probe (not required):** if impl time is tight, a conflict-only hook on
A_plain can be run before polishing UW — but default path implements both primary GS arms
because REVIEW_PHASES treats the untried balancing *family* as the gap to close.

---

## 11. Docs to update as we go

| Doc | When |
|---|---|
| `PLAN_B1_GS.md` | critique disposition (§12); any lock changes |
| `DECISIONS.md` | FE sufficiency (done at plan); impl locks; run outcomes |
| `REPORT_B1_GS.md` | after R5 (sibling of freeze REPORT_B1 — do not overwrite) |
| `Training.md` §4 B1 line | short pointer: freeze stands; GS retry status |
| `AGENTS.md` | add PLAN/REPORT_B1_GS to index when claim written |

---

## 12. Critique disposition

Fresh `critiquer` (`opencode-go/glm-5.2:high`, run `c21253dc`). Verdict was **revise**
(no blockers). Parent disposition:

| ID | Severity | Critiquer point | Disposition |
|---|---|---|---|
| C1 | medium | `self.attn` is CE-only; including it in “shared” deflates cos | **Accept** — shared = `input`/`lstm`/`proj` only (§4.3) |
| C2 | medium | Zero-glu batches: PCGrad fallback + conflict denom + UW drift unspecified | **Accept** — skip PCGrad / exclude from denom; document UW zero-glu → weight collapse (§4.3–4.4, §5) |
| C3 | low | UW CE regularizer coeff 1.0 ≠ symmetric Kendall/Gal | **Accept as design** — document CE-primary prior + init=λ0.5 (§4.4) |
| C4 | medium | No separate LR for `s_*` → premature clamp | **Accept** — `lr_s = 0.1 ×` backbone LR; log `s_*` trajectory (§4.4) |
| C5 | medium | Overfit only on A0; PCGrad dynamics unchecked | **Accept** — add R3b overfit50 A_pcg (§10) |
| C6 | medium | Dead zone 5–20% conflict; “dead” MSE confuses random vs constant | **Accept** — moderate band + glu-alive vs constant MSE=1.0 (§5) |
| C7 | low | Measure conflict before implementing GS | **Reject as required path** — optional note only; family gap justifies implementing both primary arms (§10) |
| M1 | — | Missing `s_*` logging | **Accept** (with C4) |
| M2 | — | ES ep1 × GS interaction | **Accept document** — keep ES parity; table best_epoch (§4.6) |
| M3 | — | C5 “starvation” not operationalized | **Accept** — operational trigger in §6.3 |
| — | low/prior | Expected null under B2/B4 SNR priors | **Accept framing** — status line + success §13 already allow data-null outcome |

**Dropped / not fighting locks:** no change to inverse-freq weights, backbone, split,
C1+C2, no Path A reopen, no CORN in claim grid, no plain-λ re-sweep.

---

## 13. Success definition (one paragraph)

GS retry succeeds scientifically if we either (a) obtain a multi-task arm with test paired
ΔAUC vs A0 whose bootstrap CI lo > 0 under pinned protocol, **or** (b) cleanly show that
after measuring conflict and applying PCGrad + uncertainty weighting, class ranking still
does not lift — with glu-head / conflict probes distinguishing **data null** from
**untried-formulation** excuses. Plumbing success (R2–R3) is necessary but not the claim.
)
