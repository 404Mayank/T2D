# B1 underperformance audit — bugs, evidence, research

**Date:** 2026-07-14  
**Trigger:** Full λ grid `b1_grid_20260714` near-chance (best test 4-AUC **0.544** vs Path A floor **0.666**); train CE barely moves 1.40→1.38.  
**Sources:** parent verification + `explorer.logic` + `general.worker` dual audit + web research brief.  
**Authority for next actions:** this file + `DECISIONS.md`.

---

## 1. Executive read

B1 is **not** primarily a “multi-task failed” story. The sequence **backbone never learned** (λ=0 val peak 0.564, test ~0.51). Two **confirmed FE/training bugs** make that outcome expected:

1. **Sleep duration unit bug** (ms treated as ns) → sleep ≈ 0; sessionization collapses all nights to one day.  
2. **No input feature scaling** for the LSTM → steps/sedentary dominate; physiological signal washed out.

Literature also says: at **n≈1.8k**, short **T≈7–16**, trees usually beat cold-start RNNs unless inputs are clean and/or person-level summaries are fused. So even after bugfixes, matching Path A is **not guaranteed** — but near-chance with flat CE is **not** an honest ceiling.

---

## 2. Confirmed bugs (severity-ranked)

### C1 — CRITICAL — Sleep duration / session gap: ms ÷ 1e9

**Where:** `pipeline/fe/watch_daily.py` → `_sleep_daily`  
```text
start_ns = s.astype("int64")   # actually milliseconds for datetime64[ms]
dur_h = (end_ns - start_ns) / 1e9 / 3600.0   # ~1e6× too small
gap_m = (...) / 1e9 / 60.0                   # all gaps ≪ 30 min → one session
```

**Evidence:**
| Check | Result |
|---|---|
| Clean sleep dtype | `datetime64[ms, UTC]` |
| int64 delta for 1860 s bout | `1_860_000` (ms) |
| Code `dur_h` | `5.17e-7` h |
| True `dur_h` | `0.517` h |
| Factor | **1e6** |
| `watch_daily` sleep mean | **~6e-5 h** |
| `watch_green` sleep mean | **~5.45 h** (uses `.dt.total_seconds()/3600`) |
| Sessions per pid | **1 day with sleep** for essentially every pid (gap never trips) |
| Non-null sleep days | **~8%** of watch days |

**Also:** `sleep_n_bouts` counts **stage rows** (often 100–700), not true sessions — mislabeled even after unit fix unless redefined.

**Contrast:** `watch_green._sleep_features` and clean `intervals.py` use `total_seconds()` — Path A never hit this bug. CLEANING.md already warns against naive int64 ms→ns paths.

**Impact:** Two of 18 sequence features are garbage; strongest GREEN sleep signal absent from B1.

**Fix:** Use `(end - start).dt.total_seconds() / 3600` (or divide int64 by **1e3** for ms). Recompute gaps the same way. Rebuild `watch_daily`. Redefine `sleep_n_bouts` as **session count** (not stage rows).

---

### C2 — CRITICAL — Watch inputs not standardized for the neural net

**Where:** `training/path_b/b1/data.py` z-scores **only** `glu_cols` (targets). Model is bare `Linear(18→64)` + BiLSTM (`model.py`).

**Evidence (model-facing X after impute):**
| Feature | mean | std | |
|---|---:|---:|---|
| `steps_sum` | ~9e3 | ~6.6e3 | |
| `sedentary_min` | ~1.1e3 | ~280 | |
| `hr_mean` | ~78 | ~10 | |
| `sleep_duration_hours` | ~1e-5 | ~1e-5 | dead |
| **std max/min ratio** | | | **~1e8–1e9** |

Trees (Path A) are scale-invariant. RNNs are not (van Hassel et al. 2025; GRU-D uses z-score).

**Impact:** Gradients dominated by activity counts; CE stays near ln(4)≈1.386.

**Fix:** Train-only per-feature z-score (or robust scale) on `feat_cols`. Optional log1p on heavy tails (`steps_sum`) before z-score. Do **not** min-max heavy-tailed physio.

---

### C3 — CRITICAL (emergent) — Backbone does not learn

**Where:** `artifacts/b1_grid_20260714/lambda_0/history.json`

- train CE 1.399 → 1.380 over 48 epochs  
- val macro-AUC peak 0.564  
- test ~0.51  

Expected once C1+C2 hold. **Diagnostic after fix:** tiny-subset overfit (50 pids) must drive CE near 0; if not, keep debugging.

---

### C4 — HIGH — Sleep NaN → fill 0 + median 0

**Where:** `config.yaml` `fill_zero` includes sleep cols; impute median for sleep is 0.

After C1, ~92% days already “no sleep”; fill 0 makes sleep a near-constant channel.

**Fix:** After unit repair, impute sleep with train median **on days with sleep**, or use missing mask channel (prefer mask + observed value).

---

### C5 — MEDIUM — Daily feature set ≪ GREEN person set

B1 omits person-level constructs that Path A uses: **SRI, RAR, onset SD, multi-day aggregates**. Short T cannot rediscover them from 18 broken/unscaled days at n=1.8k.

**Not a code bug** — design gap. Research supports fusing static summaries with sequence (ShortFuse, late fusion, LSTM→tree).

---

### C6 — LOW/MEDIUM — Class weights inverse-freq → insulin ~0.54–0.62 mass

Pinned by plan; may inflate minority gradients. Unlikely sole cause of chance AUC. Revisit after C1–C2.

---

### C7 — LOW — Activity minutes can exceed 1440

Overlapping Garmin intervals (known Path A residual). Trees tolerate; neural net gets large `sedentary_min`. Cap or normalize by wear minutes later — not first fix.

---

## 3. Suspected (not confirmed as primary)

| ID | Note |
|---|---|
| S1 | `grad_clip=1.0` may throttle updates under unscaled inputs; re-tune after scaling |
| S2 | Truncate prefers earliest CGM-valid days — small bias; OK for now |
| S3 | Ordinal loss (not plain CE) may help severity labels after floor is learnable |
| S4 | MTL λ may still be null after floor rises (teammate + literature: MTL helps rare classes, can hurt common) |

---

## 4. Explicit non-bugs / false alarms

- Glu **target** z-score is correct.  
- `glu_mask` aux gating + non-aux assert OK.  
- `pack_padded` + attention mask OK for watch-valid-only sequences.  
- Outer join then drop non-watch days intentional.  
- Path A GREEN sleep path is fine (do not “fix” GREEN for this).  
- Worse than Path A **can** remain after fixes — literature prior at this n.

---

## 5. Research brief (condensed)

Full source table in session web-researcher output. Load-bearing claims:

1. **DL vs trees at n ≲ thousands on physio/tabular:** trees usually win or tie (Liao 2022; Kinfu 2021; TabZilla 2023; Kuznetsova 2022).  
2. **RNN preprocessing:** z-score mandatory; missingness masks informative (GRU-D Che 2018; Lipton 2016).  
3. **Short T:** daily 7–16 steps often little beyond summaries — sequence must earn its keep via dynamics or fusion.  
4. **MTL/LUPI:** aux can help rare classes and hurt common (Simpson 2018); LUPI transfer claims are contested (Nobari 2025 TMLR). Fix primary task first.  
5. **Hybrids that work:** person static ⊕ sequence embedding; or sequence latents → GBM (Rex 2018; ShortFuse 2017).  
6. **Ordinal severity:** distance-aware / EMD-style losses often beat plain CE once the model learns at all.

### Practical recipe after bugfix

| Do | Don’t |
|---|---|
| Fix sleep FE; rebuild watch_daily | Expect BiLSTM to beat CatBoost cold at n=1.8k without fusion |
| Z-score watch inputs (train stats) | Min-max heavy-tailed physio |
| Tiny-subset overfit test | Tune λ multi-task before λ=0 learns |
| Fuse GREEN (or key person feats) into `z` | Over-parameterize (keep h≤64–128, 1 layer) |
| Re-run λ∈{0,0.5} only for controlled retest | Treat B1 null as B4 kill |

---

## 6. Recommended fix order (for next implement session)

1. **FE:** repair `_sleep_daily` units + session gaps + bout definition → rebuild `watch_daily` (smoke + full).  
2. **Train data:** add train-only StandardScaler on `feature_cols`; keep glu z-score; optional missing masks.  
3. **Sanity:** 50-pid overfit must work; then full λ=0.  
4. **If λ=0 still ≪ 0.60 val:** add GREEN late-fusion to class head.  
5. **Then** re-evaluate multi-task λ∈{0,0.5} under same protocol.  
6. Document outcomes in `REPORT_B1.md` addendum / new run id.

---

## 7. Doc / decision trail

| Doc | Update |
|---|---|
| `DECISIONS.md` | This audit date + locks on root cause |
| `AUDIT_B1_UNDERPERF.md` | This file |
| `REPORT_B1.md` | Pointer: metrics not interpretable until C1+C2 fixed |
| `AGENTS.md` | Index this audit |
