# Path A wrap — consolidated report

**Status:** Path A tabular **frozen**.  
**Batch:** `run_wrap --all` 2026-07-14 (50 trials/family, seed 42, DRI_PRIME=1).  
**Machine pick artifact:** `artifacts/wrap_paper_pick.json`  
**Ranks:** `artifacts/wrap_feature_ranks.json` (dual SHAP+perm from C1; `cestl` excluded)

---

## 1. Frozen recommendation (pre-registered rule applied)

| Role | Choice | Evidence |
|---|---|---|
| **Headline watch-only** | **W0** `full_20260713_221240` | 4-AUC **0.6662**, binary **0.6889** |
| **Secondary tabular (screening-ish)** | **C1 full 1C scores** | minimal_S **fail** retention; minimal_M **fail** retention → rule falls to C1 |
| **Severity-enriched upper** | **E3a severity** (rnl+circ); E3b broader | Neither beats C1 on 4-AUC; report as upper-bound stacks, not new primary |
| **Binary reporting** | **Multiclass-derived `1−P0`** | No E4 run cleared +0.01 over matching derived score |
| **Next** | **Path B** | Privileged CGM / distillation track |

**Deployable stack remains:** watch GREEN + hard onboarding + mood scores (`paidscore`+`cestl` in matrix; PAID carries signal). Comorbidity risk-factor checklist **out** (1B bar fail). Complications = narrative severity proxy only.

---

## 2. Locked ladder (claim numbers)

| Stage | Run | 4-AUC | Binary | Notes |
|---|---|---:|---:|---|
| W0 watch | full_20260713_221240 | **0.6662** | 0.6889 | paper headline |
| 1A onboard | onboarding_20260713_224744 | **0.6987** | 0.7492 | bar pass vs floor |
| 1B core comorb | comorb_core_… | 0.7085 | 0.7781 | **bar fail** |
| 1B +rnl+circ | comorb_plus_complications_… | 0.7240 | 0.7900 | non-claim severity proxies |
| **C1 mood scores** | mood_scores_20260714_014415 | **0.7378** | **0.8309** | **bar pass** vs 1A; secondary pick |

Protocol: n=1824; train/val/test 1277/270/277; balanced weights; val-select; freeze-before-test.

---

## 3. Wrap multiclass experiments

| ID | Exp | Run | n_feat | Family | 4-AUC | Binary | ΔAUC vs C1 | Δbin vs C1 |
|---|---|---|---:|---|---:|---:|---:|---:|
| E2a | paid_only | wrap_paid_only_20260714_024900 | 46 | CatBoost | **0.7366** | **0.8329** | −0.0012 | +0.0020 |
| E2c | ces_only | wrap_ces_only_20260714_025531 | 46 | CatBoost | 0.7030 | 0.7425 | −0.0348 | −0.0884 |
| E1a | minimal_S | wrap_minimal_s_20260714_030243 | 12 | CatBoost | 0.7090 | 0.8126 | −0.0288 | −0.0183 |
| E1b | minimal_M | wrap_minimal_m_20260714_030512 | 18 | CatBoost | 0.7245 | 0.8294 | −0.0133 | −0.0015 |
| E5 | watch_mood | wrap_watch_mood_20260714_030948 | 31 | CatBoost | 0.7179 | 0.8087 | −0.0199 | −0.0222 |
| E3a | severity | wrap_severity_20260714_031346 | 49 | CatBoost | 0.7369 | 0.8287 | −0.0009 | −0.0022 |
| E3b | clinical_upper | wrap_clinical_upper_20260714_032101 | 51 | CatBoost | 0.7357 | **0.8411** | −0.0021 | +0.0102 |

### 3.1 PAID vs CES (construct)

- **paid_only ≈ C1** (ΔAUC −0.001, boot CI includes 0). Dropping `cestl` is free.
- **ces_only collapses** toward 1A (ΔAUC vs C1 −0.035; binary −0.088; both CIs exclude 0 on the low side).
- Confirms 1C perm narrative: **PAID carries the mood block**; CES total is near-null once PAID is present.

### 3.2 Minimal sets (parsimony)

Dual-rank lists (from C1 val SHAP+perm; `cestl` out):

- **S (12):** paidscore, whr_vsorres, fh_dm2sb, fh_dm2pt, rhr, waist_vsorres, hr_mean, bmi_vsorres, weight_vsorres, sri, stress_sd, hr_min  
- **M (18):** S + hip_vsorres, hr_nocturnal_dip, hr_n, sleep_short_frac, age, pulse_vsorres_2  

**Retention rule:** point ΔAUC ≥ −0.01 **and** Δbin ≥ −0.015 **and** not (paired ΔAUC CI entirely &lt; 0).

| | minimal_S | minimal_M |
|---|---|---|
| ΔAUC (min−C1) | **−0.0288** | **−0.0133** |
| Δbin | **−0.0183** | −0.0015 |
| point_auc_ok | False | False |
| point_bin_ok | False | True |
| boot ΔAUC CI | [−0.052, **−0.004**] entirely &lt;0 | [−0.031, +0.005] includes 0 |
| **retain** | **False** | **False** |

**Paper secondary = C1.**  
minimal_M is the better compact model (binary nearly matched; 4-AUC short by ~1.3 pp) but fails the pre-registered point-AUC tolerance. Do not “soft promote” it past the rule.

### 3.3 Watch + mood without onboarding (E5)

- 4-AUC **0.7179** vs C1 **0.7378** (Δ −0.020) and vs 1A **+0.019**.
- Onboarding still buys ~2 pp 4-AUC and ~2 pp binary over watch+PAID alone.
- Mood without onboarding is **not** a substitute for the 1A block.

### 3.4 Severity / clinical upper (E3)

- **E3a** (C1 + kidney + circulation, **no count**): 4-AUC **0.7369** ≈ C1; binary slightly **down**.
- **E3b** (C1 + hbp + clsh + rnl + circ): 4-AUC **0.7357**; binary **0.841** (best binary in wrap multiclass, Δ +0.010 vs C1 binary).
- Adding complication/clinical checklist on top of an already mood-rich C1 does **not** raise 4-class macro-OVR. Earlier 1B+complications lift was vs **1A**, not vs C1.
- Honest framing: severity proxies are optional narrative / binary-side enrichment, **not** a new 4-class ceiling over C1.

### 3.5 Per-class OVR (selected)

| Exp | c0 | c1 | c2 | c3 |
|---|---:|---:|---:|---:|
| C1 (ref) | ~0.831 | — | — | — |
| paid_only | 0.833 | 0.696 | 0.620 | 0.797 |
| minimal_S | 0.813 | 0.677 | 0.633 | 0.713 |
| minimal_M | 0.829 | 0.680 | 0.634 | 0.755 |
| severity | 0.829 | 0.693 | 0.632 | 0.794 |
| clinical_upper | 0.841 | 0.686 | 0.627 | 0.788 |

Class-2 remains weak (~0.60–0.63) across wrap stacks.

---

## 4. Binary-primary experiments (E4)

y = `(label > 0)`; train pos rate **0.614**, val **0.737**, test **0.740**.  
Val rank: binary AUC then AUPRC. Gate: prefer binary-primary if test AUC − multiclass-derived ≥ **+0.01**.

| ID | Exp | Run | n_feat | Test AUC | AUPRC | Derived ref | Δ vs derived | Prefer? |
|---|---|---|---:|---:|---:|---:|---:|---|
| E4a | bin_watch | wrap_bin_watch_…032651 | 30 | 0.6770 | 0.839 | W0 0.6889 | **−0.012** | No |
| E4b | bin_c1 | wrap_bin_c1_…033003 | 47 | 0.8077 | 0.910 | C1 0.8309 | **−0.023** | No |
| E4c | bin_min_s | wrap_bin_min_s_…033405 | 12 | 0.7965 | 0.895 | E1a 0.8126 | **−0.016** | No |
| E4d | bin_severity | wrap_bin_severity_…033648 | 49 | 0.8241 | 0.924 | E3a 0.8287 | **−0.005** | No |

**None** clear the +0.01 gate. Dedicated binary HPO underperforms multiclass-derived `1−P0` on this cohort/split. Paper tables should keep **derived binary** from multiclass models.

---

## 5. Analytical read

1. **Path A ceiling (tabular, this FE):** ~**0.74** 4-AUC / ~**0.83** binary on C1. Wrap did not find a higher honest 4-class stack.
2. **Mood signal is almost all PAID.** CES is a clean negative control; paid_only is operationally equivalent to C1.
3. **Parsimony has a real cost on 4-class.** Top-12 dual-rank drops ~2.9 pp AUC and fails retention hard. Top-18 nearly holds binary but still misses the −0.01 AUC band.
4. **Onboarding is not redundant with mood.** E5 proves ~2 pp of C1’s lift over watch+PAID is anthropometrics/FH/BP.
5. **Complications don’t stack on C1.** The earlier severity story is “vs 1A without mood,” not “above full 1C.”
6. **Binary objective is not free performance.** With balanced weights and the same features, binary HPO lost to multiclass-derived scores — useful null for the paper.
7. **Class-2 (oral med) remains the 4-class bottleneck.** Ends of spectrum (0 vs 3) drive macro-OVR; wrap features don’t fix the middle.

---

## 6. Mandatory limitations

1. **Minimal features chosen from C1 val importances** → optimistic bias toward retention on the same test split; still **failed** retention, so the bias did not invent a false win.
2. **Re-HPO on every feature subset** confounds feature count with HPO winner / family noise (all selected CatBoost here, but params differ).
3. **Complications / clinical flags** are severity or comorbidity **proxies**, not pure prospective screening risk factors; E3 is labeled upper-bound, not claim primary.
4. Fixed person split (not nested CV); bootstrap CIs are within-split uncertainty, not external validity.
5. No sex/race; site never a feature; smoking absent from onboarding release.

---

## 7. Artifact index (production wrap runs)

| Exp | Artifact dir |
|---|---|
| ranks | `wrap_feature_ranks.json` |
| paper pick | `wrap_paper_pick.json` |
| E2a | `wrap_paid_only_20260714_024900` |
| E2c | `wrap_ces_only_20260714_025531` |
| E1a | `wrap_minimal_s_20260714_030243` |
| E1b | `wrap_minimal_m_20260714_030512` |
| E5 | `wrap_watch_mood_20260714_030948` |
| E3a | `wrap_severity_20260714_031346` |
| E3b | `wrap_clinical_upper_20260714_032101` |
| E4a–d | `wrap_bin_{watch,c1,min_s,severity}_20260714_*` |

Smoke dirs (`wrap_smoke_*`) are non-claim.

---

## 8. Code entry points

```bash
export DRI_PRIME=1
.venv/bin/python -m training.path_a_blocks.build_minimal_ranks
.venv/bin/python -m training.path_a_blocks.run_wrap --exp paid_only
.venv/bin/python -m training.path_a_blocks.run_wrap --all
```

Plan: `PLAN_A_WRAP.md`, `PLAN_A_WRAP_IMPL.md`. Decisions: `DECISIONS.md`.

---

## 9. Handoff → Path B

Path A tabular is done. Next work is **Path B** (privileged CGM / teacher–student / distillation), not more survey blocks or diet in this package. Do not move the W0 headline or re-open 1B as bar-pass without a new pre-registered protocol.
