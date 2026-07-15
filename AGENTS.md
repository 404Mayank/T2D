# AGENTS.md — Project File Index

| File | Purpose |
|---|---|
| **T2D.md** | Project objective & north-star (1-pager). |
| **DATA_STRUCTURE.md** | Raw AI-READI layout, schemas, access. |
| **DATA_AUDIT.md** | Empirical data audit + cleaning checklist. |
| **CLEANING.md** | Cleaning/FE pipeline design, config, full-cohort results. |
| **PROCESSED.md** | Processed-data layout and train-time consumer contract. |
| **FEATURES.md** | Feature inventory, leakage rules, literature. |
| **Training.md** | ML methodology (Path A/B, metrics, build order). |
| **COMPUTE.md** | Machines, storage, GPU placement. |

**Supporting:** `pipeline/` (clean/FE code), `convert_pipeline.py` (raw ETL), `audit_data*.py`, `logs/`, `data/full/AI_READI/` (raw), `data/processed/` (outputs, gitignored).

**Training packages:**

| Path | Role |
|---|---|
| `training/path_a_watch/` | Path A **watch-only** GBM floor (scientific claim baseline) |
| `training/path_a_blocks/` | Path A **block ladder** (diagnostics → +onboarding → …); deployable track |
| `training/path_a_blocks/PATH_AHEAD.md` | Raise-floor roadmap & gates |
| `training/path_a_blocks/REPORT.md` | Ladder progress report |
| `training/path_a_blocks/REPORT_A_WRAP.md` | **Path A freeze** + wrap + C1 sensitivity analytics |
| `training/path_a_blocks/PLAN_SENS_C1.md` | Post-freeze smoke/obs/via sensitivities (all bar-fail) |
| `training/path_a_blocks/DECISIONS.md` | Living decisions log for blocks |
| `training/path_b/` | **Path B** (privileged CGM): B1 **frozen** → B2 **frozen** → B4 **A+B concluded** → B3 |
| `training/path_b/b1/` | B1 package (concluded) |
| `training/path_b/b2/` | B2 package (concluded; two-stage tabular) |
| `training/path_b/PLAN_B1_DATA.md` | B1 data readiness + daily FE plan |
| `training/path_b/PLAN_B1_TRAIN.md` | B1 training protocol locks |
| `training/path_b/PLAN_B1_IMPL.md` | B1 code-shape plan |
| `training/path_b/PLAN_B1_FIX.md` | C1/C2 fix plan, gates, critique disposition |
| `training/path_b/PLAN_B2.md` | B2 two-stage plan (implemented) |
| `training/path_b/PLAN_B4.md` | B4 trajectory + rep-distill plan (A+B concluded) |
| `training/path_b/DECISIONS.md` | Path B decisions / acceptance log |
| `training/path_b/REPORT_B1.md` | **B1 final freeze report** |
| `training/path_b/REPORT_B2.md` | **B2 final freeze report** (null predicted; oracle headroom) |
| `training/path_b/REPORT_B4.md` | **B4-A claim report** (traj multi-task null; hybrid < C1) |
| `training/path_b/REPORT_B4_B.md` | **B4-B claim report** (rep-distill null; teacher OK) |
| `training/path_b/REPORT_B4_B_HARD.md` | **B4-B hard-teacher** H1/H2 null (easy-teacher gap closed) |
| `training/path_b/PLAN_B4_B_HARD.md` | Hard-teacher sensitivity plan |
| `training/path_b/AUDIT_B1_UNDERPERF.md` | B1 underperf root-cause audit (sleep unit + scaling + research) |

**Authority:** `DATA_AUDIT.md` → `CLEANING.md` → `PROCESSED.md` → `FEATURES.md` → `Training.md`.  
**Run results authority for Path A raises:** `training/path_a_blocks/REPORT.md` + package `DECISIONS.md` files.  
**Path B authority:** `training/path_b/DECISIONS.md` + per-stage `PLAN_*.md` / `REPORT_*.md`.

---

## Agent process (user locks)

**Loop (non-trivial work):**  
`plan → critique → address → implement → critique → address → run → REPORT/DECISIONS`  
Skip plan/critique only for **small patches** (typo, one-liner, doc-only nits, pure renames). Default is critique.

| Step | Rule |
|---|---|
| **Plan** | Short `PLAN_*.md` under the active package (`training/path_b/`, etc.). Do **not** re-derive Path A/B from scratch — start at this index + authority docs. |
| **Critique** | Subagent **`critiquer`**, model **`opencode-go/glm-5.2:high`**, **fresh** context. Paste the plan/artifact into the task (child has no chat history). Critique **plans and implementations by default**; skip only for small patches (above). |
| **Address** | Parent keeps real blockers/highs; **drop false positives and nits that fight locked decisions**. Log disposition in the plan or `DECISIONS.md`. |
| **Implement** | Surgical; match style; no drive-by refactors. |
| **Run** | Smoke then full; write **REPORT** + **DECISIONS** with numbers and run ids. |
| **Subagents** | Use for isolation/parallel; **parent synthesizes**. Prefer `explorer.logic` for flow bugs, `web-researcher` for external methods. |

**Other standing rules**
- Path B ladder: **B1 (frozen) → B2 (frozen) → B4 A+B (concluded) → B3 last**. Do not silently reorder.
- **Never claim Path A numbers changed** unless Path A was re-run. Path A wrap is frozen.
- Pre-fix / invalidated runs stay labeled as such (e.g. B1 `b1_grid_20260714`); new run ids after protocol/FE fixes — do not overwrite old “success” interpretations.
- Protocol locks in `PLAN_*` / `DECISIONS` are not optional; open a new plan to change them.
- Update docs **as you go**, not only at the end.
