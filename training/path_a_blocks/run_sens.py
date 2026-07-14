"""C1 sensitivity experiments: smoking / obesity / via (+ joint).

See PLAN_SENS_C1.md. Parent always original C1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import (
    block_tags_wrap,
    load_watch_onboarding_mood,
    make_block_splits,
    null_report,
)
from training.path_a_blocks.diagnostics import bootstrap_ci, paired_delta_bootstrap
from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.evaluate import write_json
from training.path_a_watch.explain import permutation_on_val, shap_summary
from training.path_a_watch.hpo import pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_a_watch.models import predict_proba, resolve_lgbm_device

EXPS = {
    "smoke": ["smoke_ever", "smoke_current"],
    "obs": ["mhoccur_obs"],
    "via": ["via1", "via2", "via3"],
    "all3": ["smoke_ever", "smoke_current", "mhoccur_obs", "via1", "via2", "via3"],
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _git_hash(repo: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]


def _strip(pack: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pack.items() if k not in ("model", "val_report")}


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def assert_parent_c1(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["parent_c1_run_id"]
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    path = art / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["parent_c1_reference"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"parent_c1 {k}: config {ref[k]} != artifact {v}")
    freeze = json.loads((art / "selected_model.json").read_text())
    model_path = art / "models" / "selected.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        **checks,
        "family": m.get("selected_family"),
        "feature_cols": list(freeze["feature_cols"]),
        "model_path": str(model_path),
        "run_id": run_id,
    }


def load_parent_c1_proba(
    repo: Path,
    cfg: dict[str, Any],
    c1: dict[str, Any],
    test_pids: np.ndarray,
    *,
    tol: float = 1e-6,
) -> np.ndarray:
    df, _, _, _, full = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=cfg["data"]["onboarding_keep"],
        mood_cols=list(cfg["data"]["mood_scores"]),
        expected_n=int(cfg["data"]["expected_n"]),
    )
    feat = list(c1["feature_cols"])
    X = df.set_index("person_id")[feat].loc[list(test_pids)].reindex(columns=feat)
    if list(X.columns) != feat:
        raise AssertionError("C1 col order mismatch")
    model = joblib.load(c1["model_path"])
    proba = predict_proba(model, X)
    y = df.set_index("person_id").loc[list(test_pids), "label"].to_numpy(dtype=np.int64)
    recomp = macro_ovr_auc(y, proba)
    if abs(recomp - c1["test_macro_ovr_auc"]) > tol:
        raise AssertionError(f"C1 recompute {recomp} != {c1['test_macro_ovr_auc']}")
    return proba


def build_matrix(
    exp: str,
    repo: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    addon = list(EXPS[exp])
    df, w, o, m, base_feat = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=cfg["data"]["onboarding_keep"],
        mood_cols=list(cfg["data"]["mood_scores"]),
        expected_n=int(cfg["data"]["expected_n"]),
    )
    # ensure via present (mood table has them even if not in mood_cols load)
    mood_path = _resolve(repo, cfg["paths"]["mood"])
    md = pd.read_parquet(mood_path)
    need_via = [c for c in addon if c.startswith("via")]
    if need_via:
        miss = [c for c in need_via if c not in md.columns]
        if miss:
            raise ValueError(f"mood missing {miss}")
        extra = md[["person_id"] + need_via].drop_duplicates("person_id")
        # may already be in df if we loaded them — only merge missing
        for c in need_via:
            if c not in df.columns:
                df = df.merge(extra[["person_id", c]], on="person_id", how="left")

    if any(c.startswith("smoke") for c in addon):
        smoke_path = repo / "data/processed/features/smoking.parquet"
        if not smoke_path.exists():
            raise FileNotFoundError(
                f"missing {smoke_path}; run: python -m training.path_a_blocks.build_smoking_features"
            )
        sm = pd.read_parquet(smoke_path)
        need = [c for c in addon if c.startswith("smoke")]
        miss = [c for c in need if c not in sm.columns]
        if miss:
            raise ValueError(miss)
        sm = sm[["person_id"] + need].drop_duplicates("person_id")
        n0 = len(df)
        df = df.merge(sm, on="person_id", how="left")
        if len(df) != n0:
            raise AssertionError("smoke merge changed n")

    if "mhoccur_obs" in addon:
        comb = pd.read_parquet(_resolve(repo, cfg["paths"]["comorbidity"]))
        if "mhoccur_obs" not in comb.columns:
            raise ValueError("mhoccur_obs missing")
        comb = comb[["person_id", "mhoccur_obs"]].drop_duplicates("person_id")
        n0 = len(df)
        df = df.merge(comb, on="person_id", how="left")
        if len(df) != n0:
            raise AssertionError("obs merge changed n")

    feature_cols = list(base_feat) + [c for c in addon if c not in base_feat]
    for c in feature_cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature {c}")

    tags = block_tags_wrap(w, o, m)
    for c in addon:
        if c.startswith("smoke"):
            tags[c] = "smoking"
        elif c == "mhoccur_obs":
            tags[c] = "comorbidity"
        elif c.startswith("via"):
            tags[c] = "vision"
        else:
            tags[c] = "addon"
    return {
        "df": df,
        "feature_cols": feature_cols,
        "addon_cols": list(addon),
        "tags": tags,
        "watch_cols": w,
        "onboard_cols": o,
        "mood_cols": m,
        "base_feat": base_feat,
    }


def run_one(
    exp: str,
    *,
    cfg: dict[str, Any],
    repo: Path,
    run_id: str,
    quick: bool,
    n_trials: int | None,
    skip_shap: bool,
) -> int:
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    art.mkdir(parents=True, exist_ok=True)
    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError(f"refuse overwrite {art}")

    if quick:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = 2
        skip_shap = True
    if n_trials is not None:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = int(n_trials)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    t0 = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"run_id={run_id} sens={exp}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")

    parent = assert_parent_c1(repo, cfg)
    log(f"parent C1 ok auc={parent['test_macro_ovr_auc']:.4f}")

    mat = build_matrix(exp, repo, cfg)
    df = mat["df"]
    feature_cols = mat["feature_cols"]
    addon = mat["addon_cols"]
    tags = mat["tags"]

    if list(mat["base_feat"]) != list(parent["feature_cols"]):
        # order may differ — require same set
        if set(mat["base_feat"]) != set(parent["feature_cols"]):
            raise AssertionError("C1 base feature set drift")

    write_json(
        art / "features.json",
        {
            "exp": exp,
            "phase": "C1_sensitivity",
            "addon_cols": addon,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "block_tags": tags,
            "nulls": null_report(df, feature_cols),
            "feature_hash": _feature_hash(feature_cols),
            "bar_eligible": exp != "all3" or True,  # all run bar numbers; keep rule uses independent
            "keep_eligible": exp in ("smoke", "obs", "via"),
            "notes": {
                "smoke": "post-dx quit caveat; FE gap fix susmk*",
                "obs": "BMI-redundant expectation",
                "via": "severity-adjacent vision self-report",
                "all3": "joint ceiling; cannot alone expand C1",
            }.get(exp),
        },
    )
    log(f"features n={len(feature_cols)} addon={addon}")

    splits = make_block_splits(df, feature_cols, feature_set=f"sens_{exp}")
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={lgbm_device} n_trials={cfg['run']['n_trials']}")

    log("HPO LGBM")
    lgbm_pack = tune_lightgbm(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg, device=lgbm_device
    )
    log(f"LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log("HPO CatBoost")
    cat_pack = tune_catboost(splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg)
    log(f"Cat val_auc={cat_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family([lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"VAL-SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    freeze = {
        "run_id": run_id,
        "phase": "C1_sensitivity",
        "exp": exp,
        "addon_cols": addon,
        "keep_eligible": exp in ("smoke", "obs", "via"),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_cols": feature_cols,
        "feature_hash": _feature_hash(feature_cols),
        "parent_c1_run_id": cfg["paths"]["parent_c1_run_id"],
        "n_trials": int(cfg["run"]["n_trials"]),
    }
    write_json(art / "selected_model.json", freeze)
    log(f"FREEZE {art / 'selected_model.json'}")

    models_dir = art / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(selected["model"], models_dir / "selected.joblib")
    joblib.dump(lgbm_pack["model"], models_dir / "lgbm.joblib")
    joblib.dump(cat_pack["model"], models_dir / "catboost.joblib")

    proba_val = predict_proba(selected["model"], splits.X_val)
    cal = fit_calibrators(
        proba_val,
        splits.y_val,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test = predict_proba(selected["model"], splits.X_test)
    tr = full_report(splits.y_test, proba_test, tag=f"sens_{exp}_test")

    log("parent C1 proba")
    proba_c1 = load_parent_c1_proba(repo, cfg, parent, splits.person_id_test)

    d_auc = tr["macro_ovr_auc"] - parent["test_macro_ovr_auc"]
    d_bin = tr["binary_auc"] - parent["test_binary_auc"]
    boot = bootstrap_ci(
        splits.y_test,
        proba_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed,
    )
    boot_d = paired_delta_bootstrap(
        splits.y_test,
        proba_test,
        proba_c1,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed + 19,
    )

    c1 = d_auc > 0.01
    c2 = bool(boot_d["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))

    metrics_test: dict[str, Any] = {
        "phase": "C1_sensitivity",
        "exp": exp,
        "addon_cols": addon,
        "keep_eligible": exp in ("smoke", "obs", "via"),
        "selected_family": selected["family"],
        "selected_raw": tr,
        "selected_cal_sigmoid": full_report(
            splits.y_test, cal["primary"].transform(proba_test), tag=f"sens_{exp}_cal"
        ),
        "delta_vs_c1": {
            "parent_run": cfg["paths"]["parent_c1_run_id"],
            "delta_macro_ovr_auc": d_auc,
            "delta_binary_auc": d_bin,
            "delta_macro_auprc": tr["macro_auprc"] - parent["test_macro_auprc"],
            "criterion1_point_delta_gt_0p01": c1,
            "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
            "criterion3_perm_addon_stable": None,
            "decision_bar_pass": None,
            "can_expand_c1": None,
            "bootstrap_paired_delta": boot_d,
        },
        "bootstrap_test": boot,
    }
    write_json(
        art / "metrics_val.json",
        {
            "selected_raw": full_report(splits.y_val, proba_val, tag=f"sens_{exp}_val"),
            "lightgbm": lgbm_pack.get("val_report"),
            "catboost": cat_pack.get("val_report"),
        },
    )
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {"test_scored": True, "test_scored_at": datetime.now(timezone.utc).isoformat()},
    )
    log(f"TEST auc={tr['macro_ovr_auc']:.4f} bin={tr['binary_auc']:.4f} Δc1={d_auc:+.4f}")

    explain: dict[str, Any] = {}
    c3 = False
    per_feat: dict[str, float] = {}
    if not skip_shap:
        log("SHAP + perm")
        try:
            explain["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_sens_{exp}",
            )
        except Exception as e:
            log(f"SHAP failed: {e}")
            explain["shap_error"] = str(e)
        try:
            explain["permutation"] = permutation_on_val(
                selected["model"],
                splits.X_val,
                splits.y_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_sens_{exp}",
            )
            s = pd.read_csv(explain["permutation"]["csv"], index_col=0).iloc[:, 0]
            addon_imp = s.reindex(addon).dropna()
            mean_a = float(addon_imp.mean()) if len(addon_imp) else 0.0
            n_pos = int((addon_imp > 0).sum())
            per_feat = {c: float(s[c]) if c in s.index else float("nan") for c in addon}
            c3 = (mean_a > 0.0) and (n_pos >= 1)
            metrics_test["delta_vs_c1"]["criterion3_detail"] = {
                "mean_addon_perm_auc_drop": mean_a,
                "n_addon_perm_positive": n_pos,
                "n_addon": len(addon),
            }
            metrics_test["delta_vs_c1"]["per_feature_perm"] = per_feat
            log(f"c3 mean_perm={mean_a:.5f} n_pos={n_pos} per={per_feat}")
        except Exception as e:
            log(f"perm failed: {e}")
            explain["perm_error"] = str(e)
            c3 = False
        write_json(art / "explain.json", explain)
    else:
        metrics_test["delta_vs_c1"]["criterion3_detail"] = {"skipped": True}

    metrics_test["delta_vs_c1"]["criterion3_perm_addon_stable"] = c3
    bar = bool(c1 and c2 and c3)
    metrics_test["delta_vs_c1"]["decision_bar_pass"] = bar
    # expand only independent passes
    metrics_test["delta_vs_c1"]["can_expand_c1"] = bool(bar and exp in ("smoke", "obs", "via"))
    if exp == "all3":
        metrics_test["delta_vs_c1"]["can_expand_c1"] = False
        metrics_test["delta_vs_c1"]["decision_bar_note"] = (
            "joint ceiling; expand C1 only via independent S1/S2/S3 passes"
        )
    write_json(art / "metrics_test.json", metrics_test)

    elapsed = time.time() - t0
    d = metrics_test["delta_vs_c1"]
    report = [
        f"# C1 sensitivity — {exp} — {run_id}",
        "",
        f"**addon:** {addon}",
        f"- Selected: **{selected['family']}**",
        f"- Test 4-AUC: **{tr['macro_ovr_auc']:.4f}** binary: **{tr['binary_auc']:.4f}**",
        f"- Δ vs C1 AUC: **{d_auc:+.4f}** binary: **{d_bin:+.4f}**",
        f"- c1={d['criterion1_point_delta_gt_0p01']} c2={d['criterion2_bootstrap_delta_auc_lo_gt_0']} "
        f"c3={d['criterion3_perm_addon_stable']}",
        f"- **decision_bar_pass: {d['decision_bar_pass']}** can_expand_c1={d['can_expand_c1']}",
        f"- per-feature perm: {per_feat}",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    (art / "REPORT.md").write_text("\n".join(report))
    write_json(
        art / "run_manifest.json",
        {
            "run_id": run_id,
            "exp": exp,
            "elapsed_sec": elapsed,
            "git_hash": _git_hash(repo),
            "python": sys.version,
            "platform": platform.platform(),
            "decision_bar_pass": d["decision_bar_pass"],
            "can_expand_c1": d["can_expand_c1"],
            "delta_vs_c1_auc": d_auc,
        },
    )
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done bar={d['decision_bar_pass']} expand={d['can_expand_c1']} → {art}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C1 sensitivity: smoke/obs/via/all3")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--exp", type=str, choices=list(EXPS.keys()), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if not args.all and not args.exp:
        ap.error("need --exp or --all")

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    order = ["smoke", "obs", "via", "all3"] if args.all else [args.exp]

    for exp in order:
        run_id = args.run_id
        if run_id is None or (args.all and len(order) > 1):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"sens_{exp}_{ts}"
        run_one(
            exp,
            cfg=cfg,
            repo=repo,
            run_id=run_id,
            quick=args.quick,
            n_trials=args.n_trials,
            skip_shap=args.skip_shap,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
