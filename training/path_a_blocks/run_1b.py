"""Phase 1B: watch + onboarding + comorbidity (product checklist)."""

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
    block_tags_1b,
    load_watch_onboarding_comorbidity,
    make_block_splits,
    null_report,
    resolve_comorbidity_binaries,
)
from training.path_a_blocks.diagnostics import bootstrap_ci, paired_delta_bootstrap
from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.evaluate import reliability_diagrams, write_json
from training.path_a_watch.explain import permutation_on_val, shap_summary
from training.path_a_watch.hpo import pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_a_watch.models import predict_proba, resolve_lgbm_device


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


def assert_metrics_ref(
    repo: Path,
    *,
    artifacts_root: str,
    run_id: str,
    ref: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    path = repo / artifacts_root / run_id / "metrics_test.json"
    if not path.exists():
        # 1A lives under path_a_blocks artifacts
        path = repo / "training/path_a_blocks/artifacts" / run_id / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(f"{label} metrics missing: {path}")
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if k in ref and abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"{label} {k}: config {ref[k]} != artifact {v}")
    return {
        **checks,
        "family": m.get("selected_family"),
        "source_path": str(path),
        "per_class_ovr_auc": raw.get("per_class_ovr_auc"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A 1B watch+onboarding+comorbidity")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument(
        "--feature-set",
        type=str,
        default="core",
        choices=["core", "no_hbp", "plus_complications", "plus_obs", "ge5pct"],
    )
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    fs = args.feature_set
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"comorb_{fs}_%Y%m%dT%H%M%SZ"
    )
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    art.mkdir(parents=True, exist_ok=True)

    if args.quick:
        cfg["run"]["n_trials"] = 2
        args.skip_shap = True
    if args.n_trials is not None:
        cfg["run"]["n_trials"] = int(args.n_trials)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    t0 = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"run_id={run_id} 1B feature_set={fs}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")

    # parent 1A + floor asserts
    parent_ref = assert_metrics_ref(
        repo,
        artifacts_root=cfg["paths"]["artifacts_root"],
        run_id=cfg["paths"]["parent_1a_run_id"],
        ref=cfg["parent_1a_reference"],
        label="parent_1a",
    )
    log(f"parent_1a ok {parent_ref['source_path']} auc={parent_ref['test_macro_ovr_auc']:.4f}")

    floor_path = (
        repo
        / cfg["paths"]["floor_artifacts"]
        / cfg["paths"]["floor_run_id"]
        / "metrics_test.json"
    )
    floor_m = json.loads(floor_path.read_text())
    floor_ref = {
        "test_macro_ovr_auc": float(floor_m["selected_raw"]["macro_ovr_auc"]),
        "test_binary_auc": float(floor_m["selected_raw"]["binary_auc"]),
        "test_macro_auprc": float(floor_m["selected_raw"]["macro_auprc"]),
        "source_path": str(floor_path),
    }
    for k in ("test_macro_ovr_auc", "test_binary_auc", "test_macro_auprc"):
        if abs(float(cfg["floor_reference"][k]) - floor_ref[k]) > 1e-9:
            raise AssertionError(f"floor_reference mismatch {k}")
    log(f"floor ok auc={floor_ref['test_macro_ovr_auc']:.4f}")

    comorb_bins = resolve_comorbidity_binaries(cfg["data"], fs)
    df, watch_cols, onboard_cols, comorb_cols, feature_cols = (
        load_watch_onboarding_comorbidity(
            repo,
            watch_green=cfg["paths"]["watch_green"],
            onboarding=cfg["paths"]["onboarding"],
            comorbidity=cfg["paths"]["comorbidity"],
            pool_masks=cfg["paths"]["pool_masks"],
            onboarding_keep=cfg["data"]["onboarding_keep"],
            comorbidity_binaries=comorb_bins,
            expected_n=int(cfg["data"]["expected_n"]),
        )
    )
    # hard-lock: onboarding must match 1A keep list exactly
    if list(onboard_cols) != list(cfg["data"]["onboarding_keep"]):
        raise AssertionError("onboarding cols drifted from 1A keep list")
    if len(watch_cols) != 30:
        raise AssertionError(f"expected 30 watch cols, got {len(watch_cols)}")

    tags = block_tags_1b(watch_cols, onboard_cols, comorb_cols)
    nulls = null_report(df, feature_cols)
    write_json(
        art / "features.json",
        {
            "feature_set": fs,
            "n_watch": len(watch_cols),
            "n_onboarding": len(onboard_cols),
            "n_comorbidity": len(comorb_cols),
            "n_total": len(feature_cols),
            "watch_cols": watch_cols,
            "onboarding_cols": onboard_cols,
            "comorbidity_cols": comorb_cols,
            "comorbidity_binaries": comorb_bins,
            "feature_cols": feature_cols,
            "block_tags": tags,
            "nulls": nulls,
            "feature_hash": _feature_hash(feature_cols),
        },
    )
    log(
        f"features w={len(watch_cols)} o={len(onboard_cols)} c={len(comorb_cols)} "
        f"total={len(feature_cols)} bins={comorb_bins}"
    )

    splits = make_block_splits(df, feature_cols, feature_set=f"1b_{fs}")
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={lgbm_device}")

    log(f"HPO LGBM trials={cfg['run']['n_trials']}")
    lgbm_pack = tune_lightgbm(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg, device=lgbm_device
    )
    log(f"LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log(f"HPO CatBoost trials={cfg['run']['n_trials']}")
    cat_pack = tune_catboost(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg
    )
    log(f"Cat val_auc={cat_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family([lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"VAL-SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError("refuse overwrite freeze/test")

    freeze = {
        "run_id": run_id,
        "phase": "1B_comorbidity",
        "feature_set": fs,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_cols": feature_cols,
        "comorbidity_binaries": comorb_bins,
        "feature_hash": _feature_hash(feature_cols),
        "parent_1a_run_id": cfg["paths"]["parent_1a_run_id"],
        "parent_1a_auc": parent_ref["test_macro_ovr_auc"],
        "is_primary_claim_set": fs == "core",
        "selected_device": selected.get("device"),
        "selected_boosting_type": selected.get("boosting_type"),
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
    proba_test_cal = cal["primary"].transform(proba_test)

    # parent 1A model proba on same test persons (watch+onboard only cols)
    parent_model_path = (
        repo
        / cfg["paths"]["artifacts_root"]
        / cfg["paths"]["parent_1a_run_id"]
        / "models"
        / "selected.joblib"
    )
    parent_model = joblib.load(parent_model_path)
    parent_feat = watch_cols + onboard_cols
    # ensure parent feature list matches 1A freeze
    parent_freeze = json.loads(
        (
            repo
            / cfg["paths"]["artifacts_root"]
            / cfg["paths"]["parent_1a_run_id"]
            / "selected_model.json"
        ).read_text()
    )
    parent_cols_1a = list(parent_freeze["feature_cols"])
    if parent_feat != parent_cols_1a:
        raise AssertionError(
            "parent_feat order/content != 1A selected_model.feature_cols"
        )
    proba_parent_test = predict_proba(parent_model, splits.X_test[parent_feat])
    parent_recomp = macro_ovr_auc(splits.y_test, proba_parent_test)
    if abs(parent_recomp - parent_ref["test_macro_ovr_auc"]) > 1e-9:
        raise AssertionError(
            f"recomputed parent 1A auc={parent_recomp} != "
            f"stored {parent_ref['test_macro_ovr_auc']}"
        )

    tr = full_report(splits.y_test, proba_test, tag="1b_test_raw")
    d_auc_1a = tr["macro_ovr_auc"] - parent_ref["test_macro_ovr_auc"]
    d_bin_1a = tr["binary_auc"] - parent_ref["test_binary_auc"]
    d_auc_fl = tr["macro_ovr_auc"] - floor_ref["test_macro_ovr_auc"]

    boot_1b = bootstrap_ci(
        splits.y_test,
        proba_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed,
    )
    boot_delta_1a = paired_delta_bootstrap(
        splits.y_test,
        proba_test,
        proba_parent_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed + 7,
    )

    # per-class delta vs 1A
    pc_1a = parent_ref.get("per_class_ovr_auc") or {}
    pc_delta = {
        str(k): float(tr["per_class_ovr_auc"][int(k)]) - float(pc_1a.get(str(k), pc_1a.get(k, 0)))
        for k in tr["per_class_ovr_auc"]
    }

    c1 = d_auc_1a > 0.01
    c2 = bool(boot_delta_1a["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))

    metrics_test: dict[str, Any] = {
        "phase": "1B_comorbidity",
        "feature_set": fs,
        "is_primary_claim_set": fs == "core",
        "selected_family": selected["family"],
        "selected_raw": tr,
        "selected_cal_sigmoid": full_report(
            splits.y_test, proba_test_cal, tag="1b_test_cal"
        ),
        "delta_vs_1a": {
            "parent_run": cfg["paths"]["parent_1a_run_id"],
            "delta_macro_ovr_auc": d_auc_1a,
            "delta_binary_auc": d_bin_1a,
            "delta_macro_auprc": tr["macro_auprc"] - parent_ref["test_macro_auprc"],
            "per_class_ovr_auc_delta": pc_delta,
            "class2_delta": pc_delta.get("2"),
            "criterion1_point_delta_gt_0p01": c1,
            "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
            "criterion3_perm_comorbidity_stable": None,
            "decision_bar_pass": None,
            "bootstrap_paired_delta": boot_delta_1a,
        },
        "delta_vs_watch_floor": {
            "delta_macro_ovr_auc": d_auc_fl,
            "delta_binary_auc": tr["binary_auc"] - floor_ref["test_binary_auc"],
        },
        "bootstrap_1b_test": boot_1b,
    }
    write_json(art / "metrics_val.json", {
        "lightgbm": lgbm_pack.get("val_report"),
        "catboost": cat_pack.get("val_report"),
        "selected_raw": full_report(splits.y_val, proba_val, tag="1b_val"),
    })
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {"test_scored": True, "test_scored_at": datetime.now(timezone.utc).isoformat()},
    )
    log(
        f"TEST auc={tr['macro_ovr_auc']:.4f} bin={tr['binary_auc']:.4f} "
        f"Δ1a_auc={d_auc_1a:+.4f} Δ1a_bin={d_bin_1a:+.4f} c1={c1} c2={c2} "
        f"class2_Δ={pc_delta.get('2')}"
    )

    reliability_diagrams(
        splits.y_test,
        proba_test,
        art / "calibration" / "reliability_raw.png",
        title=f"1B {fs} reliability",
    )

    explain: dict[str, Any] = {}
    c3 = False
    if not args.skip_shap:
        log("SHAP + permutation")
        try:
            explain["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_1b_{fs}",
            )
            top = list(explain["shap"]["top10"].keys())
            explain["shap"]["top10_block_tags"] = {c: tags.get(c, "?") for c in top}
            n_c = sum(1 for c in top if tags.get(c) == "comorbidity")
            n_w = sum(1 for c in top if tags.get(c) == "watch_green")
            explain["shap"]["top10_comorbidity_count"] = n_c
            explain["shap"]["top10_watch_count"] = n_w
            explain["shap"]["guardrail_all_non_watch"] = n_w == 0
            log(f"SHAP top10 comorbidity={n_c} watch={n_w}")
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
                prefix=f"{selected['family']}_1b_{fs}",
            )
            csv_path = explain["permutation"]["csv"]
            s = pd.read_csv(csv_path, index_col=0).iloc[:, 0]
            comorb_imp = s.reindex(comorb_cols).dropna()
            mean_c = float(comorb_imp.mean()) if len(comorb_imp) else 0.0
            n_pos = int((comorb_imp > 0).sum())
            top_half = set(s.sort_values(ascending=False).head(max(1, len(s) // 2)).index)
            n_top = sum(1 for c in comorb_cols if c in top_half)
            c3 = (mean_c > 0.0) and (n_pos >= 1)
            metrics_test["delta_vs_1a"]["criterion3_detail"] = {
                "mean_comorbidity_perm_auc_drop": mean_c,
                "n_comorbidity_perm_positive": n_pos,
                "n_comorbidity_features": len(comorb_cols),
                "n_comorbidity_in_top_half": n_top,
            }
            log(f"c3 mean_perm={mean_c:.5f} n_pos={n_pos}/{len(comorb_cols)}")
        except Exception as e:
            log(f"perm failed: {e}")
            explain["perm_error"] = str(e)
            c3 = False
        write_json(art / "explain.json", explain)
    else:
        metrics_test["delta_vs_1a"]["criterion3_detail"] = {"skipped": True}

    metrics_test["delta_vs_1a"]["criterion3_perm_comorbidity_stable"] = c3
    metrics_test["delta_vs_1a"]["decision_bar_pass"] = bool(
        fs == "core" and c1 and c2 and c3
    )
    # non-core sets: report criteria but decision_bar_pass only meaningful for core
    if fs != "core":
        metrics_test["delta_vs_1a"]["decision_bar_pass"] = False
        metrics_test["delta_vs_1a"]["decision_bar_note"] = (
            "decision_bar_pass only evaluated for feature_set=core"
        )
    write_json(art / "metrics_test.json", metrics_test)

    elapsed = time.time() - t0
    d = metrics_test["delta_vs_1a"]
    report = [
        f"# Path A 1B comorbidity — {run_id}",
        "",
        f"**feature_set:** `{fs}`  |  primary_claim={fs == 'core'}",
        "",
        f"- Selected: **{selected['family']}**",
        f"- Test 4-AUC: **{tr['macro_ovr_auc']:.4f}**  binary: **{tr['binary_auc']:.4f}**",
        f"- Δ vs 1A AUC: **{d_auc_1a:+.4f}**  binary: **{d_bin_1a:+.4f}**",
        f"- Δ vs floor AUC: **{d_auc_fl:+.4f}**",
        f"- Per-class Δ vs 1A: {pc_delta}",
        f"- c1: {d['criterion1_point_delta_gt_0p01']}  c2: {d['criterion2_bootstrap_delta_auc_lo_gt_0']}  "
        f"c3: {d['criterion3_perm_comorbidity_stable']}",
        f"- **decision_bar_pass: {d['decision_bar_pass']}**",
        f"- Comorbidity cols: {comorb_cols}",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    if explain.get("shap"):
        report += [
            "## SHAP",
            f"- Top10: {list(explain['shap']['top10'].keys())}",
            f"- Tags: {explain['shap'].get('top10_block_tags')}",
            "",
        ]
    (art / "REPORT.md").write_text("\n".join(report))
    write_json(
        art / "run_manifest.json",
        {
            "run_id": run_id,
            "feature_set": fs,
            "elapsed_sec": elapsed,
            "git_hash": _git_hash(repo),
            "python": sys.version,
            "platform": platform.platform(),
            "decision_bar_pass": d["decision_bar_pass"],
            "delta_vs_1a": {
                "auc": d_auc_1a,
                "binary": d_bin_1a,
                "class2": pc_delta.get("2"),
            },
        },
    )
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done decision_bar_pass={d['decision_bar_pass']} → {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
