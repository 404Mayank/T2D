"""Phase 1C: watch + onboarding + mood (parent = 1A)."""

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
    block_tags_1c,
    load_watch_onboarding_mood,
    make_block_splits,
    null_report,
    resolve_mood_cols,
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


def assert_parent_1a(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["parent_1a_run_id"]
    path = repo / cfg["paths"]["artifacts_root"] / run_id / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["parent_1a_reference"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"parent_1a {k}: config {ref[k]} != artifact {v}")
    freeze = json.loads(
        (repo / cfg["paths"]["artifacts_root"] / run_id / "selected_model.json").read_text()
    )
    return {
        **checks,
        "family": m.get("selected_family"),
        "source_path": str(path),
        "feature_cols": list(freeze["feature_cols"]),
        "per_class_ovr_auc": raw.get("per_class_ovr_auc"),
        "model_path": str(
            repo / cfg["paths"]["artifacts_root"] / run_id / "models" / "selected.joblib"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A 1C watch+onboarding+mood")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument(
        "--feature-set",
        type=str,
        default="scores",
        choices=["scores", "scores_via", "paid_items", "full"],
    )
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    fs = args.feature_set
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"mood_{fs}_%Y%m%dT%H%M%SZ")
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

    log(f"run_id={run_id} 1C feature_set={fs}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")
    log("NOTE: only feature_set=scores is decision-bar eligible")

    parent = assert_parent_1a(repo, cfg)
    log(f"parent_1a ok auc={parent['test_macro_ovr_auc']:.4f}")

    mood_cols = resolve_mood_cols(cfg["data"], fs)
    df, watch_cols, onboard_cols, mood_feat, feature_cols = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=cfg["data"]["onboarding_keep"],
        mood_cols=mood_cols,
        expected_n=int(cfg["data"]["expected_n"]),
    )
    if list(onboard_cols) != list(cfg["data"]["onboarding_keep"]):
        raise AssertionError("onboarding drift from 1A")
    if len(watch_cols) != 30:
        raise AssertionError(f"expected 30 watch, got {len(watch_cols)}")
    parent_feat = watch_cols + onboard_cols
    if parent_feat != parent["feature_cols"]:
        raise AssertionError("parent feature list != 1A freeze feature_cols")

    tags = block_tags_1c(watch_cols, onboard_cols, mood_feat)
    nulls = null_report(df, feature_cols)
    # nullness by label on full core (soft diagnostic)
    null_by_label = {}
    for lab, g in df.groupby("label"):
        null_by_label[str(int(lab))] = {
            c: float(g[c].isna().mean()) for c in mood_feat if g[c].isna().any()
        }
    write_json(
        art / "features.json",
        {
            "feature_set": fs,
            "bar_eligible": fs == "scores",
            "n_watch": len(watch_cols),
            "n_onboarding": len(onboard_cols),
            "n_mood": len(mood_feat),
            "n_total": len(feature_cols),
            "mood_cols": mood_feat,
            "feature_cols": feature_cols,
            "block_tags": tags,
            "nulls": nulls,
            "mood_null_by_label": null_by_label,
            "feature_hash": _feature_hash(feature_cols),
        },
    )
    log(f"features total={len(feature_cols)} mood={mood_feat}")

    splits = make_block_splits(df, feature_cols, feature_set=f"1c_{fs}")
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={lgbm_device}")

    log(f"HPO LGBM trials={cfg['run']['n_trials']}")
    lgbm_pack = tune_lightgbm(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg, device=lgbm_device
    )
    log(f"LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log(f"HPO CatBoost trials={cfg['run']['n_trials']}")
    cat_pack = tune_catboost(splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg)
    log(f"Cat val_auc={cat_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family([lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"VAL-SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError("refuse overwrite")

    freeze = {
        "run_id": run_id,
        "phase": "1C_mood",
        "feature_set": fs,
        "bar_eligible": fs == "scores",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_cols": feature_cols,
        "mood_cols": mood_feat,
        "feature_hash": _feature_hash(feature_cols),
        "parent_1a_run_id": cfg["paths"]["parent_1a_run_id"],
        "n_trials": int(cfg["run"]["n_trials"]),
        "selected_device": selected.get("device"),
        "selected_boosting_type": selected.get("boosting_type"),
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

    parent_model = joblib.load(parent["model_path"])
    proba_parent = predict_proba(parent_model, splits.X_test[parent_feat])
    parent_recomp = macro_ovr_auc(splits.y_test, proba_parent)
    if abs(parent_recomp - parent["test_macro_ovr_auc"]) > 1e-9:
        raise AssertionError(
            f"recomputed parent auc={parent_recomp} != {parent['test_macro_ovr_auc']}"
        )

    tr = full_report(splits.y_test, proba_test, tag="1c_test_raw")
    d_auc = tr["macro_ovr_auc"] - parent["test_macro_ovr_auc"]
    d_bin = tr["binary_auc"] - parent["test_binary_auc"]
    floor_auc = float(cfg["floor_reference"]["test_macro_ovr_auc"])
    d_floor = tr["macro_ovr_auc"] - floor_auc

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
        proba_parent,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed + 11,
    )

    pc_1a = parent.get("per_class_ovr_auc") or {}
    pc_delta = {
        str(k): float(tr["per_class_ovr_auc"][int(k)]) - float(pc_1a.get(str(k), pc_1a.get(k, 0)))
        for k in tr["per_class_ovr_auc"]
    }

    c1 = d_auc > 0.01
    c2 = bool(boot_d["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))

    metrics_test: dict[str, Any] = {
        "phase": "1C_mood",
        "feature_set": fs,
        "bar_eligible": fs == "scores",
        "selected_family": selected["family"],
        "selected_raw": tr,
        "selected_cal_sigmoid": full_report(
            splits.y_test, cal["primary"].transform(proba_test), tag="1c_cal"
        ),
        "delta_vs_1a": {
            "parent_run": cfg["paths"]["parent_1a_run_id"],
            "delta_macro_ovr_auc": d_auc,
            "delta_binary_auc": d_bin,
            "delta_macro_auprc": tr["macro_auprc"] - parent["test_macro_auprc"],
            "per_class_ovr_auc_delta": pc_delta,
            "class2_delta": pc_delta.get("2"),
            "criterion1_point_delta_gt_0p01": c1,
            "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
            "criterion3_perm_mood_stable": None,
            "per_feature_perm": None,
            "decision_bar_pass": None,
            "bootstrap_paired_delta": boot_d,
            "power_note": "bar-pass often needs ΔAUC≳0.015 at n_test=277 (1B c2 failed at +0.01)",
        },
        "delta_vs_watch_floor": {
            "delta_macro_ovr_auc": d_floor,
            "delta_binary_auc": tr["binary_auc"] - float(cfg["floor_reference"]["test_binary_auc"]),
        },
        "bootstrap_1c_test": boot,
    }
    write_json(art / "metrics_val.json", {
        "selected_raw": full_report(splits.y_val, proba_val, tag="1c_val"),
        "lightgbm": lgbm_pack.get("val_report"),
        "catboost": cat_pack.get("val_report"),
    })
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {"test_scored": True, "test_scored_at": datetime.now(timezone.utc).isoformat()},
    )
    log(
        f"TEST auc={tr['macro_ovr_auc']:.4f} bin={tr['binary_auc']:.4f} "
        f"Δ1a={d_auc:+.4f} Δbin={d_bin:+.4f} c1={c1} c2={c2} class2_Δ={pc_delta.get('2')}"
    )

    reliability_diagrams(
        splits.y_test,
        proba_test,
        art / "calibration" / "reliability_raw.png",
        title=f"1C {fs} reliability",
    )

    explain: dict[str, Any] = {}
    c3 = False
    per_feat: dict[str, float] = {}
    if not args.skip_shap:
        log("SHAP + permutation")
        try:
            explain["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_1c_{fs}",
            )
            top = list(explain["shap"]["top10"].keys())
            explain["shap"]["top10_block_tags"] = {c: tags.get(c, "?") for c in top}
            n_m = sum(1 for c in top if tags.get(c) == "mood")
            n_w = sum(1 for c in top if tags.get(c) == "watch_green")
            explain["shap"]["top10_mood_count"] = n_m
            explain["shap"]["top10_watch_count"] = n_w
            explain["shap"]["guardrail_all_non_watch"] = n_w == 0
            log(f"SHAP top10 mood={n_m} watch={n_w}")
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
                prefix=f"{selected['family']}_1c_{fs}",
            )
            s = pd.read_csv(explain["permutation"]["csv"], index_col=0).iloc[:, 0]
            mood_imp = s.reindex(mood_feat).dropna()
            mean_m = float(mood_imp.mean()) if len(mood_imp) else 0.0
            n_pos = int((mood_imp > 0).sum())
            per_feat = {c: float(s[c]) if c in s.index else float("nan") for c in mood_feat}
            c3 = (mean_m > 0.0) and (n_pos >= 1)
            metrics_test["delta_vs_1a"]["criterion3_detail"] = {
                "mean_mood_perm_auc_drop": mean_m,
                "n_mood_perm_positive": n_pos,
                "n_mood_features": len(mood_feat),
            }
            metrics_test["delta_vs_1a"]["per_feature_perm"] = per_feat
            log(f"c3 mean_perm={mean_m:.5f} n_pos={n_pos} per_feat={per_feat}")
        except Exception as e:
            log(f"perm failed: {e}")
            explain["perm_error"] = str(e)
            c3 = False
        write_json(art / "explain.json", explain)
    else:
        metrics_test["delta_vs_1a"]["criterion3_detail"] = {"skipped": True}

    metrics_test["delta_vs_1a"]["criterion3_perm_mood_stable"] = c3
    if fs == "scores":
        metrics_test["delta_vs_1a"]["decision_bar_pass"] = bool(c1 and c2 and c3)
    else:
        metrics_test["delta_vs_1a"]["decision_bar_pass"] = False
        metrics_test["delta_vs_1a"]["decision_bar_note"] = (
            "only feature_set=scores is bar-eligible"
        )
    write_json(art / "metrics_test.json", metrics_test)

    elapsed = time.time() - t0
    d = metrics_test["delta_vs_1a"]
    report = [
        f"# Path A 1C mood — {run_id}",
        "",
        f"**feature_set:** `{fs}` | bar_eligible={fs == 'scores'}",
        f"- Selected: **{selected['family']}**",
        f"- Test 4-AUC: **{tr['macro_ovr_auc']:.4f}** binary: **{tr['binary_auc']:.4f}**",
        f"- Δ vs 1A AUC: **{d_auc:+.4f}** binary: **{d_bin:+.4f}**",
        f"- Δ vs floor AUC: **{d_floor:+.4f}**",
        f"- class-2 Δ: {pc_delta.get('2')}",
        f"- per-feature perm: {per_feat}",
        f"- c1={d['criterion1_point_delta_gt_0p01']} c2={d['criterion2_bootstrap_delta_auc_lo_gt_0']} "
        f"c3={d['criterion3_perm_mood_stable']}",
        f"- **decision_bar_pass: {d['decision_bar_pass']}**",
        f"- Mood cols: {mood_feat}",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    if explain.get("shap"):
        report += [
            f"- SHAP top10: {list(explain['shap']['top10'].keys())}",
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
            "delta_vs_1a_auc": d_auc,
            "delta_vs_1a_binary": d_bin,
        },
    )
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done decision_bar_pass={d['decision_bar_pass']} → {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
