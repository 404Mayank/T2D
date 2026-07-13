"""Phase 1A: watch GREEN + hard onboarding block."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from training.path_a_blocks.data_blocks import (
    block_tags,
    load_watch_onboarding,
    make_block_splits,
    null_report,
)
from training.path_a_blocks.diagnostics import bootstrap_ci, paired_delta_bootstrap
from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.evaluate import reliability_diagrams, write_json
from training.path_a_watch.explain import permutation_on_val, shap_summary
from training.path_a_watch.hpo import pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report
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


def _pkg_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "sklearn", "lightgbm", "catboost", "optuna", "shap"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "missing"
    return out


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]



def assert_floor_reference(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Load floor metrics_test.json and assert config floor_reference matches."""
    fr_path = (
        repo
        / cfg["paths"]["floor_artifacts"]
        / cfg["paths"]["floor_run_id"]
        / "metrics_test.json"
    )
    if not fr_path.exists():
        raise FileNotFoundError(f"floor metrics missing: {fr_path}")
    import json
    floor_m = json.loads(fr_path.read_text())
    raw = floor_m["selected_raw"]
    ref = cfg.get("floor_reference", {})
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if k not in ref:
            continue
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(
                f"floor_reference.{k}={ref[k]} != artifact {v} ({fr_path})"
            )
    # return artifact values as source of truth
    return {
        "test_macro_ovr_auc": checks["test_macro_ovr_auc"],
        "test_binary_auc": checks["test_binary_auc"],
        "test_macro_auprc": checks["test_macro_auprc"],
        "family": floor_m.get("selected_family", ref.get("family")),
        "source_path": str(fr_path),
    }

def _strip(pack: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pack.items() if k not in ("model", "val_report")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A 1A watch+onboarding")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("onboarding_%Y%m%dT%H%M%SZ")
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

    log(f"run_id={run_id} 1A watch+onboarding")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")

    floor_ref = assert_floor_reference(repo, cfg)
    log(f"floor_reference ok from {floor_ref['source_path']}")

    df, watch_cols, onboard_cols, feature_cols = load_watch_onboarding(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=cfg["data"]["onboarding_keep"],
        expected_n=int(cfg["data"]["expected_n"]),
    )
    tags = block_tags(watch_cols, onboard_cols)
    nulls = null_report(df, feature_cols)
    write_json(
        art / "features.json",
        {
            "n_watch": len(watch_cols),
            "n_onboarding": len(onboard_cols),
            "n_total": len(feature_cols),
            "watch_cols": watch_cols,
            "onboarding_cols": onboard_cols,
            "feature_cols": feature_cols,
            "block_tags": tags,
            "nulls": nulls,
            "feature_hash": _feature_hash(feature_cols),
        },
    )
    log(
        f"features watch={len(watch_cols)} onboard={len(onboard_cols)} "
        f"total={len(feature_cols)} any_null_rows={nulls['rows_any_null']}"
    )

    splits = make_block_splits(df, feature_cols)
    log(
        f"splits train/val/test="
        f"{len(splits.y_train)}/{len(splits.y_val)}/{len(splits.y_test)}"
    )

    # HPO cfg shape expected by path_a_watch.hpo
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={lgbm_device}")

    log(f"HPO LGBM trials={cfg['run']['n_trials']}")
    lgbm_pack = tune_lightgbm(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        cfg,
        device=lgbm_device,
    )
    log(
        f"LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f} "
        f"auprc={lgbm_pack['val_macro_auprc']:.4f} src={lgbm_pack.get('source')}"
    )
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log(f"HPO CatBoost trials={cfg['run']['n_trials']}")
    cat_pack = tune_catboost(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        cfg,
    )
    log(
        f"Cat val_auc={cat_pack['val_macro_ovr_auc']:.4f} "
        f"auprc={cat_pack['val_macro_auprc']:.4f} src={cat_pack.get('source')}"
    )
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family(
        [lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"])
    )
    log(f"VAL-SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    if (art / "metrics_test.json").exists() or (art / "selected_model.json").exists():
        raise RuntimeError("refuse overwrite existing freeze/test artifacts")

    # floor_ref already loaded+asserted from artifact
    freeze = {
        "run_id": run_id,
        "phase": "1A_watch_onboarding",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_set": "watch_onboarding",
        "n_features": len(feature_cols),
        "n_watch": len(watch_cols),
        "n_onboarding": len(onboard_cols),
        "feature_hash": _feature_hash(feature_cols),
        "feature_cols": feature_cols,
        "onboarding_cols": onboard_cols,
        "lgbm_device_probe": lgbm_device,
        "selected_device": selected.get("device"),
        "device_fallback": selected.get("device_fallback"),
        "selected_boosting_type": selected.get("boosting_type"),
        "selection_rule": "global max val macro_ovr_auc then auprc within eps",
        "floor_reference": floor_ref,
        "competitors_val_only": {
            "lightgbm": {
                "val_macro_ovr_auc": lgbm_pack["val_macro_ovr_auc"],
                "val_macro_auprc": lgbm_pack["val_macro_auprc"],
            },
            "catboost": {
                "val_macro_ovr_auc": cat_pack["val_macro_ovr_auc"],
                "val_macro_auprc": cat_pack["val_macro_auprc"],
            },
        },
    }
    write_json(art / "selected_model.json", freeze)
    log(f"FREEZE written {art / 'selected_model.json'}")

    models_dir = art / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(lgbm_pack["model"], models_dir / "lgbm.joblib")
    joblib.dump(cat_pack["model"], models_dir / "catboost.joblib")
    joblib.dump(selected["model"], models_dir / "selected.joblib")

    # cal + test
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

    metrics_val = {
        "lightgbm": lgbm_pack.get("val_report"),
        "catboost": cat_pack.get("val_report"),
        "selected_raw": full_report(splits.y_val, proba_val, tag="1a_val_raw"),
        "note": "val post-HPO diagnostic",
    }
    write_json(art / "metrics_val.json", metrics_val)

    # floor proba on same test persons for paired Δ bootstrap
    floor_model_path = (
        repo
        / cfg["paths"]["floor_artifacts"]
        / cfg["paths"]["floor_run_id"]
        / "models"
        / cfg["paths"]["floor_model"]
    )
    floor_model = joblib.load(floor_model_path)
    # floor model expects watch-only columns
    proba_floor_test = predict_proba(floor_model, splits.X_test[watch_cols])

    metrics_test = {
        "phase": "1A_watch_onboarding",
        "selected_family": selected["family"],
        "claim": "raw ranking primary; cal diagnostic",
        "selected_raw": full_report(splits.y_test, proba_test, tag="1a_test_raw"),
        "selected_cal_sigmoid": full_report(
            splits.y_test, proba_test_cal, tag="1a_test_cal"
        ),
        "selected_cal_isotonic": full_report(
            splits.y_test,
            cal["secondary"].transform(proba_test),
            tag="1a_test_cal_iso",
        ),
    }
    fr = floor_ref
    tr = metrics_test["selected_raw"]
    d_auc = tr["macro_ovr_auc"] - float(fr["test_macro_ovr_auc"])
    d_bin = tr["binary_auc"] - float(fr["test_binary_auc"])
    d_ap = tr["macro_auprc"] - float(fr["test_macro_auprc"])

    boot_1a = bootstrap_ci(
        splits.y_test,
        proba_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed,
    )
    boot_delta = paired_delta_bootstrap(
        splits.y_test,
        proba_test,
        proba_floor_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed + 1,
    )
    c1 = d_auc > 0.01
    c2 = bool(boot_delta["delta_macro_ovr_auc"].get("ci_lower_gt_zero"))
    # criterion3 filled after SHAP/perm; placeholder False until then
    metrics_test["delta_vs_watch_floor"] = {
        "floor_run": cfg["paths"]["floor_run_id"],
        "floor_metrics": fr,
        "delta_macro_ovr_auc": d_auc,
        "delta_binary_auc": d_bin,
        "delta_macro_auprc": d_ap,
        "criterion1_point_delta_gt_0p01": c1,
        "criterion2_bootstrap_delta_auc_lo_gt_0": c2,
        "criterion3_perm_onboarding_stable": None,
        "decision_bar_pass": None,
        "bootstrap_1a_test": boot_1a,
        "bootstrap_paired_delta": boot_delta,
    }
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {
            "test_scored": True,
            "test_scored_at": datetime.now(timezone.utc).isoformat(),
            "selected_family": selected["family"],
        },
    )

    reliability_diagrams(
        splits.y_test,
        proba_test,
        art / "calibration" / "reliability_raw.png",
        title="1A reliability raw",
    )

    d = metrics_test["delta_vs_watch_floor"]
    log(
        f"TEST auc={tr['macro_ovr_auc']:.4f} bin={tr['binary_auc']:.4f} "
        f"Δauc={d['delta_macro_ovr_auc']:+.4f} Δbin={d['delta_binary_auc']:+.4f} "
        f"decision_bar_pass={d.get('decision_bar_pass')} c1={d.get('criterion1_point_delta_gt_0p01')} c2={d.get('criterion2_bootstrap_delta_auc_lo_gt_0')}"
    )

    explain = {}
    if not args.skip_shap:
        log("SHAP + permutation")
        try:
            explain["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_1a",
            )
            # block composition of top10
            top = list(explain["shap"]["top10"].keys())
            explain["shap"]["top10_block_tags"] = {c: tags.get(c, "?") for c in top}
            n_on = sum(1 for c in top if tags.get(c) == "onboarding")
            explain["shap"]["top10_onboarding_count"] = n_on
            explain["shap"]["guardrail_all_onboarding"] = n_on >= 10
            log(
                f"SHAP top10 onboarding_count={n_on}/10 "
                f"guardrail_all_onboarding={n_on >= 10}"
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
                prefix=f"{selected['family']}_1a",
            )
        except Exception as e:
            log(f"perm failed: {e}")
            explain["perm_error"] = str(e)
        write_json(art / "explain.json", explain)

    # Decision bar criterion 3: onboarding block has stable positive perm importance
    # (mean perm ΔAUC over onboarding cols > 0 with at least one col in top half)
    c3 = False
    if explain.get("permutation") and explain["permutation"].get("top10"):
        # re-read full perm csv if present
        try:
            import pandas as pd
            csv_path = explain["permutation"]["csv"]
            s = pd.read_csv(csv_path, index_col=0).iloc[:, 0]
            onboard_imp = s.reindex(onboard_cols).dropna()
            mean_on = float(onboard_imp.mean()) if len(onboard_imp) else 0.0
            top_half = set(s.sort_values(ascending=False).head(max(1, len(s)//2)).index)
            n_on_top = sum(1 for c in onboard_cols if c in top_half)
            c3 = (mean_on > 0.0) and (n_on_top >= 1)
            metrics_test["delta_vs_watch_floor"]["criterion3_detail"] = {
                "mean_onboarding_perm_auc_drop": mean_on,
                "n_onboarding_in_top_half": n_on_top,
            }
        except Exception as e:
            log(f"criterion3 compute failed: {e}")
            c3 = False
    elif args.skip_shap:
        # without perm, criterion3 unknown — do not pass composite bar
        c3 = False
        metrics_test["delta_vs_watch_floor"]["criterion3_detail"] = {
            "skipped": True,
            "reason": "skip_shap",
        }

    metrics_test["delta_vs_watch_floor"]["criterion3_perm_onboarding_stable"] = c3
    c1 = bool(metrics_test["delta_vs_watch_floor"]["criterion1_point_delta_gt_0p01"])
    c2 = bool(metrics_test["delta_vs_watch_floor"]["criterion2_bootstrap_delta_auc_lo_gt_0"])
    metrics_test["delta_vs_watch_floor"]["decision_bar_pass"] = bool(c1 and c2 and c3)
    write_json(art / "metrics_test.json", metrics_test)

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "phase": "1A_watch_onboarding",
        "elapsed_sec": elapsed,
        "git_hash": _git_hash(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "DRI_PRIME": os.environ.get("DRI_PRIME"),
        "seed": seed,
        "n_trials": cfg["run"]["n_trials"],
        "selected_family": selected["family"],
        "delta_vs_watch_floor": metrics_test["delta_vs_watch_floor"],
        "framing": (
            "Deployable config track (watch+onboarding). "
            "Paper watch-only claim remains path_a_watch floor unless GREEN v2 improves it."
        ),
    }
    write_json(art / "run_manifest.json", manifest)

    report = [
        f"# Path A 1A — watch + onboarding — {run_id}",
        "",
        "## Framing",
        manifest["framing"],
        "",
        "## Val selection",
        f"- LGBM val AUC={lgbm_pack['val_macro_ovr_auc']:.4f}",
        f"- CatBoost val AUC={cat_pack['val_macro_ovr_auc']:.4f}",
        f"- **Selected: {selected['family']}**",
        "",
        "## Test (once)",
        f"- 4-class macro-OVR AUC: **{tr['macro_ovr_auc']:.4f}**",
        f"- Binary AUC: **{tr['binary_auc']:.4f}**",
        f"- Macro AUPRC: **{tr['macro_auprc']:.4f}**",
        f"- Per-class OVR AUC: {tr['per_class_ovr_auc']}",
        f"- QWK: {tr['ordinal']['qwk']:.4f}",
        "",
        "## Δ vs watch-only floor",
        f"- Floor run: `{cfg['paths']['floor_run_id']}`",
        f"- Δ 4-AUC: **{d['delta_macro_ovr_auc']:+.4f}**",
        f"- Δ binary AUC: **{d['delta_binary_auc']:+.4f}**",
        f"- Δ AUPRC: **{d['delta_macro_auprc']:+.4f}**",
        f"- criterion1 point ΔAUC>0.01: **{d['criterion1_point_delta_gt_0p01']}**",
        f"- criterion2 bootstrap ΔAUC lo>0: **{d['criterion2_bootstrap_delta_auc_lo_gt_0']}**",
        f"- criterion3 onboarding perm stable: **{d['criterion3_perm_onboarding_stable']}**",
        f"- **decision_bar_pass (all 3): {d['decision_bar_pass']}**",
        "",
        f"## Features: {len(watch_cols)} watch + {len(onboard_cols)} onboarding",
        f"Onboarding: {onboard_cols}",
        "",
    ]
    if explain.get("shap"):
        report += [
            "## SHAP guardrail",
            f"- Top10: {list(explain['shap']['top10'].keys())}",
            f"- Top10 block tags: {explain['shap'].get('top10_block_tags')}",
            f"- Onboarding in top10: {explain['shap'].get('top10_onboarding_count')}/10",
            f"- All-onboarding guardrail trip: {explain['shap'].get('guardrail_all_onboarding')}",
            "",
        ]
    report += [
        "## Next",
        "- If decision bar pass → 1B comorbidity (HTN-first).",
        "- If fail → GREEN v2 and/or label collapse before Path B.",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    (art / "REPORT.md").write_text("\n".join(report))
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done in {elapsed/60:.1f} min → {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
