"""CLI: Path A watch-only GBM floor.

Freeze contract: selected_model.json is written BEFORE any test metrics.
"""

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

from .calibrate import fit_calibrators
from .data import (
    COVERAGE_COLS_DEFAULT,
    describe_cohort,
    feature_columns,
    load_merged,
    make_splits,
)
from .evaluate import evaluate_split, reliability_diagrams, write_json
from .explain import permutation_on_val, shap_summary
from .hpo import pick_family, tune_catboost, tune_lightgbm
from .metrics import full_report
from .models import (
    best_iteration,
    fit_ordinal_logistic,
    predict_proba,
    resolve_lgbm_device,
)


def _repo_root_from_cwd() -> Path:
    return Path.cwd().resolve()


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
    for name in (
        "numpy",
        "pandas",
        "sklearn",
        "lightgbm",
        "catboost",
        "optuna",
        "shap",
        "statsmodels",
    ):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "missing"
    return out


def _feature_hash(cols: list[str]) -> str:
    h = hashlib.sha256(",".join(cols).encode()).hexdigest()
    return h[:16]


def _strip_model(pack: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pack.items() if k not in ("model", "val_report")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Path A watch-only GBM floor")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=None, help="override trials")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-ordinal", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke: 2 trials, skip shap/ordinal/ablation",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or _repo_root_from_cwd()).resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    art_root = repo / cfg["paths"]["artifacts_root"] / run_id
    art_root.mkdir(parents=True, exist_ok=True)

    if args.quick:
        cfg["run"]["n_trials"] = 2
        args.skip_shap = True
        args.skip_ordinal = True
        args.skip_ablation = True
    if args.n_trials is not None:
        cfg["run"]["n_trials"] = int(args.n_trials)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)

    t0 = time.time()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"run_id={run_id}")
    log(f"repo={repo}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")
    log(f"artifacts={art_root}")

    # --- data ---
    df = load_merged(
        repo,
        cfg["paths"]["watch_green"],
        cfg["paths"]["pool_masks"],
        expected_n=int(cfg["data"]["expected_n"]),
        id_col=cfg["data"]["id_col"],
    )
    fcols_full = feature_columns(
        df,
        feature_set="full_green",
        id_col=cfg["data"]["id_col"],
        deny_cols=cfg["data"]["deny_cols"],
        coverage_cols=cfg["data"]["coverage_cols"],
    )
    cohort = describe_cohort(df, fcols_full)
    write_json(art_root / "cohort.json", cohort)
    log(f"cohort n={cohort['n']} features={cohort['n_features']}")

    splits = make_splits(df, fcols_full, feature_set="full_green")
    log(
        f"split sizes train/val/test="
        f"{len(splits.y_train)}/{len(splits.y_val)}/{len(splits.y_test)}"
    )

    # --- device ---
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lightgbm device={lgbm_device}")

    # --- HPO full_green ---
    log(f"HPO LightGBM n_trials={cfg['run']['n_trials']}")
    lgbm_pack = tune_lightgbm(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        cfg,
        device=lgbm_device,
    )
    log(
        f"LGBM best source={lgbm_pack['source']} "
        f"val_auc={lgbm_pack['val_macro_ovr_auc']:.4f} "
        f"val_auprc={lgbm_pack['val_macro_auprc']:.4f} "
        f"iter={lgbm_pack['best_iteration']}"
    )
    write_json(art_root / "best_params_lgbm.json", _strip_model(lgbm_pack))

    log(f"HPO CatBoost n_trials={cfg['run']['n_trials']}")
    cat_pack = tune_catboost(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        cfg,
    )
    log(
        f"CatBoost best source={cat_pack['source']} "
        f"boosting={cat_pack.get('boosting_type')} "
        f"val_auc={cat_pack['val_macro_ovr_auc']:.4f} "
        f"val_auprc={cat_pack['val_macro_auprc']:.4f} "
        f"iter={cat_pack['best_iteration']}"
    )
    write_json(art_root / "best_params_catboost.json", _strip_model(cat_pack))

    # --- val-select family ---
    selected = pick_family(
        [lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"])
    )
    log(
        f"VAL-SELECT family={selected['family']} "
        f"val_auc={selected['val_macro_ovr_auc']:.4f}"
    )

    # FREEZE before test (immutable — never rewrite this file)
    if (art_root / "metrics_test.json").exists():
        raise RuntimeError("metrics_test.json already exists — refusing to re-score test")
    freeze = {
        "run_id": run_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_set": "full_green",
        "feature_cols": fcols_full,
        "feature_hash": _feature_hash(fcols_full),
        "lgbm_device_probe": lgbm_device,
        "selected_device": selected.get("device"),
        "device_fallback": selected.get("device_fallback"),
        "selected_boosting_type": selected.get("boosting_type"),
        "selection_rule": (
            "global: max val macro_ovr_auc; among AUC>=best-eps max macro_auprc"
        ),
        "auc_tie_eps": cfg["run"]["auc_tie_eps"],
        "competitors_val_only": {
            "lightgbm": {
                "val_macro_ovr_auc": lgbm_pack["val_macro_ovr_auc"],
                "val_macro_auprc": lgbm_pack["val_macro_auprc"],
                "source": lgbm_pack.get("source"),
                "n_failures": lgbm_pack.get("n_failures"),
            },
            "catboost": {
                "val_macro_ovr_auc": cat_pack["val_macro_ovr_auc"],
                "val_macro_auprc": cat_pack["val_macro_auprc"],
                "source": cat_pack.get("source"),
                "boosting_type": cat_pack.get("boosting_type"),
                "n_failures": cat_pack.get("n_failures"),
            },
        },
    }
    freeze_path = art_root / "selected_model.json"
    if freeze_path.exists():
        raise RuntimeError(f"freeze already exists: {freeze_path}")
    write_json(freeze_path, freeze)
    log(f"FREEZE written (immutable): {freeze_path}")

    # save models
    models_dir = art_root / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(lgbm_pack["model"], models_dir / "lgbm_full_green.joblib")
    joblib.dump(cat_pack["model"], models_dir / "catboost_full_green.joblib")
    joblib.dump(selected["model"], models_dir / "selected_full_green.joblib")

    # --- calibration on val (selected) ---
    proba_val = predict_proba(selected["model"], splits.X_val)
    cal_bundle = fit_calibrators(
        proba_val,
        splits.y_val,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_val_cal = cal_bundle["primary"].transform(proba_val)
    proba_val_iso = cal_bundle["secondary"].transform(proba_val)

    metrics_val = {
        "selected_family": selected["family"],
        "lightgbm": lgbm_pack.get("val_report"),
        "catboost": cat_pack.get("val_report"),
        "selected_raw": full_report(splits.y_val, proba_val, tag="selected_val_raw"),
        "selected_cal_sigmoid": full_report(
            splits.y_val, proba_val_cal, tag="selected_val_cal_sigmoid"
        ),
        "selected_cal_isotonic": full_report(
            splits.y_val, proba_val_iso, tag="selected_val_cal_isotonic"
        ),
        "calibration_support": cal_bundle["support"],
        "isotonic_reliable": cal_bundle["isotonic_reliable"],
        "note": "val calibration metrics are post-HPO diagnostics; test is the claim",
    }
    write_json(art_root / "metrics_val.json", metrics_val)

    # --- TEST ONCE (selected family only — claim metrics) ---
    proba_test_raw = predict_proba(selected["model"], splits.X_test)
    proba_test_cal = cal_bundle["primary"].transform(proba_test_raw)
    proba_test_iso = cal_bundle["secondary"].transform(proba_test_raw)

    metrics_test = {
        "selected_family": selected["family"],
        "claim": "raw ranking metrics are primary; calibration is diagnostic",
        "selected_raw": full_report(splits.y_test, proba_test_raw, tag="selected_test_raw"),
        "selected_cal_sigmoid": full_report(
            splits.y_test, proba_test_cal, tag="selected_test_cal_sigmoid"
        ),
        "selected_cal_isotonic": full_report(
            splits.y_test, proba_test_iso, tag="selected_test_cal_isotonic"
        ),
        "isotonic_reliable": cal_bundle["isotonic_reliable"],
        "calibration_support_val": cal_bundle["support"],
    }
    write_json(art_root / "metrics_test.json", metrics_test)

    # Non-selected family test scores: appendix only, not for selection/claims
    proba_test_lgbm = predict_proba(lgbm_pack["model"], splits.X_test)
    proba_test_cat = predict_proba(cat_pack["model"], splits.X_test)
    appendix = {
        "note": (
            "Appendix only — family was selected on VAL before test. "
            "Do not cherry-pick these for the Path A claim."
        ),
        "selected_family": selected["family"],
        "lightgbm_raw": full_report(splits.y_test, proba_test_lgbm, tag="lgbm_test_raw"),
        "catboost_raw": full_report(splits.y_test, proba_test_cat, tag="cat_test_raw"),
    }
    write_json(art_root / "appendix_both_families_test.json", appendix)

    write_json(
        art_root / "selected_model_post.json",
        {
            "freeze_path": str(freeze_path),
            "test_scored": True,
            "test_scored_at": datetime.now(timezone.utc).isoformat(),
            "selected_family": selected["family"],
        },
    )
    log(
        f"TEST selected raw macro_ovr_auc="
        f"{metrics_test['selected_raw']['macro_ovr_auc']:.4f} "
        f"macro_auprc={metrics_test['selected_raw']['macro_auprc']:.4f} "
        f"binary_auc={metrics_test['selected_raw']['binary_auc']:.4f}"
    )

    cal_dir = art_root / "calibration"
    reliability_diagrams(
        splits.y_test,
        proba_test_raw,
        cal_dir / "reliability_test_raw.png",
        title=f"Reliability raw ({selected['family']})",
    )
    reliability_diagrams(
        splits.y_test,
        proba_test_cal,
        cal_dir / "reliability_test_sigmoid.png",
        title=f"Reliability sigmoid-cal ({selected['family']})",
    )

    # --- physio_only ablation (same HPs, no re-HPO) ---
    ablation = None
    if not args.skip_ablation:
        log("physio_only ablation (frozen HPs, no re-HPO)")
        fcols_phys = feature_columns(
            df,
            feature_set="physio_only",
            deny_cols=cfg["data"]["deny_cols"],
            coverage_cols=cfg["data"]["coverage_cols"],
        )
        sp = make_splits(df, fcols_phys, feature_set="physio_only")
        # refit selected family with same params
        from .models import fit_lgbm, make_lgbm, try_catboost_boosting

        if selected["family"] == "lightgbm":
            m = make_lgbm(
                selected["params"],
                seed=seed,
                n_jobs=int(cfg["run"]["n_jobs"]),
                device=lgbm_device,
                n_estimators=int(cfg["run"]["n_estimators_max"]),
                class_weight=cfg["class_weights"]["lightgbm"],
            )
            fit_lgbm(
                m,
                sp.X_train,
                sp.y_train,
                sp.X_val,
                sp.y_val,
                es_rounds=int(cfg["run"]["es_rounds"]),
            )
        else:
            m, bt = try_catboost_boosting(
                selected["params"],
                sp.X_train,
                sp.y_train,
                sp.X_val,
                sp.y_val,
                seed=seed,
                n_estimators=int(cfg["run"]["n_estimators_max"]),
                es_rounds=int(cfg["run"]["es_rounds"]),
                preferred=selected.get("boosting_type")
                or cfg["run"].get("catboost_boosting_type", "Ordered"),
                fallback=cfg["run"].get("catboost_boosting_fallback", "Plain"),
                auto_class_weights=cfg["class_weights"]["catboost"],
                task_type=cfg["run"].get("catboost_task_type", "CPU"),
            )
        p_te = predict_proba(m, sp.X_test)
        p_va = predict_proba(m, sp.X_val)
        ablation = {
            "feature_set": "physio_only",
            "n_features": len(fcols_phys),
            "dropped": list(COVERAGE_COLS_DEFAULT),
            "family": selected["family"],
            "params": selected["params"],
            "note": (
                "Same hyperparams as selected full_green; early stopping re-run "
                "on physio_only so best_iteration may differ."
            ),
            "full_green_best_iteration": selected.get("best_iteration"),
            "physio_best_iteration": best_iteration(m),
            "val": full_report(sp.y_val, p_va, tag="physio_val"),
            "test": full_report(sp.y_test, p_te, tag="physio_test"),
            "delta_test_macro_ovr_auc_vs_full": (
                full_report(sp.y_test, p_te)["macro_ovr_auc"]
                - metrics_test["selected_raw"]["macro_ovr_auc"]
            ),
        }
        write_json(art_root / "ablation_physio_only.json", ablation)
        joblib.dump(m, models_dir / "selected_physio_only.joblib")
        log(
            f"physio_only test auc={ablation['test']['macro_ovr_auc']:.4f} "
            f"delta={ablation['delta_test_macro_ovr_auc_vs_full']:+.4f}"
        )

    # --- ordinal logistic baseline ---
    ordinal = None
    if not args.skip_ordinal:
        log("ordinal logistic baseline")
        try:
            ord_pack = fit_ordinal_logistic(
                splits.X_train,
                splits.y_train,
                splits.X_val,
                maxiter=int(cfg["ordinal"]["maxiter"]),
                method=cfg["ordinal"]["method"],
            )
            op_va = ord_pack["predict_proba"](splits.X_val)
            op_te = ord_pack["predict_proba"](splits.X_test)
            ordinal = {
                "val": full_report(splits.y_val, op_va, tag="ordinal_val"),
                "test": full_report(splits.y_test, op_te, tag="ordinal_test"),
                "llf": ord_pack["llf"],
                "n_features": ord_pack["n_features"],
                "converged": ord_pack.get("converged"),
                "mle_retvals": ord_pack.get("mle_retvals"),
            }
            write_json(art_root / "ordinal_logistic.json", ordinal)
            if not ord_pack.get("converged", True):
                log("WARNING: ordinal logistic BFGS did not report convergence")
            log(
                f"ordinal test auc={ordinal['test']['macro_ovr_auc']:.4f} "
                f"qwk={ordinal['test']['ordinal']['qwk']:.4f} "
                f"converged={ordinal.get('converged')}"
            )
        except Exception as e:
            log(f"ordinal logistic FAILED: {e}")
            write_json(
                art_root / "ordinal_logistic.json",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

    # --- explain selected on val ---
    explain_out = {}
    if not args.skip_shap:
        log("SHAP + permutation on val (selected)")
        try:
            explain_out["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art_root / "shap",
                seed=seed,
                prefix=f"{selected['family']}_val",
            )
            log(f"SHAP top5={list(explain_out['shap']['top10'].items())[:5]}")
            log(f"coverage ranks={explain_out['shap'].get('coverage_ranks')}")
        except Exception as e:
            log(f"SHAP failed: {e}")
            explain_out["shap_error"] = str(e)
        try:
            explain_out["permutation"] = permutation_on_val(
                selected["model"],
                splits.X_val,
                splits.y_val,
                art_root / "shap",
                seed=seed,
                prefix=f"{selected['family']}_val",
            )
        except Exception as e:
            log(f"permutation importance failed: {e}")
            explain_out["perm_error"] = str(e)
        write_json(art_root / "explain.json", explain_out)

    # --- both families always reported on test already ---

    elapsed = time.time() - t0
    manifest = {
        "run_id": run_id,
        "elapsed_sec": elapsed,
        "git_hash": _git_hash(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "DRI_PRIME": os.environ.get("DRI_PRIME"),
        "packages": _pkg_versions(),
        "seed": seed,
        "n_trials": cfg["run"]["n_trials"],
        "lgbm_device": lgbm_device,
        "cohort_n": cohort["n"],
        "base_rates": cohort["base_rates"],
        "site_by_label_train": cohort["site_by_label_train"],
        "feature_hash_full_green": _feature_hash(fcols_full),
        "selected_family": selected["family"],
        "freeze_path": str(freeze_path),
        "framing": (
            "Risk stratification, not early diagnosis; wearables recorded after "
            "participants knew status. Watch-only claim; survey blocks deferred."
        ),
        "protocol": (
            "Fixed recommended_split + val HPO/ES/cal; test once after freeze; "
            "val-select family by macro-OVR AUC (AUPRC tie)."
        ),
    }
    write_json(art_root / "run_manifest.json", manifest)

    # short REPORT
    report = _render_report(
        run_id=run_id,
        selected=selected,
        metrics_test=metrics_test,
        metrics_val=metrics_val,
        ablation=ablation,
        ordinal=ordinal,
        explain_out=explain_out,
        manifest=manifest,
        lgbm_pack=lgbm_pack,
        cat_pack=cat_pack,
    )
    (art_root / "REPORT.md").write_text(report)
    (art_root / "run.log").write_text("\n".join(log_lines) + "\n")
    log(f"done in {elapsed/60:.1f} min → {art_root}")
    return 0


def _render_report(**kw: Any) -> str:
    sel = kw["selected"]
    mt = kw["metrics_test"]
    mv = kw["metrics_val"]
    ab = kw["ablation"]
    ord_ = kw["ordinal"]
    man = kw["manifest"]
    lg = kw["lgbm_pack"]
    cat = kw["cat_pack"]
    ex = kw["explain_out"]

    lines = [
        f"# Path A watch-only GBM floor — {kw['run_id']}",
        "",
        "## Framing",
        man["framing"],
        "",
        "## Protocol",
        man["protocol"],
        "",
        "## Val selection (pre-test freeze)",
        f"- LightGBM val AUC={lg['val_macro_ovr_auc']:.4f} AUPRC={lg['val_macro_auprc']:.4f} ({lg.get('source')})",
        f"- CatBoost val AUC={cat['val_macro_ovr_auc']:.4f} AUPRC={cat['val_macro_auprc']:.4f} ({cat.get('source')})",
        f"- **Selected: {sel['family']}**",
        "",
        "## Test (once)",
        f"- Selected raw macro-OVR AUC: **{mt['selected_raw']['macro_ovr_auc']:.4f}**",
        f"- Selected raw macro AUPRC: **{mt['selected_raw']['macro_auprc']:.4f}**",
        f"- Binary AUC (1−P0): **{mt['selected_raw']['binary_auc']:.4f}**",
        f"- Multiclass Brier raw: {mt['selected_raw']['multiclass_brier']:.4f}",
        f"- Multiclass Brier sigmoid-cal: {mt['selected_cal_sigmoid']['multiclass_brier']:.4f}",
        f"- QWK: {mt['selected_raw']['ordinal']['qwk']:.4f}",
        "",
        "### Claim note",
        "- Primary claim metrics = **selected family raw** multiclass AUC/AUPRC above.",
        "- Calibration Brier/curves are diagnostic (may not improve ranking metrics).",
        "- Other family test scores (if present) live in appendix_both_families_test.json only.",
        "",
    ]
    if ab:
        lines += [
            "## physio_only ablation (coverage cols dropped, same HPs)",
            f"- Test AUC: {ab['test']['macro_ovr_auc']:.4f} (Δ {ab['delta_test_macro_ovr_auc_vs_full']:+.4f})",
            f"- Dropped: {ab['dropped']}",
            "",
        ]
    if ord_:
        lines += [
            "## Ordinal logistic baseline",
            f"- Test macro-OVR AUC: {ord_['test']['macro_ovr_auc']:.4f}",
            f"- Test QWK: {ord_['test']['ordinal']['qwk']:.4f}",
            f"- Converged: {ord_.get('converged')}",
            "",
            "## Test-set looks (pre-specified)",
            "1. Selected family claim (metrics_test.json)",
            "2. Appendix both families (not for claims)",
            "3. physio_only ablation",
            "4. Ordinal logistic baseline",
            "",
        ]
    if ex.get("shap"):
        lines += [
            "## SHAP (val)",
            f"- Coverage feature ranks: {ex['shap'].get('coverage_ranks')}",
            f"- Top features: {list(ex['shap']['top10'].keys())[:10]}",
            "",
        ]
    lines += [
        "## Notes",
        f"- Val metrics are post-HPO diagnostics (see metrics_val.json note).",
        f"- Site×label train: see cohort.json / run_manifest.json.",
        f"- Git: {man.get('git_hash')}",
        f"- Packages: {man.get('packages')}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
