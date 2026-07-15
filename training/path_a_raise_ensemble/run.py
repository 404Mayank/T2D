"""Orchestrate Path A ensemble raise (smoke + full)."""

from __future__ import annotations

import argparse
import hashlib
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
import yaml

from training.path_a_raise_ensemble.bag import (
    bag_meta,
    fit_family_bag,
    save_bag_models,
)
from training.path_a_raise_ensemble.data import (
    assert_and_load_parent,
    load_c1_matrix,
    load_config,
    require_lgbm_gpu,
)
from training.path_a_raise_ensemble.ensemble import arith_mean, geom_mean
from training.path_a_raise_ensemble.metrics_raise import (
    c3_bag_cat,
    c3_e_arith,
    evaluate_arm_vs_parent,
    near_bar_trigger,
)
from training.path_a_raise_ensemble.stack import (
    StackResult,
    build_train_oof,
    fit_stacker,
)
from training.path_a_watch.evaluate import write_json
from training.path_a_watch.metrics import full_report, macro_ovr_auc


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


def _save_proba(path: Path, arr: np.ndarray, pids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, proba=np.asarray(arr, dtype=np.float64), person_id=pids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A raise: multi-seed + ensemble")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke: 2 seeds, K=2, no claim")
    ap.add_argument(
        "--skip-stack",
        action="store_true",
        help="skip E_stack OOF (faster debug; full default includes stack)",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds override (e.g. 42,43,...,51 for S10)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    claim_eligible = not args.quick
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "ens_%Y%m%d_%H%M%S" + ("_quick" if args.quick else "")
    )
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    art.mkdir(parents=True, exist_ok=True)

    # quick / seed overrides
    seeds = list(cfg["run"]["seeds"])
    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if not seeds:
            raise ValueError("--seeds parsed empty")
    k_folds = int(cfg["stack"]["k_folds"])
    C_grid = list(cfg["stack"]["C_grid"])
    if args.quick:
        seeds = seeds[:2]
        k_folds = 2
        C_grid = [1.0]
        cfg["run"]["bootstrap_n"] = min(int(cfg["run"]["bootstrap_n"]), 200)
        cfg["run"]["min_n_boot_ok"] = min(int(cfg["run"]["min_n_boot_ok"]), 150)

    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    t0 = time.time()
    log(f"run_id={run_id} quick={args.quick} claim_eligible={claim_eligible}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")
    log(f"seeds={seeds} skip_stack={args.skip_stack}")

    if (art / "metrics_test.json").exists() or (art / "selected_ensemble.json").exists():
        raise RuntimeError(f"refuse overwrite existing run artifacts under {art}")

    # GPU lock
    lgbm_device = require_lgbm_gpu(cfg, quick=args.quick)
    log(f"lgbm_device={lgbm_device}")

    # Data + parent
    log("load C1 matrix")
    _df, feature_cols, splits, load_meta = load_c1_matrix(repo, cfg)
    log(
        f"features n={len(feature_cols)} splits "
        f"{load_meta['n_train']}/{load_meta['n_val']}/{load_meta['n_test']}"
    )

    log("assert parent C1")
    parent = assert_and_load_parent(repo, cfg, splits, feature_cols)
    write_json(art / "parent_assert.json", parent.assert_record)
    log(
        f"parent ok auc={parent.test_macro_ovr_auc:.4f} "
        f"bin={parent.test_binary_auc:.4f} hash={parent.feature_hash}"
    )

    write_json(
        art / "features.json",
        {
            "feature_cols": feature_cols,
            "feature_hash": _feature_hash(feature_cols),
            "parent_feature_hash": parent.feature_hash,
            "feature_set": "scores",
            "n_features": len(feature_cols),
            **load_meta,
        },
    )

    # Bags
    log("fit Bag_lgbm")
    bag_lgbm = fit_family_bag(
        splits,
        family="lightgbm",
        seeds=seeds,
        params=parent.lgbm_params,
        cfg_run=cfg["run"],
        class_weights=cfg["class_weights"],
        lgbm_device=lgbm_device,
        log=log,
    )
    log("fit Bag_cat")
    bag_cat = fit_family_bag(
        splits,
        family="catboost",
        seeds=seeds,
        params=parent.cat_params,
        cfg_run=cfg["run"],
        class_weights=cfg["class_weights"],
        lgbm_device=lgbm_device,
        log=log,
    )

    models_dir = art / "models"
    save_bag_models(bag_lgbm, models_dir)
    save_bag_models(bag_cat, models_dir)
    write_json(art / "bag_lgbm.json", bag_meta(bag_lgbm))
    write_json(art / "bag_cat.json", bag_meta(bag_cat))

    # Single-seed diagnostics (seed 42 members)
    s_cat = next(f for f in bag_cat.fits if f.seed == seeds[0])
    s_lgbm = next(f for f in bag_lgbm.fits if f.seed == seeds[0])
    # Prefer seed 42 if present
    for f in bag_cat.fits:
        if f.seed == 42:
            s_cat = f
            break
    for f in bag_lgbm.fits:
        if f.seed == 42:
            s_lgbm = f
            break

    s_cat_vs_b0 = {
        "seed": s_cat.seed,
        "val_macro_ovr_auc": s_cat.val_macro_ovr_auc,
        "test_macro_ovr_auc": s_cat.test_macro_ovr_auc,
        "test_auc_delta_vs_b0": float(
            s_cat.test_macro_ovr_auc - parent.test_macro_ovr_auc
        ),
        "note": "refit path; bit-match to frozen joblib not required but delta logged",
    }
    write_json(art / "s_cat_diagnostic.json", s_cat_vs_b0)
    log(
        f"S_cat seed={s_cat.seed} test_auc={s_cat.test_macro_ovr_auc:.4f} "
        f"ΔB0={s_cat_vs_b0['test_auc_delta_vs_b0']:+.4f}"
    )

    # Blends
    assert bag_lgbm.proba_val is not None and bag_cat.proba_val is not None
    assert bag_lgbm.proba_test is not None and bag_cat.proba_test is not None
    assert bag_lgbm.proba_train is not None and bag_cat.proba_train is not None

    p_arith_val = arith_mean(bag_lgbm.proba_val, bag_cat.proba_val)
    p_arith_test = arith_mean(bag_lgbm.proba_test, bag_cat.proba_test)
    p_arith_train = arith_mean(bag_lgbm.proba_train, bag_cat.proba_train)
    eps = float(cfg["run"]["geom_eps"])
    p_geom_val = geom_mean(bag_lgbm.proba_val, bag_cat.proba_val, eps=eps)
    p_geom_test = geom_mean(bag_lgbm.proba_test, bag_cat.proba_test, eps=eps)

    # Stack (optional skip)
    stack_res: StackResult | None = None
    stack_meta: dict[str, Any] | None = None
    if not args.skip_stack:
        log(f"stack OOF K={k_folds} seeds={seeds}")
        oof_feats, fold_counts = build_train_oof(
            splits,
            seeds=seeds,
            lgbm_params=parent.lgbm_params,
            cat_params=parent.cat_params,
            cfg_run=cfg["run"],
            class_weights=cfg["class_weights"],
            lgbm_device=lgbm_device,
            k_folds=k_folds,
            fold_seed=int(cfg["run"]["seed"]),
            log=log,
        )
        stack_res = fit_stacker(
            splits,
            oof_train_feats=oof_feats,
            bag_lgbm_val=bag_lgbm.proba_val,
            bag_cat_val=bag_cat.proba_val,
            bag_lgbm_test=bag_lgbm.proba_test,
            bag_cat_test=bag_cat.proba_test,
            bag_lgbm_train=bag_lgbm.proba_train,
            bag_cat_train=bag_cat.proba_train,
            C_grid=[float(c) for c in C_grid],
            max_iter=int(cfg["stack"]["max_iter"]),
            degeneracy_l2_ratio_min=float(cfg["stack"]["degeneracy_l2_ratio_min"]),
            collapse_val_margin=float(cfg["stack"]["collapse_val_margin"]),
            bag_cat_val_auc=float(bag_cat.val_macro_ovr_auc or 0.0),
            bag_lgbm_val_auc=float(bag_lgbm.val_macro_ovr_auc or 0.0),
            log=log,
        )
        stack_res.fold_label_counts = fold_counts
        joblib.dump(stack_res.model, art / "stacker.joblib")
        stack_meta = {
            "C": stack_res.C,
            "grid": stack_res.grid,
            "val_macro_ovr_auc": stack_res.val_macro_ovr_auc,
            "test_macro_ovr_auc": stack_res.test_macro_ovr_auc,
            "oof_train_auc_proxy": stack_res.oof_val_auc_proxy,
            "degeneracy": stack_res.degeneracy,
            "claim_eligible": stack_res.claim_eligible and claim_eligible,
            "fold_label_counts": fold_counts,
            "coef_shape": list(stack_res.model.coef_.shape),
        }
        write_json(art / "stacker_meta.json", stack_meta)
        proba_dir_early = art / "proba"
        proba_dir_early.mkdir(parents=True, exist_ok=True)
        np.save(
            str(proba_dir_early / "oof_train_stack_features.npy"),
            oof_feats,
        )
    else:
        log("skip stack")

    # Freeze before test metrics write
    freeze = {
        "run_id": run_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "claim_eligible": claim_eligible,
        "seeds": seeds,
        "parent_c1_run_id": parent.run_id,
        "feature_hash": parent.feature_hash,
        "primary_a": "Bag_cat",
        "primary_b": "E_arith",
        "skip_stack": bool(args.skip_stack),
        "stack_C": None if stack_res is None else stack_res.C,
        "bag_cat_val_auc": bag_cat.val_macro_ovr_auc,
        "bag_lgbm_val_auc": bag_lgbm.val_macro_ovr_auc,
        "e_arith_val_auc": float(macro_ovr_auc(splits.y_val, p_arith_val)),
        "e_geom_val_auc": float(macro_ovr_auc(splits.y_val, p_geom_val)),
        "lgbm_device": lgbm_device,
        "n_features": len(feature_cols),
    }
    write_json(art / "selected_ensemble.json", freeze)
    log(f"FREEZE {art / 'selected_ensemble.json'}")

    # Persist probas
    proba_dir = art / "proba"
    _save_proba(proba_dir / "Bag_cat_test.npz", bag_cat.proba_test, splits.person_id_test)
    _save_proba(proba_dir / "Bag_lgbm_test.npz", bag_lgbm.proba_test, splits.person_id_test)
    _save_proba(proba_dir / "E_arith_test.npz", p_arith_test, splits.person_id_test)
    _save_proba(proba_dir / "E_geom_test.npz", p_geom_test, splits.person_id_test)
    _save_proba(proba_dir / "B0_test.npz", parent.proba_test, splits.person_id_test)
    if stack_res is not None:
        _save_proba(proba_dir / "E_stack_test.npz", stack_res.proba_test, splits.person_id_test)

    # Evaluate arms
    boot_n = int(cfg["run"]["bootstrap_n"])
    boot_seed = int(cfg["run"]["bootstrap_seed"])
    paired_seed = int(cfg["run"]["paired_bootstrap_seed"])
    min_ok = int(cfg["run"]["min_n_boot_ok"])
    d_bar = float(cfg["bar"]["delta_auc"])

    def _eval(tag: str, p: np.ndarray) -> dict[str, Any]:
        return evaluate_arm_vs_parent(
            splits.y_test,
            p,
            parent.proba_test,
            parent.test_macro_ovr_auc,
            parent.test_binary_auc,
            tag=tag,
            bootstrap_n=boot_n,
            bootstrap_seed=boot_seed,
            paired_bootstrap_seed=paired_seed,
            min_n_boot_ok=min_ok,
            delta_auc_bar=d_bar,
        )

    arms: dict[str, dict[str, Any]] = {}
    arms["B0"] = {
        "tag": "B0",
        "selected_raw": full_report(splits.y_test, parent.proba_test, tag="B0"),
        "note": "frozen C1 recompute",
    }
    arms["S_cat"] = _eval("S_cat", s_cat.proba_test)
    arms["S_lgbm"] = _eval("S_lgbm", s_lgbm.proba_test)
    arms["Bag_cat"] = _eval("Bag_cat", bag_cat.proba_test)
    arms["Bag_lgbm"] = _eval("Bag_lgbm", bag_lgbm.proba_test)
    arms["E_arith"] = _eval("E_arith", p_arith_test)
    arms["E_geom"] = _eval("E_geom", p_geom_test)
    if stack_res is not None:
        arms["E_stack"] = _eval("E_stack", stack_res.proba_test)
        arms["E_stack"]["stack_claim_eligible"] = bool(
            stack_res.claim_eligible and claim_eligible
        )
        arms["E_stack"]["degeneracy"] = stack_res.degeneracy

    # c3 for primaries
    c3_a = c3_bag_cat(
        [f.val_macro_ovr_auc for f in bag_cat.fits],
        s_cat_val_auc=float(s_cat.val_macro_ovr_auc),
        bag_val_auc=float(bag_cat.val_macro_ovr_auc or 0.0),
        c1_freeze_val_auc=float(
            json_load_val_auc(parent.art / "best_params_catboost.json")
        ),
    )
    # best bag by val
    if (bag_cat.val_macro_ovr_auc or 0) >= (bag_lgbm.val_macro_ovr_auc or 0):
        best_name, best_p = "Bag_cat", bag_cat.proba_test
    else:
        best_name, best_p = "Bag_lgbm", bag_lgbm.proba_test
    c3_b = c3_e_arith(
        splits.y_test,
        p_arith_test,
        best_p,
        best_bag_name=best_name,
        bootstrap_n=boot_n,
        paired_bootstrap_seed=paired_seed,
        min_n_boot_ok=min_ok,
    )

    def _primary_pass(arm_key: str, c3: dict[str, Any]) -> dict[str, Any]:
        a = arms[arm_key]
        c1 = bool(a["delta_vs_c1"]["criterion1_point_delta_gt_0p01"])
        c2 = bool(a["delta_vs_c1"]["criterion2_bootstrap_delta_auc_lo_gt_0"])
        c3p = bool(c3["criterion3_pass"])
        return {
            "arm": arm_key,
            "c1": c1,
            "c2": c2,
            "c3": c3p,
            "decision_bar_pass": bool(c1 and c2 and c3p and claim_eligible),
            "c3_detail": c3,
            "delta_macro_ovr_auc": a["delta_vs_c1"]["delta_macro_ovr_auc"],
            "delta_binary_auc": a["delta_vs_c1"]["delta_binary_auc"],
            "test_macro_ovr_auc": a["selected_raw"]["macro_ovr_auc"],
            "test_binary_auc": a["selected_raw"]["binary_auc"],
            "near_bar_s10_trigger": near_bar_trigger(
                float(a["delta_vs_c1"]["delta_macro_ovr_auc"]), c1, c2
            ),
        }

    primary_a = _primary_pass("Bag_cat", c3_a)
    primary_b = _primary_pass("E_arith", c3_b)

    # Sibling exploratory flag
    for sib in ("E_geom", "E_stack", "Bag_lgbm"):
        if sib not in arms:
            continue
        if "delta_vs_c1" not in arms[sib]:
            continue
        c1 = arms[sib]["delta_vs_c1"]["criterion1_point_delta_gt_0p01"]
        c2 = arms[sib]["delta_vs_c1"]["criterion2_bootstrap_delta_auc_lo_gt_0"]
        arms[sib]["exploratory_only"] = True
        arms[sib]["sibling_c1_c2"] = bool(c1 and c2)
        arms[sib]["claim_eligible"] = False

    metrics_val = {
        "Bag_cat": bag_cat.val_macro_ovr_auc,
        "Bag_lgbm": bag_lgbm.val_macro_ovr_auc,
        "E_arith": float(macro_ovr_auc(splits.y_val, p_arith_val)),
        "E_geom": float(macro_ovr_auc(splits.y_val, p_geom_val)),
        "S_cat": s_cat.val_macro_ovr_auc,
        "S_lgbm": s_lgbm.val_macro_ovr_auc,
        "E_stack": None if stack_res is None else stack_res.val_macro_ovr_auc,
        "per_seed_cat": [f.val_macro_ovr_auc for f in bag_cat.fits],
        "per_seed_lgbm": [f.val_macro_ovr_auc for f in bag_lgbm.fits],
    }
    write_json(art / "metrics_val.json", metrics_val)

    metrics_test: dict[str, Any] = {
        "phase": "path_a_raise_ensemble",
        "claim_eligible": claim_eligible,
        "parent_c1_run_id": parent.run_id,
        "parent_test_macro_ovr_auc": parent.test_macro_ovr_auc,
        "parent_test_binary_auc": parent.test_binary_auc,
        "seeds": seeds,
        "arms": arms,
        "primary_a": primary_a,
        "primary_b": primary_b,
        "decision_summary": {
            "primary_a_pass": primary_a["decision_bar_pass"],
            "primary_b_pass": primary_b["decision_bar_pass"],
            "any_primary_pass": bool(
                primary_a["decision_bar_pass"] or primary_b["decision_bar_pass"]
            ),
            "s10_trigger": bool(
                primary_a["near_bar_s10_trigger"] or primary_b["near_bar_s10_trigger"]
            ),
        },
    }
    write_json(art / "metrics_test.json", metrics_test)

    # Manifest + REPORT
    manifest = {
        "run_id": run_id,
        "claim_eligible": claim_eligible,
        "quick": bool(args.quick),
        "skip_stack": bool(args.skip_stack),
        "seeds": seeds,
        "parent_c1_run_id": parent.run_id,
        "git_hash": _git_hash(repo),
        "platform": platform.platform(),
        "python": sys.version,
        "lgbm_device": lgbm_device,
        "elapsed_sec": time.time() - t0,
        "DRI_PRIME": os.environ.get("DRI_PRIME"),
    }
    write_json(art / "run_manifest.json", manifest)

    report = _render_report(
        run_id=run_id,
        claim_eligible=claim_eligible,
        parent_auc=parent.test_macro_ovr_auc,
        parent_bin=parent.test_binary_auc,
        primary_a=primary_a,
        primary_b=primary_b,
        metrics_val=metrics_val,
        arms=arms,
        seeds=seeds,
    )
    (art / "REPORT.md").write_text(report)
    (art / "run.log").write_text("\n".join(logs) + "\n")

    log(
        f"DONE primary_A_pass={primary_a['decision_bar_pass']} "
        f"primary_B_pass={primary_b['decision_bar_pass']} "
        f"Bag_cat Δ={primary_a['delta_macro_ovr_auc']:+.4f} "
        f"E_arith Δ={primary_b['delta_macro_ovr_auc']:+.4f} "
        f"elapsed={time.time()-t0:.1f}s"
    )
    print(report)
    return 0


def json_load_val_auc(path: Path) -> float:
    import json

    d = json.loads(path.read_text())
    return float(d.get("val_macro_ovr_auc") or d.get("selected_val_macro_ovr_auc") or 0.0)


def _render_report(
    *,
    run_id: str,
    claim_eligible: bool,
    parent_auc: float,
    parent_bin: float,
    primary_a: dict[str, Any],
    primary_b: dict[str, Any],
    metrics_val: dict[str, Any],
    arms: dict[str, Any],
    seeds: list[int],
) -> str:
    lines = [
        f"# Ensemble raise report — `{run_id}`",
        "",
        f"**claim_eligible:** {claim_eligible}",
        f"**Parent C1:** 4-AUC {parent_auc:.4f} / binary {parent_bin:.4f}",
        f"**Seeds:** {seeds}",
        "",
        "## Primaries vs C1",
        "",
        "| Primary | Arm | Test 4-AUC | Binary | ΔAUC | Δbin | c1 | c2 | c3 | bar |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for name, p in [("A", primary_a), ("B", primary_b)]:
        lines.append(
            f"| {name} | {p['arm']} | {p['test_macro_ovr_auc']:.4f} | "
            f"{p['test_binary_auc']:.4f} | {p['delta_macro_ovr_auc']:+.4f} | "
            f"{p['delta_binary_auc']:+.4f} | {p['c1']} | {p['c2']} | {p['c3']} | "
            f"**{p['decision_bar_pass']}** |"
        )
    lines += [
        "",
        "## Val AUCs",
        "",
        f"- Bag_cat: {metrics_val['Bag_cat']}",
        f"- Bag_lgbm: {metrics_val['Bag_lgbm']}",
        f"- E_arith: {metrics_val['E_arith']}",
        f"- E_geom: {metrics_val['E_geom']}",
        f"- E_stack: {metrics_val['E_stack']}",
        "",
        "## All arms (test)",
        "",
    ]
    for k, a in arms.items():
        if "selected_raw" not in a:
            continue
        raw = a["selected_raw"]
        d = a.get("delta_vs_c1", {})
        lines.append(
            f"- **{k}**: auc={raw['macro_ovr_auc']:.4f} bin={raw['binary_auc']:.4f} "
            f"Δauc={d.get('delta_macro_ovr_auc', float('nan'))} "
            f"Δbin={d.get('delta_binary_auc', float('nan'))}"
        )
    lines += [
        "",
        f"S10 trigger A: {primary_a['near_bar_s10_trigger']} "
        f"B: {primary_b['near_bar_s10_trigger']}",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
