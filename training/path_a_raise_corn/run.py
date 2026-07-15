"""Orchestrate CORN / CE-MLP raise runs (smoke or full).

Usage:
  .venv/bin/python -m training.path_a_raise_corn --self-check
  .venv/bin/python -m training.path_a_raise_corn --quick --arm corn
  .venv/bin/python -m training.path_a_raise_corn --arm both
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.path_a_raise_corn.data import (
    apply_impute_scale,
    assert_parent_c1,
    fh_post_impute_rates,
    fit_impute_scale,
    load_c1_matrix,
    load_config,
    load_parent_c1_proba,
    split_frames,
)
from training.path_a_raise_corn.evaluate import (
    arm_vs_arm_delta,
    assert_proba_contract,
    compare_to_parent,
)
from training.path_a_raise_corn.explain import permutation_importance_mlp
from training.path_a_raise_corn.losses_proba import self_check
from training.path_a_raise_corn.train import (
    hpo_arm,
    predict_hard_arm,
    predict_proba_arm,
    train_fixed,
)
from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.evaluate import write_json
from training.path_a_watch.metrics import full_report


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


def _device_str(cfg: dict[str, Any]) -> torch.device:
    name = cfg["run"].get("device", "auto")
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_arm(
    *,
    arm: str,
    repo: Path,
    cfg: dict[str, Any],
    run_id: str,
    quick: bool,
    skip_perm: bool,
    n_trials: int | None,
    parent: dict[str, Any],
    splits: dict[str, Any],
    scale_state: dict[str, Any],
    Ztr: np.ndarray,
    Zva: np.ndarray,
    Zte: np.ndarray,
    fh_rates: dict[str, Any],
    bundle_meta: dict[str, Any],
) -> dict[str, Any]:
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    art.mkdir(parents=True, exist_ok=True)
    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError(f"refuse overwrite {art}")

    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    t0 = time.time()
    log(f"run_id={run_id} arm={arm} quick={quick}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")
    log(f"feature_hash={bundle_meta['feature_hash']} n_feat={len(splits['feature_cols'])}")
    log(f"fh_post_impute={json.dumps(fh_rates)}")

    write_json(
        art / "features.json",
        {
            **bundle_meta,
            "arm": arm,
            "fh_post_impute_rates": fh_rates,
            "scale_state_keys": list(scale_state.keys()),
        },
    )
    write_json(art / "scale_state.json", scale_state)

    ytr, yva, yte = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]
    n_trials_eff = int(
        n_trials
        if n_trials is not None
        else (2 if quick else cfg["run"]["n_trials_per_arm"])
    )

    if quick:
        cfg_q = dict(cfg)
        cfg_q["run"] = dict(cfg["run"])
        cfg_q["run"]["smoke_max_epochs"] = 15
        cfg_q["run"]["smoke_patience"] = 5
        pack = train_fixed(Ztr, ytr, Zva, yva, cfg_q, arm=arm, log=log)
        pack["n_trials"] = 0
        pack["trial_number"] = None
    else:
        pack = hpo_arm(
            Ztr,
            ytr,
            Zva,
            yva,
            cfg,
            arm=arm,
            n_trials=n_trials_eff,
            seed=int(cfg["run"]["seed"]),
            log=log,
        )

    log(
        f"VAL {arm} auc={pack['val_macro_ovr_auc']:.4f} "
        f"auprc={pack['val_macro_auprc']:.4f} best_ep={pack['best_epoch']}"
    )

    freeze = {
        "run_id": run_id,
        "phase": "path_a_raise_corn",
        "arm": arm,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": pack["family"],
        "params": pack["params"],
        "trial_number": pack.get("trial_number"),
        "n_trials": pack.get("n_trials"),
        "best_epoch": pack["best_epoch"],
        "val_macro_ovr_auc": pack["val_macro_ovr_auc"],
        "val_macro_auprc": pack["val_macro_auprc"],
        "feature_cols": splits["feature_cols"],
        "feature_hash": splits["feature_hash"],
        "parent_c1_run_id": cfg["paths"]["parent_c1_run_id"],
        "imbalance": cfg["run"]["imbalance"],
        "quick": bool(quick),
        "seed": int(cfg["run"]["seed"]),
    }
    # FREEZE before test metrics
    write_json(art / "selected_model.json", freeze)
    log(f"FREEZE {art / 'selected_model.json'}")

    models_dir = art / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": pack["model"].state_dict(),
            "arm": arm,
            "params": pack["params"],
            "d_in": Ztr.shape[1],
            "n_classes": int(cfg["data"]["n_classes"]),
        },
        models_dir / "selected.pt",
    )

    dev = _device_str(cfg)
    pack["model"].to(dev)
    proba_val = predict_proba_arm(pack["model"], Zva, arm, dev)
    proba_test = predict_proba_arm(pack["model"], Zte, arm, dev)
    hard_test = predict_hard_arm(pack["model"], Zte, arm, dev)
    assert_proba_contract(proba_test)

    # perm on val (before test write is fine; does not use test)
    if skip_perm or quick:
        perm = {
            "perm_stable": True if quick else False,
            "skipped": True,
            "reason": "quick/skip_perm",
        }
        perm_stable = True if quick else False
    else:
        log("permutation importance on val")

        def _pf(X: np.ndarray) -> np.ndarray:
            return predict_proba_arm(pack["model"], X, arm, dev)

        perm = permutation_importance_mlp(
            _pf,
            Zva,
            yva,
            splits["feature_cols"],
            n_repeats=3,
            seed=int(cfg["run"]["seed"]),
        )
        perm_stable = bool(perm["perm_stable"])
        write_json(art / "perm_importance.json", perm)
        log(f"perm_stable={perm_stable} mean_drop={perm['mean_perm_auc_drop']:.4f}")

    # parent proba on test pids
    proba_parent = load_parent_c1_proba(
        repo, cfg, parent, splits["test"]["pid"], tol=1e-6
    )
    log(f"parent C1 recompute ok")

    bar_eligible = arm == "corn" and not quick
    cmp = compare_to_parent(
        yte,
        proba_test,
        proba_parent,
        parent_class2=float(parent["class2_ovr_auc"]),
        cfg=cfg,
        perm_stable=perm_stable if bar_eligible else False,
        n_boot=int(cfg["run"]["bootstrap_n"]) if not quick else 50,
        seed=int(cfg["run"]["seed"]),
    )
    if not bar_eligible:
        cmp["delta_vs_c1"]["decision_bar_pass"] = False
        cmp["delta_vs_c1"]["bar_eligible"] = False
        cmp["delta_vs_c1"]["bar_note"] = "smoke or non-CORN arm — not bar-eligible"
    else:
        cmp["delta_vs_c1"]["bar_eligible"] = True

    # calibration diagnostic
    cal = fit_calibrators(
        proba_val,
        yva,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test_cal = cal["primary"].transform(proba_test)
    cal_rep = full_report(yte, proba_test_cal, tag="test_cal_primary")

    metrics = {
        "arm": arm,
        "run_id": run_id,
        "quick": bool(quick),
        "bar_eligible": bar_eligible,
        "selected_family": pack["family"],
        "val": {
            "macro_ovr_auc": pack["val_macro_ovr_auc"],
            "macro_auprc": pack["val_macro_auprc"],
            "raw": full_report(yva, proba_val, tag="val_raw"),
        },
        "selected_raw": cmp["selected_raw"],
        "selected_cal": cal_rep,
        "delta_vs_c1": cmp["delta_vs_c1"],
        "soft_class2": cmp["soft_class2"],
        "hard_label_test": full_report(
            yte, proba_test, y_pred=hard_test, tag="test_hard_labels"
        )["ordinal"],
        "perm": {
            "perm_stable": perm_stable,
            "mean_perm_auc_drop": perm.get("mean_perm_auc_drop"),
            "skipped": perm.get("skipped", False),
        },
        "parent_c1": {
            "run_id": parent["run_id"],
            "test_macro_ovr_auc": parent["test_macro_ovr_auc"],
            "class2_ovr_auc": parent["class2_ovr_auc"],
        },
    }
    write_json(art / "metrics_test.json", metrics)
    # also save proba for later CORN-vs-CE
    np.savez_compressed(
        art / "proba_test.npz",
        proba=proba_test,
        proba_parent=proba_parent,
        y=yte,
        pid=splits["test"]["pid"],
        hard=hard_test,
    )

    manifest = {
        "run_id": run_id,
        "arm": arm,
        "quick": quick,
        "seconds": time.time() - t0,
        "git": _git_hash(repo),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "feature_hash": splits["feature_hash"],
        "decision_bar_pass": metrics["delta_vs_c1"].get("decision_bar_pass"),
        "test_macro_ovr_auc": metrics["selected_raw"]["macro_ovr_auc"],
        "test_binary_auc": metrics["selected_raw"]["binary_auc"],
    }
    write_json(art / "run_manifest.json", manifest)
    (art / "run.log").write_text("\n".join(logs) + "\n")

    log(
        f"TEST auc={metrics['selected_raw']['macro_ovr_auc']:.4f} "
        f"bin={metrics['selected_raw']['binary_auc']:.4f} "
        f"c2={metrics['soft_class2']['class2_ovr_auc']:.4f} "
        f"d_auc={metrics['delta_vs_c1']['point_delta_macro_ovr_auc']:+.4f} "
        f"bar={metrics['delta_vs_c1'].get('decision_bar_pass')}"
    )
    log(f"done in {manifest['seconds']:.1f}s → {art}")
    return {
        "art": str(art),
        "metrics": metrics,
        "proba_test": proba_test,
        "y_test": yte,
        "run_id": run_id,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A raise CORN/CE MLP")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--arm",
        type=str,
        default="both",
        choices=["corn", "ce", "coral", "both"],
    )
    ap.add_argument("--quick", action="store_true", help="smoke: fixed hparams, short epochs")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-perm", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)

    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return 0

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    parent = assert_parent_c1(repo, cfg)
    bundle = load_c1_matrix(repo, cfg)
    splits = split_frames(bundle, cfg)
    scale_state = fit_impute_scale(splits["train"]["X"])
    Ztr = apply_impute_scale(splits["train"]["X"], scale_state)
    Zva = apply_impute_scale(splits["val"]["X"], scale_state)
    Zte = apply_impute_scale(splits["test"]["X"], scale_state)
    if np.isnan(Ztr).any() or np.isnan(Zva).any() or np.isnan(Zte).any():
        raise AssertionError("NaNs remain after impute/scale")
    fh_rates = fh_post_impute_rates(splits["train"]["X"], scale_state)

    bundle_meta = {
        "feature_cols": splits["feature_cols"],
        "feature_hash": splits["feature_hash"],
        "n_total": len(splits["feature_cols"]),
        "n_watch": len(bundle.watch_cols),
        "n_onboarding": len(bundle.onboard_cols),
        "n_mood": len(bundle.mood_cols),
        "nulls_before": bundle.nulls_before,
        "n_train": splits["train"]["n"],
        "n_val": splits["val"]["n"],
        "n_test": splits["test"]["n"],
    }

    arms: list[str]
    if args.arm == "both":
        arms = ["corn", "ce"]
    else:
        arms = [args.arm]

    results: dict[str, Any] = {}
    for arm in arms:
        prefix = {
            "corn": "corn_smoke" if args.quick else "corn_full",
            "ce": "ce_mlp_smoke" if args.quick else "ce_mlp_full",
            "coral": "coral_smoke" if args.quick else "coral_full",
        }[arm]
        run_id = args.run_id if (args.run_id and len(arms) == 1) else f"{prefix}_{ts}"
        # unique if both
        if args.run_id and len(arms) > 1:
            run_id = f"{args.run_id}_{arm}"
        results[arm] = run_arm(
            arm=arm,
            repo=repo,
            cfg=cfg,
            run_id=run_id,
            quick=args.quick,
            skip_perm=args.skip_perm or args.quick,
            n_trials=args.n_trials,
            parent=parent,
            splits=splits,
            scale_state=scale_state,
            Ztr=Ztr,
            Zva=Zva,
            Zte=Zte,
            fh_rates=fh_rates,
            bundle_meta=bundle_meta,
        )

    if "corn" in results and "ce" in results and not args.quick:
        y = results["corn"]["y_test"]
        d = arm_vs_arm_delta(
            y,
            results["corn"]["proba_test"],
            results["ce"]["proba_test"],
            n_boot=int(cfg["run"]["bootstrap_n"]),
            seed=int(cfg["run"]["seed"]),
        )
        out = {
            "corn_run_id": results["corn"]["run_id"],
            "ce_run_id": results["ce"]["run_id"],
            "corn_minus_ce": d,
            "interpretation_hint": (
                "CORN credit only if corn bar-pass AND corn_minus_ce point>0 "
                "with boot lo preferably >0"
            ),
        }
        comp_path = (
            repo
            / cfg["paths"]["artifacts_root"]
            / f"compare_corn_ce_{ts}.json"
        )
        write_json(comp_path, out)
        print(f"CORN−CE ΔAUC={d['point_delta_macro_ovr_auc']:+.4f} → {comp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
