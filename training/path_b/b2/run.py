"""Path B2 CLI — two-stage glucose emulator → T2D."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from training.path_a_watch.evaluate import write_json
from training.path_a_watch.models import resolve_lgbm_device
from training.path_b.b2.data import (
    assert_no_leakage,
    assert_oracle_features,
    load_b2_frame,
    merge_yhat_into_df,
    pred_col_names,
    split_xy,
    subsample_train_for_smoke,
    yhat_drift_table,
)
from training.path_b.b2.evaluate import apply_decision_bars, compare_arms, summarize_arm
from training.path_b.b2.stage1 import run_stage1
from training.path_b.b2.stage2 import (
    arm_feature_cols,
    arm_pool,
    is_deployable_arm,
    is_oracle_arm,
    train_stage2,
)


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


def _strip_model(pack: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "model",
        "calibrator",
        "proba_train",
        "proba_val",
        "proba_test",
        "proba_test_cal",
        "lgbm_pack_meta",
        "cat_pack_meta",
        "y_test",
        "pid_test",
    }
    return {k: v for k, v in pack.items() if k not in skip}


DEFAULT_FULL_ARMS = ["D0", "D1", "T0", "T1", "D1a", "O1"]
DEFAULT_SMOKE_ARMS = ["D1", "T1", "D1a", "O1"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path B2 two-stage ablation")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--n-trials", type=int, default=None, help="Stage-2 HPO trials/family")
    ap.add_argument("--stage1-n-trials", type=int, default=None)
    ap.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated arm ids (default: full or smoke set)",
    )
    ap.add_argument("--quick", action="store_true", help="Smoke: few trials + train subsample")
    ap.add_argument("--smoke-n-train", type=int, default=200)
    ap.add_argument("--skip-stage2", action="store_true", help="Only Stage-1")
    ap.add_argument(
        "--stage1-only",
        action="store_true",
        help="Alias of --skip-stage2",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    quick = bool(args.quick)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "b2_%Y%m%d_%H%M%S" + ("_smoke" if quick else "")
    )
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    if art.exists() and any(art.iterdir()):
        # allow re-entry only if empty-ish; refuse overwrite of metrics
        if (art / "decision_bars.json").exists() or (art / "stage1_metrics.json").exists():
            raise RuntimeError(f"refuse overwrite existing run dir {art}")
    art.mkdir(parents=True, exist_ok=True)

    if quick:
        cfg["run"]["n_trials"] = 5
        cfg["run"]["stage1_n_trials"] = min(5, int(cfg["run"]["stage1_n_trials"]))
    if args.n_trials is not None:
        cfg["run"]["n_trials"] = int(args.n_trials)
    if args.stage1_n_trials is not None:
        cfg["run"]["stage1_n_trials"] = int(args.stage1_n_trials)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    t0 = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"run_id={run_id} quick={quick}")
    log(f"repo={repo}")

    # --- data ---
    df, groups = load_b2_frame(repo, cfg)
    log(
        f"loaded core={len(df)} aux={int(df['aux_eligible'].sum())} "
        f"w0={len(groups['w0'])} c1={len(groups['c1'])}"
    )
    write_json(
        art / "c1_feature_manifest.json",
        {
            "n_feat": len(groups["c1"]),
            "feature_cols": groups["c1"],
            "w0": groups["w0"],
            "onboarding": groups["onboarding"],
            "mood": groups["mood"],
            "feature_hash": _feature_hash(groups["c1"]),
            "expected_n_feat": int(cfg["data"]["expected_c1_n_feat"]),
        },
    )
    if len(groups["c1"]) != int(cfg["data"]["expected_c1_n_feat"]):
        raise AssertionError("C1 manifest size mismatch")

    if quick:
        df = subsample_train_for_smoke(df, n_train=int(args.smoke_n_train), seed=seed)
        tr_smoke = df[df["recommended_split"] == "train"]
        aux_tr_smoke = tr_smoke[tr_smoke["aux_eligible"].astype(bool)]
        k_folds = int(cfg["run"]["oof_folds"])
        if len(aux_tr_smoke) == 0:
            raise AssertionError("smoke: no aux train rows after subsample")
        lab_min = int(aux_tr_smoke.groupby("label").size().min())
        if lab_min < k_folds:
            raise AssertionError(
                f"smoke: aux-train min class count {lab_min} < oof_folds={k_folds}; "
                f"raise --smoke-n-train (got {len(tr_smoke)})"
            )
        log(
            f"smoke subsample train={int((df.recommended_split=='train').sum())} "
            f"val={int((df.recommended_split=='val').sum())} "
            f"test={int((df.recommended_split=='test').sum())} "
            f"aux_train_min_class={lab_min}"
        )

    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={device}")

    # --- stage 1 ---
    preds = run_stage1(df, groups, cfg, device=device, log=log)
    pred_cols = pred_col_names(cfg, groups["glu_targets"])
    write_json(
        art / "stage1_metrics.json",
        {
            "val": preds.stage1_val_metrics,
            "test": preds.stage1_test_metrics,
            "fold_label_counts": preds.fold_label_counts,
            "best_params_per_target": preds.best_params_per_target,
            "pred_cols": pred_cols,
        },
    )
    # persist Ŷ
    yhat_dir = art / "yhat"
    yhat_dir.mkdir(exist_ok=True)
    preds.yhat_train.to_parquet(yhat_dir / "yhat_train.parquet", index=False)
    preds.yhat_val.to_parquet(yhat_dir / "yhat_val.parquet", index=False)
    preds.yhat_test.to_parquet(yhat_dir / "yhat_test.parquet", index=False)

    df2 = merge_yhat_into_df(df, preds, pred_cols)
    drift = {
        "val": yhat_drift_table(df2, pred_cols, split="val"),
        "test": yhat_drift_table(df2, pred_cols, split="test"),
    }
    write_json(art / "yhat_drift.json", drift)
    log("wrote stage1 metrics + Ŷ + drift")

    if not preds.stage1_val_metrics.get("gate_pass", False):
        log("WARNING: Stage-1 val R² gate FAILED (mean/sd/tar not all > 0)")
        if not quick:
            log("aborting Stage-2 on full run due to Stage-1 gate")
            (art / "run.log").write_text("\n".join(logs) + "\n")
            return 2

    if args.skip_stage2 or args.stage1_only:
        log("skip stage2")
        (art / "run.log").write_text("\n".join(logs) + "\n")
        write_json(
            art / "run_manifest.json",
            {
                "run_id": run_id,
                "stage2": False,
                "elapsed_sec": time.time() - t0,
            },
        )
        return 0

    # --- stage 2 arms ---
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        arms = list(DEFAULT_SMOKE_ARMS if quick else DEFAULT_FULL_ARMS)
    log(f"stage2 arms={arms}")

    arm_results: dict[str, dict[str, Any]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}

    for arm in arms:
        pool = arm_pool(arm)
        feats = arm_feature_cols(arm, groups, pred_cols)
        if is_oracle_arm(arm):
            assert_oracle_features(feats, groups["true_cols"])
            # oracle may include ytrue_; still ban meta / coverage
            meta_hit = [
                c
                for c in feats
                if c
                in (
                    "label",
                    "recommended_split",
                    "clinical_site",
                    "person_id",
                    "study_group",
                    "wearable_core",
                    "wearable_core_strict",
                    "aux_eligible",
                    "age_discrepancy",
                )
                or c in cfg["data"]["glu_forbid"]
            ]
            if meta_hit:
                raise AssertionError(f"{arm} forbidden cols: {meta_hit}")
        else:
            # deployable: no true CGM; full deny incl. coverage counts
            if any(c.startswith("ytrue_") for c in feats):
                raise AssertionError(f"{arm} has ytrue_ in deployable features")
            assert_no_leakage(feats, cfg)

        log(f"=== arm {arm} pool={pool} n_feat={len(feats)} ===")
        splits = split_xy(df2, feats, pool=pool)
        log(
            f"  n_train={splits['n_train']} n_val={splits['n_val']} "
            f"n_test={splits['n_test']}"
        )
        result = train_stage2(splits, cfg, log=log)
        result["arm"] = arm
        result["deployable"] = is_deployable_arm(arm)
        result["oracle"] = is_oracle_arm(arm)
        arm_results[arm] = result
        arm_summaries[arm] = summarize_arm(result)

        arm_dir = art / "arms" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_json(arm_dir / "summary.json", arm_summaries[arm])
        write_json(arm_dir / "metrics.json", result["metrics"])
        write_json(
            arm_dir / "selected.json",
            {
                "arm": arm,
                "family": result["family"],
                "params": result["params"],
                "best_iteration": result.get("best_iteration"),
                "feature_cols": result["feature_cols"],
                "feature_hash": _feature_hash(result["feature_cols"]),
                "pool": pool,
            },
        )
        joblib.dump(result["model"], arm_dir / "model.joblib")
        np.save(arm_dir / "proba_test.npy", result["proba_test"])
        np.save(arm_dir / "pid_test.npy", result["pid_test"])
        np.save(arm_dir / "y_test.npy", result["y_test"])
        log(
            f"  TEST auc={arm_summaries[arm]['test_macro_ovr_auc']:.4f} "
            f"bin={arm_summaries[arm]['test_binary_auc']:.4f} "
            f"auprc={arm_summaries[arm]['test_macro_auprc']:.4f}"
        )

    # --- comparisons ---
    n_boot = int(cfg["run"]["bootstrap_n"])
    if quick:
        n_boot = min(n_boot, 500)
    alpha = 1.0 - float(cfg["run"]["bootstrap_ci"])
    comparisons: dict[str, dict[str, Any]] = {}
    pairs = [
        ("T1_vs_D1", "T1", "D1"),
        ("T0_vs_D0", "T0", "D0"),
        ("O1_vs_D1a", "O1", "D1a"),
        ("O0_vs_D0a", "O0", "D0a"),
    ]
    for key, a, b in pairs:
        if a in arm_results and b in arm_results:
            log(f"bootstrap {key} n_boot={n_boot}")
            comparisons[key] = compare_arms(
                arm_results[a],
                arm_results[b],
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            )
            write_json(art / f"compare_{key}.json", comparisons[key])
            d = comparisons[key]["delta_macro_ovr_auc"]
            log(
                f"  ΔAUC point={d['point']:+.4f} "
                f"CI[{d['lo']:+.4f},{d['hi']:+.4f}] "
                f"lo>0={d['ci_lower_gt_zero']}"
            )

    bars = apply_decision_bars(
        arm_summaries, comparisons, preds.stage1_val_metrics, cfg
    )
    write_json(art / "decision_bars.json", bars)
    write_json(art / "arm_summaries.json", arm_summaries)
    log(f"decision_bars={json.dumps(bars, default=str)[:500]}...")

    manifest = {
        "run_id": run_id,
        "quick": quick,
        "arms": arms,
        "seed": seed,
        "n_trials_stage2": int(cfg["run"]["n_trials"]),
        "n_trials_stage1": int(cfg["run"]["stage1_n_trials"]),
        "git": _git_hash(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_sec": time.time() - t0,
        "frozen_c1_reference": cfg["frozen_c1_reference"],
        "stage1_gate_pass": preds.stage1_val_metrics.get("gate_pass"),
    }
    write_json(art / "run_manifest.json", manifest)
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"DONE elapsed={manifest['elapsed_sec']:.1f}s art={art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
