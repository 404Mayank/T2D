"""Path B3 CLI — logit knowledge distillation baseline."""

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
from training.path_b.b3.data import load_b3_frame, subsample_train_for_smoke
from training.path_b.b3.evaluate import apply_decision_bars, compare_arms, summarize_arm
from training.path_b.b3.student_gbm import train_d1, train_g_alpha
from training.path_b.b3.student_mlp import train_mlp_student
from training.path_b.b3.teacher import build_teacher_and_oof


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


def _arm_key_g(alpha: float) -> str:
    if float(alpha) == 0.0:
        return "G0"
    return f"G_a={alpha:g}"


def _arm_key_n(alpha: float) -> str:
    if float(alpha) == 0.0:
        return "N0"
    return f"N_a={alpha:g}"


def _strip_for_summary(pack: dict[str, Any]) -> dict[str, Any]:
    s = summarize_arm(pack)
    s["alpha"] = pack.get("alpha")
    s["temperature"] = pack.get("temperature")
    s["family"] = pack.get("family")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path B3 logit-KD baseline")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "teacher", "students", "smoke"],
        help="all=teacher+students; teacher only; students need prior teacher art",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma alphas for G/N grids (default from config / smoke set)",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--quick", action="store_true", help="Smoke: few trials + train subsample")
    ap.add_argument("--smoke-n-train", type=int, default=400)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-gbm-grid", action="store_true", help="Only G0 + decision alpha")
    ap.add_argument("--device", type=str, default=None, help="torch device for MLP")
    ap.add_argument(
        "--teacher-from",
        type=Path,
        default=None,
        help="Reuse teacher artifacts dir (soft labels + Tch/D1a)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    quick = bool(args.quick or args.mode == "smoke")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "b3_%Y%m%d_%H%M%S" + ("_smoke" if quick else "")
    )
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    if art.exists() and (
        (art / "decision_bars.json").exists() or (art / "teacher_metrics.json").exists()
    ):
        raise RuntimeError(f"refuse overwrite existing run dir {art}")
    art.mkdir(parents=True, exist_ok=True)

    if quick:
        cfg["run"]["n_trials"] = 5
        cfg["run"]["mlp_epochs"] = 15
        cfg["run"]["mlp_patience"] = 5
    if args.n_trials is not None:
        cfg["run"]["n_trials"] = int(args.n_trials)
    if args.temperature is not None:
        cfg["run"]["temperature"] = float(args.temperature)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    t0 = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"run_id={run_id} quick={quick} mode={args.mode}")
    log(f"repo={repo}")

    # --- data ---
    df, groups = load_b3_frame(repo, cfg)
    log(
        f"loaded core={len(df)} aux={int(df['aux_eligible'].sum())} "
        f"c1={len(groups['c1'])}"
    )
    write_json(
        art / "c1_feature_manifest.json",
        {
            "n_feat": len(groups["c1"]),
            "feature_cols": groups["c1"],
            "feature_hash": _feature_hash(groups["c1"]),
            "expected_n_feat": int(cfg["data"]["expected_c1_n_feat"]),
        },
    )
    if len(groups["c1"]) != int(cfg["data"]["expected_c1_n_feat"]):
        raise AssertionError("C1 manifest size mismatch")

    if quick:
        df = subsample_train_for_smoke(df, n_train=int(args.smoke_n_train), seed=seed)
        tr_smoke = df[df["recommended_split"] == "train"]
        aux_tr = tr_smoke[tr_smoke["aux_eligible"].astype(bool)]
        k_folds = int(cfg["run"]["oof_folds"])
        lab_min = int(aux_tr.groupby("label").size().min()) if len(aux_tr) else 0
        if lab_min < k_folds:
            raise AssertionError(
                f"smoke: aux-train min class {lab_min} < oof_folds={k_folds}; "
                f"raise --smoke-n-train (got train={len(tr_smoke)})"
            )
        log(
            f"smoke train={int((df.recommended_split=='train').sum())} "
            f"aux_train_min_class={lab_min}"
        )

    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={device}")

    # --- teacher ---
    if args.teacher_from is not None:
        tdir = Path(args.teacher_from)
        log(f"loading teacher from {tdir}")
        import pandas as pd

        soft_df = pd.read_parquet(tdir / "oof_soft_labels.parquet")
        soft_by_pid = {
            int(row["person_id"]): row[[f"p{k}" for k in range(4)]].to_numpy(
                dtype=np.float64
            )
            for _, row in soft_df.iterrows()
        }
        teacher_meta = json.loads((tdir / "teacher_metrics.json").read_text())
        # rebuild minimal teacher_pack for bars; full Tch/D1a models optional
        teacher_pack = {
            "soft_by_pid": soft_by_pid,
            "oof": teacher_meta["oof"],
            "soft_diagnostics": teacher_meta.get("soft_diagnostics", {}),
            "tch": None,
            "d1a": None,
            "c1_feats": groups["c1"],
        }
        # load proba for Tch/D1a compare if present
        for arm in ("Tch", "D1a"):
            apath = tdir / "arms" / arm
            if not (apath / "proba_test.npy").exists():
                continue
            metrics = json.loads((apath / "metrics.json").read_text())
            if "test_raw" not in metrics:
                metrics = {
                    "test_raw": metrics,
                    "test_cal": metrics,
                    "val_raw": metrics,
                }
            summary = {}
            if (apath / "summary.json").exists():
                summary = json.loads((apath / "summary.json").read_text())
            meta_arm = teacher_meta.get(arm, {})
            pack = {
                "arm": arm,
                "proba_test": np.load(apath / "proba_test.npy"),
                "pid_test": np.load(apath / "pid_test.npy"),
                "y_test": np.load(apath / "y_test.npy"),
                "metrics": metrics,
                "val_macro_ovr_auc": float(
                    meta_arm.get("val_macro_ovr_auc")
                    or summary.get("val_macro_ovr_auc")
                    or metrics["test_raw"].get("macro_ovr_auc")
                    or 0.0
                ),
                "val_macro_auprc": float(
                    summary.get("val_macro_auprc")
                    or metrics.get("val_raw", {}).get("macro_auprc")
                    or metrics["test_raw"].get("macro_auprc")
                    or 0.0
                ),
                "family": meta_arm.get("family") or summary.get("family"),
                "pool": "aux",
                "n_train": meta_arm.get("n_train") or summary.get("n_train"),
                "n_val": meta_arm.get("n_val") or summary.get("n_val"),
                "n_test": meta_arm.get("n_test") or summary.get("n_test"),
                "feature_cols": groups["c1"] + groups["true_cols"]
                if arm == "Tch"
                else groups["c1"],
                "deployable": arm != "Tch",
                "oracle": arm == "Tch",
                "proba_test_cal": np.load(apath / "proba_test.npy"),
                "calibrator": None,
            }
            if arm == "Tch":
                teacher_pack["tch"] = pack
            else:
                teacher_pack["d1a"] = pack
        log(f"loaded {len(soft_by_pid)} OOF soft labels")
    else:
        teacher_pack = build_teacher_and_oof(
            df, groups, cfg, n_trials=int(cfg["run"]["n_trials"]), log=log
        )
        soft_by_pid = teacher_pack["soft_by_pid"]

        # persist soft labels
        rows = []
        for pid, p in soft_by_pid.items():
            rows.append({"person_id": pid, **{f"p{k}": float(p[k]) for k in range(4)}})
        import pandas as pd

        pd.DataFrame(rows).to_parquet(art / "oof_soft_labels.parquet", index=False)

        tch = teacher_pack["tch"]
        d1a = teacher_pack["d1a"]
        for arm_name, pack in (("Tch", tch), ("D1a", d1a)):
            ad = art / "arms" / arm_name
            ad.mkdir(parents=True, exist_ok=True)
            write_json(ad / "summary.json", _strip_for_summary(pack))
            write_json(ad / "metrics.json", pack["metrics"])
            write_json(
                ad / "selected.json",
                {
                    "arm": arm_name,
                    "family": pack["family"],
                    "params": pack["params"],
                    "feature_cols": pack["feature_cols"],
                },
            )
            joblib.dump(pack["model"], ad / "model.joblib")
            np.save(ad / "proba_test.npy", pack["proba_test"])
            np.save(ad / "pid_test.npy", pack["pid_test"])
            np.save(ad / "y_test.npy", pack["y_test"])
            log(
                f"{arm_name} TEST auc={pack['metrics']['test_raw']['macro_ovr_auc']:.4f} "
                f"bin={pack['metrics']['test_raw']['binary_auc']:.4f}"
            )

        write_json(
            art / "teacher_metrics.json",
            {
                "oof": teacher_pack["oof"],
                "soft_diagnostics": teacher_pack["soft_diagnostics"],
                "Tch": {
                    "val_macro_ovr_auc": tch["val_macro_ovr_auc"],
                    "test_macro_ovr_auc": tch["metrics"]["test_raw"]["macro_ovr_auc"],
                    "family": tch["family"],
                    "n_train": tch["n_train"],
                    "n_val": tch["n_val"],
                    "n_test": tch["n_test"],
                    "teacher_val_brier": tch["metrics"].get("teacher_val_brier"),
                    "teacher_val_ece": tch["metrics"].get("teacher_val_ece"),
                },
                "D1a": {
                    "val_macro_ovr_auc": d1a["val_macro_ovr_auc"],
                    "test_macro_ovr_auc": d1a["metrics"]["test_raw"]["macro_ovr_auc"],
                    "family": d1a["family"],
                    "n_train": d1a["n_train"],
                    "n_val": d1a["n_val"],
                    "n_test": d1a["n_test"],
                },
            },
        )

    if args.mode == "teacher":
        (art / "run.log").write_text("\n".join(logs) + "\n")
        write_json(
            art / "run_manifest.json",
            {
                "run_id": run_id,
                "mode": "teacher",
                "elapsed_sec": time.time() - t0,
                "oof_gate_pass": teacher_pack["oof"]["gate_pass"],
            },
        )
        log(f"teacher-only DONE art={art}")
        return 0

    # --- alphas ---
    if args.alphas:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
    elif quick:
        alphas = [0.0, float(cfg["run"]["decision_alpha"])]
    else:
        alphas = [float(a) for a in cfg["run"]["alphas"]]
    decision_alpha = float(cfg["run"]["decision_alpha"])
    temp = float(cfg["run"]["temperature"])
    log(f"alphas={alphas} T={temp} decision_alpha={decision_alpha}")

    arm_results: dict[str, dict[str, Any]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}

    # attach teacher arms if present
    if teacher_pack.get("tch") is not None:
        arm_results["Tch"] = teacher_pack["tch"]
        arm_summaries["Tch"] = _strip_for_summary(teacher_pack["tch"])
    if teacher_pack.get("d1a") is not None:
        arm_results["D1a"] = teacher_pack["d1a"]
        arm_summaries["D1a"] = _strip_for_summary(teacher_pack["d1a"])

    # --- D1 ---
    log("=== D1 (matched C1 hard) ===")
    d1 = train_d1(df, groups["c1"], cfg, n_trials=int(cfg["run"]["n_trials"]), log=log)
    arm_results["D1"] = d1
    arm_summaries["D1"] = _strip_for_summary(d1)
    _save_arm(art, d1, log)

    # LGBM params for G0 pin: always from D1's LGBM pack (even if Cat won)
    lgbm_params = d1["lgbm_params"]
    write_json(art / "d1_lgbm_params.json", lgbm_params)

    # freeze drift
    freeze_auc = float(cfg["frozen_c1_reference"]["test_macro_ovr_auc"])
    d1_test = float(d1["metrics"]["test_raw"]["macro_ovr_auc"])
    log(f"D1 test auc={d1_test:.4f} freeze={freeze_auc:.4f} Δ={d1_test-freeze_auc:+.4f}")

    # --- G grid ---
    g_alphas = alphas
    if args.skip_gbm_grid:
        g_alphas = [0.0, decision_alpha]
    for a in g_alphas:
        key = _arm_key_g(a)
        log(f"=== {key} ===")
        pinned = lgbm_params if float(a) == 0.0 else None
        pack = train_g_alpha(
            df,
            groups["c1"],
            soft_by_pid,
            cfg,
            alpha=float(a),
            temperature=temp,
            pinned_params=pinned,
            n_trials=None if float(a) == 0.0 else int(cfg["run"]["n_trials"]),
            log=log,
            arm_name=key,
        )
        arm_results[key] = pack
        arm_summaries[key] = _strip_for_summary(pack)
        _save_arm(art, pack, log)

    # G0 protocol
    g0_protocol = None
    if "G0" in arm_results and "D1" in arm_results:
        # compare G0 to D1's LightGBM-only if available; else D1 selected
        g0_auc = float(arm_results["G0"]["metrics"]["test_raw"]["macro_ovr_auc"])
        # D1 LGBM test proba for fair pin check
        from training.path_a_watch.models import predict_proba
        from training.path_b.b3.data import split_xy

        splits = split_xy(df, groups["c1"], pool="core")
        d1_lgbm_model = d1["lgbm_pack"]["model"]
        d1_lgbm_test = predict_proba(d1_lgbm_model, splits["X_test"])
        d1_lgbm_auc = float(
            __import__(
                "training.path_a_watch.metrics", fromlist=["macro_ovr_auc"]
            ).macro_ovr_auc(splits["y_test"], d1_lgbm_test)
        )
        delta = abs(g0_auc - d1_lgbm_auc)
        tol = float(cfg["decision_bars"]["g0_auc_tol"])
        g0_protocol = {
            "g0_test_auc": g0_auc,
            "d1_lgbm_test_auc": d1_lgbm_auc,
            "d1_selected_test_auc": d1_test,
            "abs_delta_vs_d1_lgbm": delta,
            "tol": tol,
            "pass": bool(delta <= tol),
            "expansion": arm_results["G0"].get("expansion_diag"),
        }
        log(
            f"G0 protocol: |G0-D1_LGBM|={delta:.6f} tol={tol} pass={g0_protocol['pass']}"
        )
        if not g0_protocol["pass"] and not quick:
            log("ERROR: G0 protocol failed — plumbing bug; abort before Gα claims")
            write_json(art / "g0_protocol.json", g0_protocol)
            (art / "run.log").write_text("\n".join(logs) + "\n")
            return 3
        write_json(art / "g0_protocol.json", g0_protocol)

    # --- N grid ---
    if not args.skip_mlp:
        n_alphas = alphas
        if args.skip_gbm_grid:
            n_alphas = [0.0, decision_alpha]
        for a in n_alphas:
            key = _arm_key_n(a)
            log(f"=== {key} ===")
            pack = train_mlp_student(
                df,
                groups["c1"],
                soft_by_pid,
                cfg,
                alpha=float(a),
                temperature=temp,
                device=args.device,
                log=log,
                arm_name=key,
            )
            arm_results[key] = pack
            arm_summaries[key] = _strip_for_summary(pack)
            _save_arm(art, pack, log, save_model=False)

    # --- comparisons ---
    n_boot = int(cfg["run"]["bootstrap_n"])
    if quick:
        n_boot = min(n_boot, 500)
    alpha_ci = 1.0 - float(cfg["run"]["bootstrap_ci"])
    comparisons: dict[str, dict[str, Any]] = {}

    def _pair(key: str, a: str, b: str) -> None:
        if a in arm_results and b in arm_results:
            log(f"bootstrap {key} n_boot={n_boot}")
            comparisons[key] = compare_arms(
                arm_results[a],
                arm_results[b],
                n_boot=n_boot,
                seed=seed,
                alpha=alpha_ci,
            )
            write_json(art / f"compare_{key}.json", comparisons[key])
            d = comparisons[key]["delta_macro_ovr_auc"]
            log(
                f"  ΔAUC point={d['point']:+.4f} "
                f"CI[{d['lo']:+.4f},{d['hi']:+.4f}] "
                f"lo>0={d['ci_lower_gt_zero']}"
            )

    g_dec = _arm_key_g(decision_alpha)
    n_dec = _arm_key_n(decision_alpha)
    _pair("Tch_vs_D1a", "Tch", "D1a")
    _pair("G_a0.3_vs_D1", g_dec, "D1")
    _pair("N_a0.3_vs_N0", n_dec, "N0")

    bars = apply_decision_bars(
        arm_summaries,
        comparisons,
        teacher_pack,
        cfg,
        g0_protocol=g0_protocol,
    )
    write_json(art / "decision_bars.json", bars)
    write_json(art / "arm_summaries.json", arm_summaries)
    log(f"decision_bars keys={list(bars.keys())}")

    # T-sensitivity trigger (report only; do not auto-run unless flag later)
    t_trigger = False
    gcmp = comparisons.get("G_a0.3_vs_D1")
    tch_cmp = comparisons.get("Tch_vs_D1a")
    if gcmp and tch_cmp:
        d = gcmp["delta_macro_ovr_auc"]
        pt = float(d["point"])
        lo = d.get("ci_lower_gt_zero")
        head = float(tch_cmp["delta_macro_ovr_auc"]["point"])
        if (pt > 0 and not lo) or (head > 0.04 and not lo and pt <= 0):
            t_trigger = True
    write_json(
        art / "t_sensitivity_trigger.json",
        {"trigger": t_trigger, "note": "run T in {1,4} at α=0.3 only if true"},
    )
    log(f"T-sensitivity trigger={t_trigger}")

    manifest = {
        "run_id": run_id,
        "quick": quick,
        "mode": args.mode,
        "seed": seed,
        "n_trials": int(cfg["run"]["n_trials"]),
        "temperature": temp,
        "alphas": alphas,
        "decision_alpha": decision_alpha,
        "git": _git_hash(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_sec": time.time() - t0,
        "frozen_c1_reference": cfg["frozen_c1_reference"],
        "oof_gate_pass": teacher_pack.get("oof", {}).get("gate_pass"),
        "g0_protocol_pass": None if g0_protocol is None else g0_protocol.get("pass"),
        "t_sensitivity_trigger": t_trigger,
    }
    write_json(art / "run_manifest.json", manifest)
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"DONE elapsed={manifest['elapsed_sec']:.1f}s art={art}")
    return 0


def _save_arm(
    art: Path,
    pack: dict[str, Any],
    log,
    *,
    save_model: bool = True,
) -> None:
    arm = pack["arm"]
    ad = art / "arms" / arm
    ad.mkdir(parents=True, exist_ok=True)
    write_json(ad / "summary.json", _strip_for_summary(pack))
    write_json(ad / "metrics.json", pack["metrics"])
    write_json(
        ad / "selected.json",
        {
            "arm": arm,
            "family": pack.get("family"),
            "params": pack.get("params"),
            "alpha": pack.get("alpha"),
            "temperature": pack.get("temperature"),
            "feature_cols": pack.get("feature_cols"),
            "pinned": pack.get("pinned"),
        },
    )
    if save_model and pack.get("model") is not None:
        try:
            joblib.dump(pack["model"], ad / "model.joblib")
        except Exception as e:
            log(f"  warn: could not joblib model for {arm}: {e}")
    np.save(ad / "proba_test.npy", pack["proba_test"])
    np.save(ad / "pid_test.npy", pack["pid_test"])
    np.save(ad / "y_test.npy", pack["y_test"])
    log(
        f"  TEST auc={pack['metrics']['test_raw']['macro_ovr_auc']:.4f} "
        f"bin={pack['metrics']['test_raw']['binary_auc']:.4f} "
        f"auprc={pack['metrics']['test_raw']['macro_auprc']:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
