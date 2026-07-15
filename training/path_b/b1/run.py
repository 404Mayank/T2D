"""CLI: B1 λ grid (controlled multi-task) + B1-GS balance arms."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# repo root on path
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.path_b.b1.data import build_sequences, load_config, subset_bundle
from training.path_b.b1.evaluate import run_eval
from training.path_b.b1.metrics import paired_bootstrap_delta_auc
from training.path_b.b1.train import train_one_lambda


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            print("cuda requested but unavailable → cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _lam_dir_name(lam: float) -> str:
    s = f"{lam:g}".replace(".", "p")
    return f"lambda_{s}"


def _arm_dir_name(arm: str) -> str:
    return f"arm_{arm}"


# Default GS claim set (PLAN_B1_GS §4.2)
_DEFAULT_GS_ARMS = ("a0", "plain", "pcgrad", "uw")


def _parse_arms(spec: str | None) -> list[dict[str, Any]]:
    """Expand arm ids → {name, balance, lam}.

    Accepts:
      - 'default' → a0, plain, pcgrad, uw
      - comma list of a0|plain|pcgrad|uw|pcgrad_uw
      - None → empty (caller uses legacy λ grid)
    """
    if spec is None or spec.strip() == "":
        return []
    if spec.strip().lower() == "default":
        names = list(_DEFAULT_GS_ARMS)
    else:
        names = [x.strip().lower() for x in spec.split(",") if x.strip()]

    out: list[dict[str, Any]] = []
    for name in names:
        if name in ("a0", "lambda0", "0", "none"):
            out.append({"name": "a0", "balance": "none", "lam": 0.0})
        elif name in ("plain", "a_plain", "lambda0.5", "0.5"):
            out.append({"name": "plain", "balance": "none", "lam": 0.5})
        elif name in ("pcgrad", "a_pcg"):
            out.append({"name": "pcgrad", "balance": "pcgrad", "lam": 1.0})
        elif name in ("uw", "uncertainty", "a_uw"):
            out.append({"name": "uw", "balance": "uncertainty", "lam": 0.0})
        elif name in ("pcgrad_uw", "a_pcg_uw"):
            out.append({"name": "pcgrad_uw", "balance": "pcgrad_uw", "lam": 0.0})
        else:
            raise ValueError(
                f"unknown arm {name!r}; expected a0|plain|pcgrad|uw|pcgrad_uw|default"
            )
    return out


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _bootstrap_vs_ref(
    run_dir: Path,
    ref_dir: Path,
    other_dirs: dict[str, Path],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    preds0 = pd.read_parquet(ref_dir / "test_preds.parquet")
    boot: dict[str, Any] = {}
    for name, odir in other_dirs.items():
        preds = pd.read_parquet(odir / "test_preds.parquet")
        a = preds0.set_index("person_id").sort_index()
        b = preds.set_index("person_id").sort_index()
        common = a.index.intersection(b.index)
        a = a.loc[common]
        b = b.loc[common]
        pa = a[["p0", "p1", "p2", "p3"]].to_numpy()
        pb = b[["p0", "p1", "p2", "p3"]].to_numpy()
        ya = a["y"].to_numpy()
        boot[name] = paired_bootstrap_delta_auc(
            ya,
            pa,
            pb,
            n_boot=int(cfg["eval"]["bootstrap_n"]),
            seed=int(cfg["eval"]["bootstrap_seed"]),
        )
    return boot


def main(argv: list[str] | None = None) -> int:
    # RX 5600 (gfx1010): MIOpen reduction kernels fail; use native PyTorch ops.
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="B1 multi-task ablation / GS balance")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lambdas", default=None, help="comma list, e.g. 0,0.3,0.5,1.0")
    ap.add_argument(
        "--balance",
        default=None,
        help="single balance mode: none|pcgrad|uncertainty|pcgrad_uw (with --lambdas)",
    )
    ap.add_argument(
        "--arms",
        default=None,
        help="GS arm grid: default | a0,plain,pcgrad,uw[,pcgrad_uw]. "
        "Overrides --lambdas when set.",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument(
        "--resume", action="store_true", help="skip arm/λ dirs that already have metrics.json"
    )
    ap.add_argument(
        "--green-fusion",
        action="store_true",
        help="late-fuse person watch_green into class head (overrides config)",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    cfg = load_config(cfg_path)
    if args.green_fusion:
        cfg.setdefault("green_fusion", {})
        cfg["green_fusion"]["enabled"] = True
    # optional UW knobs defaults
    cfg.setdefault("train", {})
    cfg["train"].setdefault("uw_lr_scale", 0.1)
    cfg["train"].setdefault("uw_clamp", 5.0)

    device = _device(args.device)
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")

    arms = _parse_arms(args.arms)
    use_gs = len(arms) > 0

    if args.quick:
        max_part = args.max_participants or int(cfg["quick"]["max_participants"])
        if not use_gs:
            lams = [
                float(x)
                for x in (
                    args.lambdas.split(",")
                    if args.lambdas
                    else cfg["quick"]["lambdas"]
                )
            ]
        else:
            lams = []
    else:
        max_part = args.max_participants
        if not use_gs:
            lams = [
                float(x)
                for x in (args.lambdas.split(",") if args.lambdas else cfg["lambdas"])
            ]
        else:
            lams = []

    single_balance = (args.balance or "none").lower()

    art_root = Path(cfg["paths"]["artifacts"])
    if not art_root.is_absolute():
        art_root = _REPO / art_root
    run_dir = art_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(
            {
                **cfg,
                "cli": vars(args),
                "lambdas_run": lams,
                "arms_run": arms,
                "single_balance": single_balance if not use_gs else None,
            },
            f,
        )

    t0 = time.time()
    print("=== build sequences ===")
    bundle = build_sequences(_REPO, cfg, max_participants=max_part)
    diag = {s: subset_bundle(bundle, s) for s in ("train", "val", "test")}
    print("  splits:", json.dumps(diag, indent=2))
    with open(run_dir / "data_diag.json", "w") as f:
        json.dump(
            {
                "splits": diag,
                "n_features": len(bundle.feature_cols),
                "feature_cols": bundle.feature_cols,
                "class_weights": bundle.class_weights.tolist(),
                "feat_mean": bundle.feat_mean.tolist(),
                "feat_std": bundle.feat_std.tolist(),
                "glu_mean": bundle.glu_mean.tolist(),
                "glu_std": bundle.glu_std.tolist(),
                "impute_values": bundle.impute_values,
                "input_zscore": "train_only_watch_valid",
                "green_fusion": bool(bundle.green is not None),
                "green_cols": bundle.green_cols,
                "green_dim": (
                    int(bundle.green.shape[1]) if bundle.green is not None else 0
                ),
            },
            f,
            indent=2,
        )

    if diag["train"]["n_glu_days"] == 0:
        print("WARN: train has 0 glu days — multi-task will be CE-only effectively")

    # Build job list: either GS arms or legacy λ list
    if use_gs:
        jobs = [
            {
                "key": a["name"],
                "dir": run_dir / _arm_dir_name(a["name"]),
                "lam": float(a["lam"]),
                "balance": a["balance"],
            }
            for a in arms
        ]
    else:
        jobs = [
            {
                "key": str(lam),
                "dir": run_dir / _lam_dir_name(lam),
                "lam": float(lam),
                "balance": single_balance,
            }
            for lam in lams
        ]

    per_job: dict[str, Any] = {}
    for job in jobs:
        jdir: Path = job["dir"]
        key = job["key"]
        lam = job["lam"]
        bal = job["balance"]
        metrics_path = jdir / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"\n=== skip {key} (resume: {metrics_path}) ===")
            metrics = json.loads(metrics_path.read_text())
            per_job[key] = {
                "train": {
                    "lam": lam,
                    "balance": bal,
                    "ckpt": str(jdir / "best.pt"),
                    "resumed": True,
                },
                "metrics": {
                    "val_macro_ovr_auc": metrics["val"]["macro_ovr_auc"],
                    "test_macro_ovr_auc": metrics["test_raw"]["macro_ovr_auc"],
                    "test_binary_auc": metrics["test_raw"]["binary_auc"],
                    "test_cal_macro_ovr_auc": metrics["test_cal"]["macro_ovr_auc"],
                    "test_glu": metrics["test_raw"].get("glu"),
                    "best_epoch": metrics.get("best_epoch"),
                },
            }
            continue
        ckpt = jdir / "best.pt"
        if args.resume and ckpt.exists() and not metrics_path.exists():
            print(f"\n=== re-eval only {key} (found {ckpt}) ===")
            train_info = {
                "lam": lam,
                "balance": bal,
                "ckpt": str(ckpt),
                "best_epoch": -1,
                "best_val_macro_ovr_auc": float("nan"),
                "n_epochs_ran": 0,
            }
        else:
            print(f"\n=== train key={key} balance={bal} λ={lam} ===")
            train_info = train_one_lambda(
                bundle,
                cfg,
                lam=lam,
                out_dir=jdir,
                device=device,
                quick=args.quick,
                balance=bal,
            )
        print(f"=== eval key={key} ===")
        metrics = run_eval(bundle, Path(train_info["ckpt"]), jdir, cfg, device)
        per_job[key] = {
            "train": train_info,
            "metrics": {
                "val_macro_ovr_auc": metrics["val"]["macro_ovr_auc"],
                "test_macro_ovr_auc": metrics["test_raw"]["macro_ovr_auc"],
                "test_binary_auc": metrics["test_raw"]["binary_auc"],
                "test_cal_macro_ovr_auc": metrics["test_cal"]["macro_ovr_auc"],
                "test_glu": metrics["test_raw"].get("glu"),
                "best_epoch": metrics["best_epoch"],
            },
        }
        print(
            f"  {key} test 4-AUC={metrics['test_raw']['macro_ovr_auc']:.4f} "
            f"bin={metrics['test_raw']['binary_auc']:.4f}"
        )

    # Paired bootstrap vs A0 / λ=0
    boot: dict[str, Any] = {}
    floor = cfg.get("path_a_floor") or {}
    if use_gs and "a0" in per_job:
        ref = run_dir / _arm_dir_name("a0")
        others = {
            k: run_dir / _arm_dir_name(k)
            for k in per_job
            if k != "a0" and (run_dir / _arm_dir_name(k) / "test_preds.parquet").exists()
        }
        if others:
            boot = _bootstrap_vs_ref(run_dir, ref, others, cfg)
    elif not use_gs:
        key0 = None
        for k in ("0.0", "0"):
            if k in per_job:
                key0 = k
                break
        if key0 is not None:
            ref = run_dir / _lam_dir_name(float(key0))
            others = {}
            for job in jobs:
                if float(job["lam"]) == 0.0:
                    continue
                p = job["dir"] / "test_preds.parquet"
                if p.exists():
                    others[job["key"]] = job["dir"]
            if others:
                boot = _bootstrap_vs_ref(run_dir, ref, others, cfg)

    summary = {
        "run_id": args.run_id,
        "quick": bool(args.quick),
        "device": str(device),
        "mode": "gs_arms" if use_gs else "lambda_grid",
        "lambdas": lams,
        "arms": arms,
        "elapsed_s": round(time.time() - t0, 2),
        "path_a_floor": floor,
        "per_arm" if use_gs else "per_lambda": per_job,
        "bootstrap_vs_a0" if use_gs else "bootstrap_vs_lambda0": boot,
        "splits": diag,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print("\n=== summary ===")
    for key, info in per_job.items():
        m = info["metrics"]
        print(
            f"  {key}: val={m['val_macro_ovr_auc']:.4f} "
            f"test={m['test_macro_ovr_auc']:.4f} bin={m['test_binary_auc']:.4f}"
        )
        if key in boot:
            b = boot[key]
            print(
                f"    Δ vs ref: {b['delta']:+.4f} CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] "
                f"lo>0={b['ci_lo_gt_0']}"
            )
    print(f"  Path A floor (info): {floor}")
    print(f"  wrote {run_dir / 'summary.json'} ({summary['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
