"""CLI: B1 λ grid (controlled multi-task ablation)."""

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


def main(argv: list[str] | None = None) -> int:
    # RX 5600 (gfx1010): MIOpen reduction kernels fail; use native PyTorch ops.
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="B1 multi-task ablation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lambdas", default=None, help="comma list, e.g. 0,0.3,0.5,1.0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="skip λ dirs that already have metrics.json")
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
    device = _device(args.device)
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")

    if args.quick:
        max_part = args.max_participants or int(cfg["quick"]["max_participants"])
        lams = [float(x) for x in (args.lambdas.split(",") if args.lambdas else cfg["quick"]["lambdas"])]
    else:
        max_part = args.max_participants
        lams = [float(x) for x in (args.lambdas.split(",") if args.lambdas else cfg["lambdas"])]

    art_root = Path(cfg["paths"]["artifacts"])
    if not art_root.is_absolute():
        art_root = _REPO / art_root
    run_dir = art_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump({**cfg, "cli": vars(args), "lambdas_run": lams}, f)

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

    # Require some glu days in train for multi-task to be meaningful
    if diag["train"]["n_glu_days"] == 0:
        print("WARN: train has 0 glu days — multi-task will be CE-only effectively")

    per_lam: dict[str, Any] = {}
    for lam in lams:
        ldir = run_dir / _lam_dir_name(lam)
        metrics_path = ldir / "metrics.json"
        if args.resume and metrics_path.exists():
            print(f"\n=== skip λ={lam} (resume: {metrics_path}) ===")
            metrics = json.loads(metrics_path.read_text())
            per_lam[str(lam)] = {
                "train": {"lam": float(lam), "ckpt": str(ldir / "best.pt"), "resumed": True},
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
        # If best.pt exists but metrics missing (crashed mid-eval), re-eval only
        ckpt = ldir / "best.pt"
        if args.resume and ckpt.exists() and not metrics_path.exists():
            print(f"\n=== re-eval only λ={lam} (found {ckpt}) ===")
            train_info = {"lam": float(lam), "ckpt": str(ckpt), "best_epoch": -1, "best_val_macro_ovr_auc": float("nan"), "n_epochs_ran": 0}
        else:
            print(f"\n=== train λ={lam} ===")
            train_info = train_one_lambda(
                bundle, cfg, lam=lam, out_dir=ldir, device=device, quick=args.quick
            )
        print(f"=== eval λ={lam} ===")
        metrics = run_eval(bundle, Path(train_info["ckpt"]), ldir, cfg, device)
        per_lam[str(lam)] = {
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
            f"  λ={lam} test 4-AUC={metrics['test_raw']['macro_ovr_auc']:.4f} "
            f"bin={metrics['test_raw']['binary_auc']:.4f}"
        )

    # Post-grid paired bootstrap vs λ=0 if present
    boot = {}
    floor = cfg.get("path_a_floor") or {}
    if "0.0" in per_lam or "0" in per_lam:
        key0 = "0.0" if "0.0" in per_lam else "0"
        preds0 = pd.read_parquet(run_dir / _lam_dir_name(float(key0)) / "test_preds.parquet")
        y = preds0["y"].to_numpy()
        p0 = preds0[["p0", "p1", "p2", "p3"]].to_numpy()
        for lam in lams:
            if float(lam) == 0.0:
                continue
            preds = pd.read_parquet(run_dir / _lam_dir_name(lam) / "test_preds.parquet")
            a = preds0.set_index("person_id").sort_index()
            b = preds.set_index("person_id").sort_index()
            common = a.index.intersection(b.index)
            a = a.loc[common]
            b = b.loc[common]
            pa = a[["p0", "p1", "p2", "p3"]].to_numpy()
            pb = b[["p0", "p1", "p2", "p3"]].to_numpy()
            ya = a["y"].to_numpy()
            boot[str(lam)] = paired_bootstrap_delta_auc(
                ya,
                pa,
                pb,
                n_boot=int(cfg["eval"]["bootstrap_n"]),
                seed=int(cfg["eval"]["bootstrap_seed"]),
            )

    summary = {
        "run_id": args.run_id,
        "quick": bool(args.quick),
        "device": str(device),
        "lambdas": lams,
        "elapsed_s": round(time.time() - t0, 2),
        "path_a_floor": floor,
        "per_lambda": per_lam,
        "bootstrap_vs_lambda0": boot,
        "splits": diag,
    }
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

    with open(run_dir / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    print("\n=== summary ===")
    for lam, info in per_lam.items():
        m = info["metrics"]
        print(
            f"  λ={lam}: val={m['val_macro_ovr_auc']:.4f} "
            f"test={m['test_macro_ovr_auc']:.4f} bin={m['test_binary_auc']:.4f}"
        )
        if lam in boot:
            b = boot[lam]
            print(
                f"    Δ vs λ0: {b['delta']:+.4f} CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] "
                f"lo>0={b['ci_lo_gt_0']}"
            )
    print(f"  Path A floor (info): {floor}")
    print(f"  wrote {run_dir / 'summary.json'} ({summary['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
