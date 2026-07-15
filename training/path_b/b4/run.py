"""CLI: B4-A λ grid + hybrid C1 ambition arms + matched D1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.path_b.b1.metrics import paired_bootstrap_delta_auc
from training.path_b.b4.data import build_grid_bundle, load_config, subset_counts
from training.path_b.b4.hybrid import (
    hybrid_from_bundle_embeddings,
    pair_boot_from_preds,
    run_d1,
)
from training.path_b.b4.distill import run_distill_pipeline
from training.path_b.b4.train import train_one_lambda


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            print("cuda requested but unavailable → cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _lam_dir(lam: float) -> str:
    return f"lambda_{f'{lam:g}'.replace('.', 'p')}"


def main(argv: list[str] | None = None) -> int:
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="B4 trajectory multi-task + hybrid C1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lambdas", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-participants", type=int, default=None)
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "neural", "hybrid", "d1", "smoke", "distill", "distill_hybrid"],
        help="all=neural+d1+hybrid; distill=B4-B teacher+students; distill_hybrid=distill then z∥C1+D1",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage2-trials", type=int, default=None)
    ap.add_argument("--mus", default=None, help="B4-B μ list, e.g. 0,0.3,1.0")
    ap.add_argument(
        "--teacher-mode",
        default="easy",
        choices=["easy", "cgm_only", "wear_cgm"],
        help="B4-B teacher: easy=X∥cgm; cgm_only=H1; wear_cgm=H2 wear→cgm",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if args.config else Path(__file__).parent / "config.yaml"
    cfg = load_config(cfg_path)
    device = _device(args.device)
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")

    quick = bool(args.quick or args.mode == "smoke")
    if quick:
        max_part = args.max_participants or int(cfg["quick"]["max_participants"])
        lams = [
            float(x)
            for x in (
                args.lambdas.split(",")
                if args.lambdas
                else cfg["quick"]["lambdas"]
            )
        ]
        mus = [
            float(x)
            for x in (
                args.mus.split(",")
                if args.mus
                else cfg["quick"].get("mus", [0.0, 1.0])
            )
        ]
        s2_trials = args.stage2_trials or int(cfg["quick"].get("stage2_n_trials", 5))
    else:
        max_part = args.max_participants
        lams = [
            float(x)
            for x in (args.lambdas.split(",") if args.lambdas else cfg["lambdas"])
        ]
        mus = [
            float(x)
            for x in (
                args.mus.split(",")
                if args.mus
                else (cfg.get("distill") or {}).get("mus", [0.0, 0.3, 1.0])
            )
        ]
        s2_trials = args.stage2_trials or int(cfg["stage2"]["n_trials"])

    art_root = Path(cfg["paths"]["artifacts"])
    if not art_root.is_absolute():
        art_root = _REPO / art_root
    run_dir = art_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(
            {**cfg, "cli": vars(args), "lambdas_run": lams, "mus_run": mus}, f
        )

    t0 = time.time()
    mode = args.mode
    if mode == "smoke":
        mode = "neural"

    do_neural = mode in ("all", "neural")
    do_distill = mode in ("distill", "distill_hybrid")
    do_d1 = mode in ("all", "d1", "hybrid", "distill_hybrid")
    do_hybrid = mode in ("all", "hybrid")
    do_distill_hybrid = mode == "distill_hybrid"

    bundle = None
    if do_neural or do_hybrid or do_distill or do_distill_hybrid:
        print("=== build grid bundle ===")
        bundle = build_grid_bundle(
            _REPO, cfg, max_participants=max_part, load_c1=True
        )
        diag = subset_counts(bundle)
        print(json.dumps(diag, indent=2))
        with open(run_dir / "data_diag.json", "w") as f:
            json.dump(
                {
                    "splits": diag,
                    "n_features": len(bundle.feature_cols),
                    "feature_cols": bundle.feature_cols,
                    "class_weights": bundle.class_weights.tolist(),
                    "feat_mean": bundle.feat_mean.tolist(),
                    "feat_std": bundle.feat_std.tolist(),
                    "cgm_mean": bundle.cgm_mean,
                    "cgm_std": bundle.cgm_std,
                    "t_bins": int(cfg["data"]["t_bins"]),
                    "t_min": int(cfg["data"]["t_min"]),
                    "dropped_pids": bundle.dropped_pids,
                    "subwindow": "wear_density_cgm_free",
                },
                f,
                indent=2,
            )

    # --- neural λ grid ---
    lam_metrics: dict[str, Any] = {}
    if do_neural:
        assert bundle is not None
        for lam in lams:
            ldir = run_dir / _lam_dir(lam)
            if args.resume and (ldir / "metrics.json").exists():
                print(f"=== skip λ={lam} (resume) ===")
                with open(ldir / "metrics.json") as f:
                    lam_metrics[str(lam)] = json.load(f)
                continue
            print(f"=== train λ={lam} ===")
            met = train_one_lambda(
                bundle, cfg, lam=lam, out_dir=ldir, device=device, quick=quick
            )
            lam_metrics[str(lam)] = {
                k: v
                for k, v in met.items()
                if not k.endswith(("_proba", "_y", "_pid"))
            }

        # multi-task boot Sλ − S0 on test
        if 0.0 in lams or "0" in [str(x) for x in lams]:
            base = run_dir / _lam_dir(0.0) / "test_preds.npz"
            if base.exists():
                boots = {}
                a = np.load(base)
                for lam in lams:
                    if float(lam) == 0.0:
                        continue
                    other = run_dir / _lam_dir(lam) / "test_preds.npz"
                    if not other.exists():
                        continue
                    b = np.load(other)
                    # align by pid
                    da = {int(p): i for i, p in enumerate(a["pid"])}
                    idx_a, idx_b = [], []
                    for i, p in enumerate(b["pid"]):
                        p = int(p)
                        if p in da:
                            idx_a.append(da[p])
                            idx_b.append(i)
                    ya = a["y"][idx_a]
                    pa = a["proba"][idx_a]
                    pb = b["proba"][idx_b]
                    boots[str(lam)] = paired_bootstrap_delta_auc(
                        ya,
                        pa,
                        pb,
                        n_boot=int(cfg["bootstrap"]["n"]),
                        seed=int(cfg["bootstrap"]["seed"]),
                    )
                with open(run_dir / "compare_Sl_vs_S0.json", "w") as f:
                    json.dump(boots, f, indent=2, default=float)
                print("Sλ−S0:", json.dumps(boots, indent=2, default=float))

    # --- D1 matched baseline ---
    # Ambition pairing: same person set as sequence bundle when available
    # (T_min drops / smoke subsample). Full-core freeze parity only when
    # pid_allow is None and n matches expected_core_n.
    d1_dir = run_dir / "d1"
    if do_d1:
        print("=== D1 re-fit (Path A C1 family) ===")
        if args.resume and (d1_dir / "d1_test_preds.npz").exists():
            print("  skip D1 (resume)")
        else:
            pid_allow = bundle.pids if bundle is not None else None
            run_d1(
                _REPO,
                cfg,
                d1_dir,
                n_trials=s2_trials,
                pid_allow=pid_allow,
                log=print,
            )

    # --- hybrid S0+C1 / Sλ+C1 ---
    if do_hybrid:
        assert bundle is not None
        hybrid_dir = run_dir / "hybrid"
        hybrid_dir.mkdir(exist_ok=True)
        # pick λ=0 and best non-zero (or each)
        emb0 = run_dir / _lam_dir(0.0) / "embeddings.npz"
        if not emb0.exists():
            # find any lambda_0*
            cands = list(run_dir.glob("lambda_0*/embeddings.npz"))
            if not cands:
                raise FileNotFoundError("need λ=0 embeddings for hybrid; run neural first")
            emb0 = cands[0]
        hybrid_from_bundle_embeddings(
            _REPO,
            cfg,
            bundle,
            emb0,
            arm_name="S0_C1",
            out_dir=hybrid_dir,
            n_trials=s2_trials,
            log=print,
        )
        for lam in lams:
            if float(lam) == 0.0:
                continue
            emb = run_dir / _lam_dir(lam) / "embeddings.npz"
            if not emb.exists():
                print(f"  skip hybrid λ={lam}: no embeddings")
                continue
            hybrid_from_bundle_embeddings(
                _REPO,
                cfg,
                bundle,
                emb,
                arm_name=f"S{str(lam).replace('.', 'p')}_C1",
                out_dir=hybrid_dir,
                n_trials=s2_trials,
                log=print,
            )

        # ambition boots vs D1
        d1_preds = d1_dir / "d1_test_preds.npz"
        boots_h = {}
        if d1_preds.exists():
            for pred in hybrid_dir.glob("*_test_preds.npz"):
                arm = pred.name.replace("_test_preds.npz", "")
                try:
                    boots_h[arm] = pair_boot_from_preds(
                        d1_preds,
                        pred,
                        n_boot=int(cfg["bootstrap"]["n"]),
                        seed=int(cfg["bootstrap"]["seed"]),
                    )
                except Exception as e:
                    boots_h[arm] = {"error": str(e)}
            # Sλ+C1 vs S0+C1
            s0p = hybrid_dir / "S0_C1_test_preds.npz"
            for pred in hybrid_dir.glob("S*_C1_test_preds.npz"):
                if pred.name.startswith("S0_"):
                    continue
                arm = pred.name.replace("_test_preds.npz", "")
                if s0p.exists():
                    try:
                        boots_h[f"{arm}_vs_S0_C1"] = pair_boot_from_preds(
                            s0p,
                            pred,
                            n_boot=int(cfg["bootstrap"]["n"]),
                            seed=int(cfg["bootstrap"]["seed"]),
                        )
                    except Exception as e:
                        boots_h[f"{arm}_vs_S0_C1"] = {"error": str(e)}
            with open(run_dir / "compare_hybrid_vs_D1.json", "w") as f:
                json.dump(boots_h, f, indent=2, default=float)
            print("hybrid vs D1:", json.dumps(boots_h, indent=2, default=float))

    # --- B4-B representation distillation ---
    if do_distill:
        assert bundle is not None
        ddir = run_dir / "distill"
        ddir.mkdir(exist_ok=True)
        run_distill_pipeline(
            bundle,
            cfg,
            run_dir=ddir,
            device=device,
            mus=mus,
            quick=quick,
            resume=args.resume,
            teacher_mode=args.teacher_mode,  # type: ignore[arg-type]
        )
        def _mu_dir(mu: float) -> Path:
            return ddir / f"mu_{str(float(mu)).replace('.', 'p')}"

        # μ>0 vs μ=0 paired boot on test
        base = _mu_dir(0.0) / "test_preds.npz"
        boots_d: dict[str, Any] = {}
        if base.exists():
            a = np.load(base)
            for mu in mus:
                if float(mu) == 0.0:
                    continue
                other = _mu_dir(mu) / "test_preds.npz"
                if not other.exists():
                    continue
                b = np.load(other)
                da = {int(p): i for i, p in enumerate(a["pid"])}
                idx_a, idx_b = [], []
                for i, p in enumerate(b["pid"]):
                    p = int(p)
                    if p in da:
                        idx_a.append(da[p])
                        idx_b.append(i)
                boots_d[str(mu)] = paired_bootstrap_delta_auc(
                    a["y"][idx_a],
                    a["proba"][idx_a],
                    b["proba"][idx_b],
                    n_boot=int(cfg["bootstrap"]["n"]),
                    seed=int(cfg["bootstrap"]["seed"]),
                )
            with open(run_dir / "compare_mu_vs_mu0.json", "w") as f:
                json.dump(boots_d, f, indent=2, default=float)
            print("μ−μ0:", json.dumps(boots_d, indent=2, default=float))

        # optional hybrid on student embeddings
        if do_distill_hybrid:
            hybrid_dir = run_dir / "hybrid_distill"
            hybrid_dir.mkdir(exist_ok=True)
            d1_dir = run_dir / "d1"
            if not (d1_dir / "d1_test_preds.npz").exists():
                print("=== D1 re-fit for distill hybrid ===")
                run_d1(
                    _REPO,
                    cfg,
                    d1_dir,
                    n_trials=s2_trials,
                    pid_allow=bundle.pids,
                    log=print,
                )
            for mu in mus:
                emb = _mu_dir(mu) / "embeddings.npz"
                if not emb.exists():
                    print(f"  skip distill hybrid μ={mu}: no emb")
                    continue
                arm = f"Dmu_{str(float(mu)).replace('.', 'p')}_C1"
                hybrid_from_bundle_embeddings(
                    _REPO,
                    cfg,
                    bundle,
                    emb,
                    arm_name=arm,
                    out_dir=hybrid_dir,
                    n_trials=s2_trials,
                    log=print,
                )
            d1_preds = d1_dir / "d1_test_preds.npz"
            boots_h = {}
            if d1_preds.exists():
                for pred in hybrid_dir.glob("*_test_preds.npz"):
                    arm = pred.name.replace("_test_preds.npz", "")
                    try:
                        boots_h[arm] = pair_boot_from_preds(
                            d1_preds,
                            pred,
                            n_boot=int(cfg["bootstrap"]["n"]),
                            seed=int(cfg["bootstrap"]["seed"]),
                        )
                    except Exception as e:
                        boots_h[arm] = {"error": str(e)}
                with open(run_dir / "compare_distill_hybrid_vs_D1.json", "w") as f:
                    json.dump(boots_h, f, indent=2, default=float)
                print("distill hybrid vs D1:", json.dumps(boots_h, indent=2, default=float))

    # decision bars stub
    bars: dict[str, Any] = {
        "run_id": args.run_id,
        "elapsed_s": round(time.time() - t0, 2),
        "mode": args.mode,
        "quick": quick,
    }
    for name in (
        "compare_Sl_vs_S0.json",
        "compare_hybrid_vs_D1.json",
        "compare_mu_vs_mu0.json",
        "compare_distill_hybrid_vs_D1.json",
    ):
        p = run_dir / name
        if p.exists():
            bars[name.replace("compare_", "").replace(".json", "")] = json.loads(
                p.read_text()
            )
    d1m = run_dir / "d1" / "d1_metrics.json"
    if d1m.exists():
        bars["d1"] = json.loads(d1m.read_text())
    dsum = run_dir / "distill" / "distill_summary.json"
    if dsum.exists():
        bars["distill_summary"] = json.loads(dsum.read_text())
    with open(run_dir / "decision_bars.json", "w") as f:
        json.dump(bars, f, indent=2, default=float)

    print(f"Done in {bars['elapsed_s']}s → {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
