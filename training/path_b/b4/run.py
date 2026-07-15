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
    build_oof_embeddings,
    hybrid_from_bundle_embeddings,
    pair_boot_from_preds,
    run_d1,
    z_only_from_embeddings,
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
        choices=[
            "all",
            "neural",
            "hybrid",
            "d1",
            "smoke",
            "distill",
            "distill_hybrid",
            "mtl_bal",
            "teacher_probe",
            "hybrid_oof",
            "hybrid_v2",
        ],
        help=(
            "all/neural/hybrid/d1/smoke=v1; distill*=B4-B; "
            "mtl_bal=PCGrad/UW; teacher_probe=H2+probe; "
            "hybrid_oof=OOF-z∥C1; hybrid_v2=F0b+F2+F1"
        ),
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
    ap.add_argument(
        "--balancer",
        default=None,
        choices=["none", "pcgrad", "uncertainty"],
        help="MTL balancer (default config mtl.balancer or none)",
    )
    ap.add_argument(
        "--distill-objective",
        default=None,
        choices=["l2", "rkd", "crd"],
        help="Distill objective (default config distill.objective or l2)",
    )
    ap.add_argument("--rkd-distance-only", action="store_true", help="RKD β=0 (distance only)")
    ap.add_argument("--aug", action="store_true", help="mask-aware TS aug on student")
    ap.add_argument(
        "--run-probe",
        action="store_true",
        help="teacher linear/MLP/5-NN probe; STOP students if fail",
    )
    ap.add_argument(
        "--force-students",
        action="store_true",
        help="train students even if probe STOP (debug)",
    )
    ap.add_argument(
        "--emb",
        default=None,
        help="path to embeddings.npz for hybrid_oof / hybrid_v2",
    )
    ap.add_argument("--oof-folds", type=int, default=None)
    ap.add_argument(
        "--plumbing",
        action="store_true",
        help="Mark run as plumbing-only (prefix b4v2_plumbing_ if missing; skip science compare tables)",
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
    run_id = args.run_id
    if args.plumbing and not str(run_id).startswith("b4v2_plumbing"):
        run_id = f"b4v2_plumbing_{run_id}"
        print(f"--plumbing → run_id rewritten to {run_id}")
    run_dir = art_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(
            {**cfg, "cli": vars(args), "lambdas_run": lams, "mus_run": mus}, f
        )

    t0 = time.time()
    mode = args.mode
    if mode == "smoke":
        mode = "neural"

    balancer = args.balancer or (cfg.get("mtl") or {}).get("balancer", "none")
    # H-2: mtl_bal must not silently run plain-λ (v1 reopen)
    if mode == "mtl_bal" and balancer == "none":
        print("mtl_bal with balancer=none → defaulting to pcgrad (plan lock)")
        balancer = "pcgrad"
    distill_obj = args.distill_objective or (cfg.get("distill") or {}).get("objective", "l2")
    dist_ratio = float((cfg.get("distill") or {}).get("rkd_dist_ratio", 1.0))
    angle_ratio = float((cfg.get("distill") or {}).get("rkd_angle_ratio", 2.0))
    if args.rkd_distance_only:
        angle_ratio = 0.0

    # H-4: mtl_bal claim grid = S0 (λ=0) + one S_pc arm (λ=1 default)
    if mode == "mtl_bal" and args.lambdas is None and not quick:
        lams = [0.0, 1.0]
        print(f"mtl_bal default lambdas locked to {lams} (S0 + S_pc)")

    do_neural = mode in ("all", "neural", "mtl_bal")
    do_distill = mode in ("distill", "distill_hybrid", "teacher_probe")
    do_d1 = mode in ("all", "d1", "hybrid", "distill_hybrid", "hybrid_v2", "hybrid_oof")
    do_hybrid = mode in ("all", "hybrid")
    do_distill_hybrid = mode == "distill_hybrid"
    do_hybrid_v2 = mode in ("hybrid_v2", "hybrid_oof")
    run_probe = bool(args.run_probe or mode == "teacher_probe")

    bundle = None
    if do_neural or do_hybrid or do_distill or do_distill_hybrid or do_hybrid_v2:
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
            # mtl_bal: balancer only on λ>0 arms; S0 is always none
            if mode == "mtl_bal":
                bal = "none" if float(lam) == 0.0 else balancer
            else:
                bal = "none"
            print(f"=== train λ={lam} balancer={bal} ===")
            met = train_one_lambda(
                bundle,
                cfg,
                lam=lam,
                out_dir=ldir,
                device=device,
                quick=quick,
                balancer=bal,  # type: ignore[arg-type]
            )
            lam_metrics[str(lam)] = {
                k: v
                for k, v in met.items()
                if not k.endswith(("_proba", "_y", "_pid"))
            }

        # multi-task boot Sλ − S0 on test (skip for plumbing-only runs)
        if args.plumbing:
            print("  skip compare_Sl_vs_S0 (plumbing run)")
        elif 0.0 in lams or "0" in [str(x) for x in lams]:
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
        # teacher_probe: train teacher + probe only (no students) unless --force-students
        if mode == "teacher_probe" and not args.force_students:
            mus_run: list[float] = []
        else:
            mus_run = list(mus)
        run_distill_pipeline(
            bundle,
            cfg,
            run_dir=ddir,
            device=device,
            mus=mus_run,
            quick=quick,
            resume=args.resume,
            teacher_mode=args.teacher_mode,  # type: ignore[arg-type]
            objective=distill_obj,  # type: ignore[arg-type]
            dist_ratio=dist_ratio,
            angle_ratio=angle_ratio,
            use_aug=bool(args.aug),
            run_probe=run_probe or mode == "teacher_probe",
            force_students=bool(args.force_students),
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

    # --- V2 hybrid: F0b z-only + F2 frozen + F1 OOF ---
    if do_hybrid_v2:
        assert bundle is not None
        hybrid_dir = run_dir / "hybrid_v2"
        hybrid_dir.mkdir(exist_ok=True)
        emb_path = Path(args.emb) if args.emb else None
        if emb_path is None:
            # prefer λ=0 or mu=0 embeddings
            cands = list(run_dir.glob("lambda_0*/embeddings.npz")) + list(
                run_dir.glob("distill/mu_0*/embeddings.npz")
            )
            if not cands:
                raise FileNotFoundError(
                    "hybrid_v2 needs --emb or prior neural/distill embeddings under run_dir"
                )
            emb_path = cands[0]
        print(f"=== hybrid_v2 using emb={emb_path} ===")

        # F0b z-only
        z_only_from_embeddings(
            _REPO,
            cfg,
            bundle,
            emb_path,
            arm_name="F0b_z_only",
            out_dir=hybrid_dir,
            n_trials=s2_trials,
            log=print,
        )
        # F2 frozen z∥C1
        hybrid_from_bundle_embeddings(
            _REPO,
            cfg,
            bundle,
            emb_path,
            arm_name="F2_z_C1",
            out_dir=hybrid_dir,
            n_trials=s2_trials,
            log=print,
        )
        # F1 OOF (expensive) — skip if mode hybrid_oof only wants OOF from existing?
        if mode in ("hybrid_v2", "hybrid_oof"):
            oof_path = hybrid_dir / "oof_embeddings.npz"
            n_folds = int(args.oof_folds or (cfg.get("fusion") or {}).get("oof_folds", 5))
            if args.resume and oof_path.exists():
                print("  resume OOF embeddings")
            else:
                print(f"=== build OOF embeddings K={n_folds} ===")
                build_oof_embeddings(
                    bundle,
                    cfg,
                    device=device,
                    out_path=oof_path,
                    n_folds=n_folds,
                    quick=quick,
                    balancer="none",
                    lam=0.0,
                    log=print,
                )
            hybrid_from_bundle_embeddings(
                _REPO,
                cfg,
                bundle,
                oof_path,
                arm_name="F1_oof_z_C1_mu0",
                out_dir=hybrid_dir,
                n_trials=s2_trials,
                log=print,
            )
            print(
                "  NOTE: F1_oof_z_C1_mu0 is class-only OOF (not RKD-μ). "
                "Distill fusion ambition uses F2 until per-fold RKD OOF lands."
            )

        d1_preds = d1_dir / "d1_test_preds.npz"
        boots_h: dict[str, Any] = {}
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
            with open(run_dir / "compare_hybrid_v2_vs_D1.json", "w") as f:
                json.dump(boots_h, f, indent=2, default=float)
            print("hybrid_v2 vs D1:", json.dumps(boots_h, indent=2, default=float))

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
        "compare_hybrid_v2_vs_D1.json",
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
