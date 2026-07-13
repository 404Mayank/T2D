"""Phase 0 diagnostics on the frozen watch-only floor model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

from training.path_a_watch.data import feature_columns, load_merged, make_splits
from training.path_a_watch.evaluate import write_json
from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_a_watch.models import predict_proba


def paired_delta_bootstrap(
    y: np.ndarray,
    proba_new: np.ndarray,
    proba_floor: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Person-bootstrap CI on ΔAUC = AUC(new) - AUC(floor) on the same resample."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p_n = np.asarray(proba_new, dtype=float)
    p_f = np.asarray(proba_floor, dtype=float)
    n = len(y)
    d_auc, d_bin = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            a_n = macro_ovr_auc(yb, p_n[idx])
            a_f = macro_ovr_auc(yb, p_f[idx])
            d_auc.append(a_n - a_f)
            yb_bin = (yb > 0).astype(int)
            if yb_bin.min() == yb_bin.max():
                continue
            b_n = roc_auc_score(yb_bin, 1.0 - p_n[idx, 0])
            b_f = roc_auc_score(yb_bin, 1.0 - p_f[idx, 0])
            d_bin.append(b_n - b_f)
        except Exception:
            continue

    def _ci(arr: list[float]) -> dict[str, Any]:
        a = np.asarray(arr, dtype=float)
        lo = float(np.quantile(a, alpha / 2)) if len(a) else float("nan")
        hi = float(np.quantile(a, 1 - alpha / 2)) if len(a) else float("nan")
        return {
            "mean": float(a.mean()) if len(a) else float("nan"),
            "lo": lo,
            "hi": hi,
            "n_boot_ok": int(len(a)),
            "ci_excludes_zero": bool(len(a) and (lo > 0 or hi < 0)),
            "ci_lower_gt_zero": bool(len(a) and lo > 0),
        }

    point_d_auc = float(macro_ovr_auc(y, p_n) - macro_ovr_auc(y, p_f))
    point_d_bin = float(
        roc_auc_score((y > 0).astype(int), 1.0 - p_n[:, 0])
        - roc_auc_score((y > 0).astype(int), 1.0 - p_f[:, 0])
    )
    return {
        "delta_macro_ovr_auc": {**_ci(d_auc), "point": point_d_auc},
        "delta_binary_auc": {**_ci(d_bin), "point": point_d_bin},
        "n_boot_requested": n_boot,
        "alpha": alpha,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _repo() -> Path:
    return Path.cwd().resolve()


def pairwise_binary_auc(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """One-vs-one for selected pairs using score = p_a / (p_a+p_b) on subset."""
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    out: dict[str, float] = {}
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]
    for a, b in pairs:
        m = (y == a) | (y == b)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            out[f"{a}_vs_{b}"] = float("nan")
            continue
        score = p[m, a] / np.clip(p[m, a] + p[m, b], 1e-15, None)
        y_bin = (y[m] == a).astype(int)
        out[f"{a}_vs_{b}"] = float(roc_auc_score(y_bin, score))
    return out


def confusion_pack(y: np.ndarray, pred: np.ndarray, labels=(0, 1, 2, 3)) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=list(labels))
    return {
        "labels": list(labels),
        "matrix": cm.tolist(),
        "row_normalized": (cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)).tolist(),
    }


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def site_stratified(
    y: np.ndarray,
    proba: np.ndarray,
    sites: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site in sorted(pd.unique(sites)):
        m = sites == site
        if m.sum() < 20:
            out[str(site)] = {"n": int(m.sum()), "note": "too small"}
            continue
        rep = full_report(y[m], proba[m], tag=f"site_{site}")
        out[str(site)] = {
            "n": int(m.sum()),
            "by_label": {int(k): int(v) for k, v in zip(*np.unique(y[m], return_counts=True))},
            "macro_ovr_auc": rep["macro_ovr_auc"],
            "binary_auc": rep["binary_auc"],
            "macro_auprc": rep["macro_auprc"],
            "per_class_ovr_auc": rep["per_class_ovr_auc"],
        }
    return out


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    aucs = []
    bins = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        # need all classes ideally; skip degenerate
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(macro_ovr_auc(yb, pb))
            y_bin = (yb > 0).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            bins.append(roc_auc_score(y_bin, 1.0 - pb[:, 0]))
        except Exception:
            continue
    def _ci(arr: list[float]) -> dict[str, float]:
        a = np.asarray(arr, dtype=float)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "lo": float(np.quantile(a, alpha / 2)),
            "hi": float(np.quantile(a, 1 - alpha / 2)),
            "n_boot_ok": int(len(a)),
        }

    return {
        "macro_ovr_auc": _ci(aucs),
        "binary_auc": _ci(bins),
        "n_boot_requested": n_boot,
        "alpha": alpha,
        "point_macro_ovr_auc": float(macro_ovr_auc(y, p)),
        "point_binary_auc": float(roc_auc_score((y > 0).astype(int), 1.0 - p[:, 0])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A floor diagnostics")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--floor-run", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or _repo()).resolve()
    run_id = args.run_id or f"diag_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    floor_run = args.floor_run or cfg["paths"]["floor_run_id"]
    out = repo / cfg["paths"]["artifacts_root"] / run_id
    out.mkdir(parents=True, exist_ok=True)
    diag = out / "diagnostics"
    diag.mkdir(exist_ok=True)

    print(f"[diag] run_id={run_id} floor={floor_run}", flush=True)

    model_path = (
        repo
        / cfg["paths"]["floor_artifacts"]
        / floor_run
        / "models"
        / cfg["paths"]["floor_model"]
    )
    if not model_path.exists():
        raise FileNotFoundError(f"floor model missing: {model_path}")
    model = joblib.load(model_path)

    df = load_merged(
        repo,
        cfg["paths"]["watch_green"],
        cfg["paths"]["pool_masks"],
        expected_n=int(cfg["data"]["expected_n"]),
    )
    # need clinical_site from pool for stratification
    meta = pd.read_parquet(repo / cfg["paths"]["pool_masks"])
    df = df.drop(columns=["clinical_site"], errors="ignore").merge(
        meta[["person_id", "clinical_site"]], on="person_id", how="left"
    )

    fcols = feature_columns(df, feature_set="full_green")
    splits = make_splits(df, fcols, feature_set="full_green")

    results: dict[str, Any] = {
        "run_id": run_id,
        "floor_run": floor_run,
        "model_path": str(model_path),
        "n_features": len(fcols),
    }

    for split_name, X, y, pids in [
        ("val", splits.X_val, splits.y_val, splits.person_id_val),
        ("test", splits.X_test, splits.y_test, splits.person_id_test),
    ]:
        proba = predict_proba(model, X)
        pred = proba.argmax(axis=1)
        rep = full_report(y, proba, y_pred=pred, tag=f"floor_{split_name}")
        write_json(diag / f"metrics_{split_name}.json", rep)

        cm_pack = confusion_pack(y, pred)
        write_json(diag / f"confusion_{split_name}.json", cm_pack)
        plot_confusion(
            np.asarray(cm_pack["matrix"]),
            diag / f"confusion_{split_name}.png",
            title=f"Floor confusion ({split_name})",
        )

        pw = pairwise_binary_auc(y, proba)
        write_json(diag / f"pairwise_auc_{split_name}.json", pw)

        # site on this split
        site_map = df.set_index("person_id")["clinical_site"]
        sites = site_map.loc[list(pids)].to_numpy()
        site_res = site_stratified(y, proba, sites)
        write_json(diag / f"site_stratified_{split_name}.json", site_res)

        results[split_name] = {
            "macro_ovr_auc": rep["macro_ovr_auc"],
            "binary_auc": rep["binary_auc"],
            "per_class_ovr_auc": rep["per_class_ovr_auc"],
            "pairwise_auc": pw,
            "confusion": cm_pack["matrix"],
            "site_stratified": site_res,
        }
        print(
            f"[diag] {split_name} auc={rep['macro_ovr_auc']:.4f} "
            f"bin={rep['binary_auc']:.4f} pairwise={pw}",
            flush=True,
        )

    # base rates
    br = {
        s: {
            "n": int((df["recommended_split"] == s).sum()),
            "by_label": {
                int(k): int(v)
                for k, v in df.loc[df["recommended_split"] == s, "label"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        for s in ("train", "val", "test")
    }
    write_json(diag / "base_rates.json", br)
    results["base_rates"] = br

    # bootstrap on test
    proba_te = predict_proba(model, splits.X_test)
    boot = bootstrap_ci(
        splits.y_test,
        proba_te,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=int(cfg["run"]["seed"]),
    )
    write_json(diag / "bootstrap_ci_test.json", boot)
    results["bootstrap_ci_test"] = boot
    print(
        f"[diag] bootstrap test 4-AUC "
        f"{boot['macro_ovr_auc']['lo']:.4f}–{boot['macro_ovr_auc']['hi']:.4f} "
        f"(point {boot['point_macro_ovr_auc']:.4f})",
        flush=True,
    )

    # Assert recomputed test matches recorded floor reference
    fr = cfg.get("floor_reference", {})
    point = float(results["test"]["macro_ovr_auc"])
    ref = float(fr.get("test_macro_ovr_auc", point))
    if abs(point - ref) > 1e-9:
        raise AssertionError(
            f"floor recompute auc={point} != floor_reference={ref}"
        )
    results["floor_reference_ok"] = True

    # short markdown report
    te = results["test"]
    lines = [
        f"# Floor diagnostics — {run_id}",
        "",
        f"Floor model: `{floor_run}` / `{cfg['paths']['floor_model']}`",
        "",
        "## Test metrics (recomputed)",
        f"- macro-OVR AUC: **{te['macro_ovr_auc']:.4f}**",
        f"- binary AUC: **{te['binary_auc']:.4f}**",
        f"- per-class OVR AUC: {te['per_class_ovr_auc']}",
        f"- pairwise AUC: {te['pairwise_auc']}",
        "",
        "## Bootstrap CI (test, person resample)",
        f"- 4-AUC: {boot['macro_ovr_auc']['lo']:.4f} – {boot['macro_ovr_auc']['hi']:.4f} "
        f"(mean {boot['macro_ovr_auc']['mean']:.4f})",
        f"- binary: {boot['binary_auc']['lo']:.4f} – {boot['binary_auc']['hi']:.4f}",
        "",
        "## Confusion (test)",
        f"```\n{np.asarray(te['confusion'])}\n```",
        "",
        "## Site-stratified (test)",
    ]
    for site, d in te["site_stratified"].items():
        if "macro_ovr_auc" in d:
            lines.append(
                f"- **{site}** n={d['n']}: 4-AUC={d['macro_ovr_auc']:.4f} "
                f"bin={d['binary_auc']:.4f} labels={d['by_label']}"
            )
        else:
            lines.append(f"- **{site}**: {d}")
    lines += [
        "",
        "## Base rates",
        f"```json\n{json.dumps(br, indent=2)}\n```",
        "",
        "## Read for 1A",
        "- Class-2 pairwise (1vs2, 2vs3) shows whether oral-med is the structural wall.",
        "- Site gaps flag confound risk before onboarding lift is interpreted.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))
    write_json(out / "diagnostics_summary.json", results)
    print(f"[diag] done → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
