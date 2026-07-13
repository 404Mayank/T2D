"""Reporting helpers: JSON-safe metrics, calibration curves, manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .metrics import full_report


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True) + "\n")


def reliability_diagrams(
    y_true: np.ndarray,
    proba: np.ndarray,
    out_path: Path,
    *,
    title: str,
    n_bins: int = 10,
) -> None:
    """Per-class reliability curves (OVR)."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(proba, dtype=float)
    n_classes = p.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(3.2 * n_classes, 3.2), squeeze=False)
    for k in range(n_classes):
        ax = axes[0, k]
        y_bin = (y == k).astype(float)
        conf = p[:, k]
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers = []
        accs = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
            if not np.any(mask):
                continue
            centers.append(conf[mask].mean())
            accs.append(y_bin[mask].mean())
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        if centers:
            ax.plot(centers, accs, "o-", label=f"class {k}")
        ax.set_title(f"class {k}")
        ax.set_xlabel("mean conf")
        ax.set_ylabel("frac positive")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_split(
    y: np.ndarray,
    proba_raw: np.ndarray,
    proba_cal: np.ndarray | None = None,
    *,
    tag: str,
) -> dict[str, Any]:
    out = {
        "raw": full_report(y, proba_raw, tag=f"{tag}_raw"),
    }
    if proba_cal is not None:
        out["calibrated"] = full_report(y, proba_cal, tag=f"{tag}_cal")
    return out
