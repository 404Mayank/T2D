"""Per-λ evaluation: metrics + once-only test preds (no cross-λ bootstrap here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from training.path_a_watch.calibrate import MulticlassCalibrator
from training.path_a_watch.metrics import full_report
from training.path_b.b1.data import PersonSeqDataset, SequenceBundle, collate_persons
from training.path_b.b1.train import load_model_from_ckpt


def _collect(
    model, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Collect person-level class preds; accumulate glu MSE without stacking variable T."""
    model.eval()
    ys, ps, pids = [], [], []
    sse = 0.0
    sae = 0.0
    n_glu = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            wm = batch["watch_mask"].to(device)
            green = batch["green"].to(device) if "green" in batch else None
            out = model(x, wm, green=green)
            proba = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            ys.append(batch["y"].numpy())
            ps.append(proba)
            pids.append(batch["pid"].numpy())

            gp = out["glu_pred"].cpu().numpy()  # [B,T,C]
            gy = batch["glu_y"].numpy()
            gm = batch["glu_mask"].numpy().astype(bool)
            if gm.any():
                # advanced index → [n_valid, C]
                d = gp[gm] - gy[gm]
                sse += float(np.sum(d ** 2))
                sae += float(np.sum(np.abs(d)))
                n_glu += int(d.size)  # elements = n_valid * C for mean over all
                # Actually mean over elements: n_glu should be number of elements
    if n_glu > 0:
        gmet = {
            "glu_mse": sse / n_glu,
            "glu_mae": sae / n_glu,
            "n_glu": int(n_glu),  # element count
        }
    else:
        gmet = {"glu_mse": float("nan"), "glu_mae": float("nan"), "n_glu": 0}

    return {
        "y": np.concatenate(ys),
        "proba": np.concatenate(ps),
        "pid": np.concatenate(pids),
        "glu": gmet,
    }


def evaluate_split(
    model,
    bundle: SequenceBundle,
    split: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    ds = PersonSeqDataset(bundle, split)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_persons
    )
    raw = _collect(model, loader, device)
    rep = full_report(raw["y"], raw["proba"], tag=split)
    rep["glu"] = raw["glu"]
    rep["pids"] = raw["pid"]
    rep["y"] = raw["y"]
    rep["proba"] = raw["proba"]
    return rep


def run_eval(
    bundle: SequenceBundle,
    ckpt_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, blob = load_model_from_ckpt(ckpt_path, device)
    bs = int(cfg["train"]["batch_size"])

    val = evaluate_split(model, bundle, "val", device, bs)
    test = evaluate_split(model, bundle, "test", device, bs)

    # Val isotonic diagnostic
    cal_method = (cfg.get("eval") or {}).get("calibrate") or "isotonic"
    cal = MulticlassCalibrator.create(cal_method, n_classes=int(cfg["model"]["n_classes"]))
    cal.fit(val["proba"], val["y"])
    test_cal_proba = cal.transform(test["proba"])
    test_cal = full_report(test["y"], test_cal_proba, tag="test_cal")

    # Persist test preds once
    preds_path = out_dir / "test_preds.parquet"
    if preds_path.exists():
        raise AssertionError(f"test preds already exist (test-once): {preds_path}")
    df = pd.DataFrame(
        {
            "person_id": test["pids"].astype(int),
            "y": test["y"].astype(int),
            "p0": test["proba"][:, 0],
            "p1": test["proba"][:, 1],
            "p2": test["proba"][:, 2],
            "p3": test["proba"][:, 3],
        }
    )
    df.to_parquet(preds_path, index=False)

    def _slim(rep: dict) -> dict:
        skip = {"pids", "y", "proba"}
        out = {}
        for k, v in rep.items():
            if k in skip:
                continue
            if isinstance(v, dict):
                out[k] = {
                    str(kk): (
                        float(vv)
                        if isinstance(vv, (float, np.floating, int, np.integer))
                        else vv
                    )
                    for kk, vv in v.items()
                }
            elif isinstance(v, (float, np.floating)):
                out[k] = float(v)
            elif isinstance(v, (int, np.integer)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    result = {
        "ckpt": str(ckpt_path),
        "best_epoch": int(blob.get("epoch", -1)),
        "val": _slim(val),
        "test_raw": _slim(test),
        "test_cal": _slim(test_cal),
        "test_preds": str(preds_path),
        "lam": float(blob.get("lam", float("nan"))),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
