"""Train loop for B1 multi-task ablation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_b.b1.data import PersonSeqDataset, SequenceBundle, collate_persons
from training.path_b.b1.model import AttnLSTM64, ce_loss, masked_mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_green(batch: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if "green" not in batch:
        return None
    return batch["green"].to(device)


def _predict_loader(
    model: AttnLSTM64, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            wm = batch["watch_mask"].to(device)
            out = model(x, wm, green=_batch_green(batch, device))
            proba = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            ys.append(batch["y"].numpy())
            ps.append(proba)
            pids.append(batch["pid"].numpy())
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(pids),
    )


def train_one_lambda(
    bundle: SequenceBundle,
    cfg: dict[str, Any],
    *,
    lam: float,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)  # reset before each λ model init

    tcfg = cfg["train"]
    mcfg = cfg["model"]
    if quick:
        max_epochs = int(cfg["quick"]["max_epochs"])
        es_patience = int(cfg["quick"]["es_patience"])
    else:
        max_epochs = int(tcfg["max_epochs"])
        es_patience = int(tcfg["es_patience"])

    d_in = len(bundle.feature_cols)
    green_dim = (
        int(bundle.green.shape[1])
        if bundle.green is not None
        else 0
    )
    model = AttnLSTM64(
        d_in,
        hidden=int(mcfg["hidden"]),
        n_classes=int(mcfg["n_classes"]),
        n_glu=len(bundle.glu_cols),
        dropout=float(mcfg["dropout"]),
        bidirectional=bool(mcfg["bidirectional"]),
        green_dim=green_dim,
    ).to(device)

    class_w = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)

    train_ds = PersonSeqDataset(bundle, "train")
    val_ds = PersonSeqDataset(bundle, "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        collate_fn=collate_persons,
        num_workers=int(tcfg.get("num_workers") or 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_persons,
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(tcfg["plateau_factor"]),
        patience=int(tcfg["plateau_patience"]),
    )
    grad_clip = float(tcfg["grad_clip"])

    best_val_auc = -1.0
    best_epoch = -1
    bad = 0
    history: list[dict[str, float]] = []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_ce = tr_glu = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["X"].to(device)
            wm = batch["watch_mask"].to(device)
            y = batch["y"].to(device)
            gy = batch["glu_y"].to(device)
            gm = batch["glu_mask"].to(device)

            opt.zero_grad(set_to_none=True)
            out = model(x, wm, green=_batch_green(batch, device))
            loss_ce = ce_loss(out["logits"], y, class_w)
            loss_glu = masked_mse(out["glu_pred"], gy, gm)
            # λ=0: still compute glu forward (matched capacity) but zero loss
            loss = loss_ce + float(lam) * loss_glu
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_ce += float(loss_ce.detach().cpu())
            tr_glu += float(loss_glu.detach().cpu())
            n_batches += 1

        tr_ce /= max(n_batches, 1)
        tr_glu /= max(n_batches, 1)

        y_val, p_val, _ = _predict_loader(model, val_loader, device)
        val_auc = macro_ovr_auc(y_val, p_val)
        val_rep = full_report(y_val, p_val, tag="val")
        # val CE (unweighted)
        with torch.no_grad():
            model.eval()
            v_ce = v_glu = 0.0
            vb = 0
            for batch in val_loader:
                x = batch["X"].to(device)
                wm = batch["watch_mask"].to(device)
                y = batch["y"].to(device)
                gy = batch["glu_y"].to(device)
                gm = batch["glu_mask"].to(device)
                out = model(x, wm, green=_batch_green(batch, device))
                v_ce += float(ce_loss(out["logits"], y, None).cpu())
                v_glu += float(masked_mse(out["glu_pred"], gy, gm).cpu())
                vb += 1
            v_ce /= max(vb, 1)
            v_glu /= max(vb, 1)

        sched.step(val_auc)
        row = {
            "epoch": epoch,
            "train_ce": tr_ce,
            "train_glu": tr_glu,
            "val_ce": v_ce,
            "val_glu": v_glu,
            "val_macro_ovr_auc": float(val_auc),
            "val_binary_auc": float(val_rep.get("binary_auc", float("nan"))),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"  λ={lam} ep={epoch:03d} tr_ce={tr_ce:.4f} tr_glu={tr_glu:.4f} "
            f"val_auc={val_auc:.4f} val_bin={row['val_binary_auc']:.4f}",
            flush=True,
        )

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_ovr_auc": best_val_auc,
                    "lam": float(lam),
                    "feature_cols": bundle.feature_cols,
                    "glu_cols": bundle.glu_cols,
                    "class_weights": bundle.class_weights.tolist(),
                    "feat_mean": bundle.feat_mean.tolist(),
                    "feat_std": bundle.feat_std.tolist(),
                    "glu_mean": bundle.glu_mean.tolist(),
                    "glu_std": bundle.glu_std.tolist(),
                    "impute_values": bundle.impute_values,
                    "cfg_model": mcfg,
                    "d_in": d_in,
                    "green_dim": green_dim,
                    "green_cols": bundle.green_cols,
                    "green_mean": (
                        bundle.green_mean.tolist()
                        if bundle.green_mean is not None
                        else None
                    ),
                    "green_std": (
                        bundle.green_std.tolist()
                        if bundle.green_std is not None
                        else None
                    ),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= es_patience:
                print(f"  early stop at ep={epoch} (best={best_epoch} auc={best_val_auc:.4f})")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "lam": float(lam),
        "best_epoch": best_epoch,
        "best_val_macro_ovr_auc": best_val_auc,
        "ckpt": str(ckpt_path),
        "n_epochs_ran": len(history),
    }


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[AttnLSTM64, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = blob["cfg_model"]
    model = AttnLSTM64(
        int(blob["d_in"]),
        hidden=int(mcfg["hidden"]),
        n_classes=int(mcfg["n_classes"]),
        n_glu=len(blob["glu_cols"]),
        dropout=float(mcfg["dropout"]),
        bidirectional=bool(mcfg["bidirectional"]),
        green_dim=int(blob.get("green_dim") or 0),
    ).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob
