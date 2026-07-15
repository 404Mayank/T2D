"""Train loop for B1 multi-task ablation (+ GS balance modes)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_b.b1.balance import (
    ConflictMeter,
    UncertaintyWeights,
    flatten_grads,
    pcgrad_backward,
    pcgrad_combine,
    plain_backward,
    shared_parameters,
)
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


def _probe_conflict(
    model: AttnLSTM64,
    loss_ce: torch.Tensor,
    loss_glu: torch.Tensor,
    meter: ConflictMeter,
    rng: torch.Generator,
) -> None:
    """Measure shared-param cos on a live graph (caller must retain until after)."""
    shared = shared_parameters(model)
    if not shared:
        return
    try:
        g_ce = torch.autograd.grad(
            loss_ce, shared, retain_graph=True, allow_unused=True
        )
        g_glu = torch.autograd.grad(
            loss_glu, shared, retain_graph=True, allow_unused=True
        )
    except RuntimeError:
        return
    ce_filled = [
        g if g is not None else torch.zeros_like(p) for p, g in zip(shared, g_ce)
    ]
    glu_filled = [
        g if g is not None else torch.zeros_like(p) for p, g in zip(shared, g_glu)
    ]
    flat_ce = flatten_grads(ce_filled)
    flat_glu = flatten_grads(glu_filled)
    if flat_ce is None or flat_glu is None:
        return
    if float(flat_glu.norm().item()) < 1e-12:
        return
    _, stats = pcgrad_combine(flat_ce, flat_glu, rng=rng)
    meter.step_glu_active(stats)


def train_one_lambda(
    bundle: SequenceBundle,
    cfg: dict[str, Any],
    *,
    lam: float,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
    balance: str = "none",
) -> dict[str, Any]:
    """Train one arm.

    balance:
      - none / plain: L = CE + lam * glu  (frozen B1 recipe; lam=0 zeros glu loss)
      - pcgrad: PCGrad on shared params; unweighted CE + glu (lam ignored)
      - uncertainty: UW with CE-primary prior (lam ignored)
      - pcgrad_uw: PCGrad on UW-weighted task losses + full total grad on s_*
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    balance = (balance or "none").lower()
    if balance == "plain":
        balance = "none"
    if balance not in ("none", "pcgrad", "uncertainty", "pcgrad_uw"):
        raise ValueError(f"unknown balance={balance!r}")

    tcfg = cfg["train"]
    mcfg = cfg["model"]
    if quick:
        max_epochs = int(cfg["quick"]["max_epochs"])
        es_patience = int(cfg["quick"]["es_patience"])
    else:
        max_epochs = int(tcfg["max_epochs"])
        es_patience = int(tcfg["es_patience"])

    d_in = len(bundle.feature_cols)
    green_dim = int(bundle.green.shape[1]) if bundle.green is not None else 0
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

    uw: UncertaintyWeights | None = None
    if balance in ("uncertainty", "pcgrad_uw"):
        uw = UncertaintyWeights(clamp=float(tcfg.get("uw_clamp", 5.0))).to(device)

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

    lr = float(tcfg["lr"])
    wd = float(tcfg["weight_decay"])
    if uw is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        lr_s = float(tcfg.get("uw_lr_scale", 0.1)) * lr
        opt = torch.optim.AdamW(
            [
                {"params": list(model.parameters()), "lr": lr},
                {"params": list(uw.parameters()), "lr": lr_s, "weight_decay": 0.0},
            ],
            weight_decay=wd,
        )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(tcfg["plateau_factor"]),
        patience=int(tcfg["plateau_patience"]),
    )
    grad_clip = float(tcfg["grad_clip"])

    pcg_rng = torch.Generator(device="cpu")
    pcg_rng.manual_seed(seed + 17)

    best_val_auc = -1.0
    best_epoch = -1
    bad = 0
    history: list[dict[str, float]] = []
    conflict_rows: list[dict[str, float]] = []
    meter = ConflictMeter()
    ckpt_path = out_dir / "best.pt"

    use_pcgrad = balance in ("pcgrad", "pcgrad_uw")
    use_uw = balance in ("uncertainty", "pcgrad_uw")

    for epoch in range(1, max_epochs + 1):
        model.train()
        if uw is not None:
            uw.train()
        tr_ce = tr_glu = tr_total = 0.0
        n_batches = 0
        n_glu_batches = 0

        for batch in train_loader:
            x = batch["X"].to(device)
            wm = batch["watch_mask"].to(device)
            y = batch["y"].to(device)
            gy = batch["glu_y"].to(device)
            gm = batch["glu_mask"].to(device)
            glu_active = bool(gm.any().item())

            opt.zero_grad(set_to_none=True)
            out = model(x, wm, green=_batch_green(batch, device))
            loss_ce = ce_loss(out["logits"], y, class_w)
            loss_glu = masked_mse(out["glu_pred"], gy, gm)

            if use_uw:
                assert uw is not None
                total, term_ce, term_glu = uw.combine(
                    loss_ce, loss_glu, glu_active=glu_active
                )
                if use_pcgrad:
                    # PCGrad writes model grads; keep graph for s_* via full total
                    pcgrad_backward(
                        model=model,
                        loss_ce=term_ce,
                        loss_glu=term_glu if glu_active else loss_ce.new_zeros(()),
                        glu_active=glu_active,
                        meter=meter,
                        rng=pcg_rng,
                        retain_graph=True,
                    )
                    s_grads = torch.autograd.grad(
                        total, list(uw.parameters()), allow_unused=True
                    )
                    for p, g in zip(uw.parameters(), s_grads):
                        p.grad = None if g is None else g.detach().clone()
                else:
                    # conflict probe on unweighted task losses (before freeing graph)
                    meter.step_total()
                    if glu_active:
                        _probe_conflict(model, loss_ce, loss_glu, meter, pcg_rng)
                    plain_backward(
                        model=model, loss=total, also_params=uw.parameters()
                    )
            elif use_pcgrad:
                if not glu_active:
                    meter.step_total()
                    plain_backward(model=model, loss=loss_ce)
                    total = loss_ce
                else:
                    pcgrad_backward(
                        model=model,
                        loss_ce=loss_ce,
                        loss_glu=loss_glu,
                        glu_active=True,
                        meter=meter,
                        rng=pcg_rng,
                    )
                    total = loss_ce + loss_glu
            else:
                # frozen plain-λ: probe then backward (probe needs live graph)
                total = loss_ce + float(lam) * loss_glu
                meter.step_total()
                if glu_active and float(lam) > 0:
                    _probe_conflict(
                        model, loss_ce, float(lam) * loss_glu, meter, pcg_rng
                    )
                plain_backward(model=model, loss=total)

            if grad_clip > 0:
                params_to_clip = list(model.parameters())
                if uw is not None:
                    params_to_clip += list(uw.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
            opt.step()

            tr_ce += float(loss_ce.detach().cpu())
            tr_glu += float(loss_glu.detach().cpu())
            tr_total += float(total.detach().cpu())
            n_batches += 1
            if glu_active:
                n_glu_batches += 1

        tr_ce /= max(n_batches, 1)
        tr_glu /= max(n_batches, 1)
        tr_total /= max(n_batches, 1)

        y_val, p_val, _ = _predict_loader(model, val_loader, device)
        val_auc = macro_ovr_auc(y_val, p_val)
        val_rep = full_report(y_val, p_val, tag="val")
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

        conf_row = meter.epoch_summary(epoch)
        conflict_rows.append(conf_row)
        sched.step(val_auc)

        # conflict_grad_source documents which tensors fed cos (not cross-arm comparable blindly)
        if use_pcgrad and use_uw:
            conf_src = "uw_weighted_pcgrad"
        elif use_pcgrad:
            conf_src = "unweighted_pcgrad"
        elif use_uw:
            conf_src = "unweighted_probe"
        elif float(lam) > 0:
            conf_src = "plain_lambda_scaled_probe"  # cos scale-invariant; norms not
        else:
            conf_src = "none"

        row: dict[str, float | str] = {
            "epoch": epoch,
            "train_ce": tr_ce,
            "train_glu": tr_glu,
            "train_total": tr_total,
            "val_ce": v_ce,
            "val_glu": v_glu,
            "val_macro_ovr_auc": float(val_auc),
            "val_binary_auc": float(val_rep.get("binary_auc", float("nan"))),
            "lr": float(opt.param_groups[0]["lr"]),
            "n_glu_batches": float(n_glu_batches),
            "conflict_rate": conf_row["conflict_rate"],
            "mean_cos": conf_row["mean_cos"],
            "n_glu_active_steps": conf_row["n_glu_active_steps"],
            "conflict_grad_source": conf_src,
        }
        if uw is not None:
            row.update(uw.effective_weights())
        history.append(row)

        uw_msg = ""
        if uw is not None:
            ew = uw.effective_weights()
            uw_msg = f" s_ce={ew['s_ce']:+.3f} s_glu={ew['s_glu']:+.3f}"
        print(
            f"  bal={balance} λ={lam} ep={epoch:03d} tr_ce={tr_ce:.4f} tr_glu={tr_glu:.4f} "
            f"val_auc={val_auc:.4f} conf={conf_row['conflict_rate']:.3f}{uw_msg}",
            flush=True,
        )

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "uw": None if uw is None else uw.state_dict(),
                    "epoch": epoch,
                    "val_macro_ovr_auc": best_val_auc,
                    "lam": float(lam),
                    "balance": balance,
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
                print(
                    f"  early stop at ep={epoch} (best={best_epoch} auc={best_val_auc:.4f})"
                )
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "conflict.json", "w") as f:
        json.dump(conflict_rows, f, indent=2)

    return {
        "lam": float(lam),
        "balance": balance,
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
