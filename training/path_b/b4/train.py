"""Train B4-A multi-task (class + masked traj) ± PCGrad / uncertainty weighting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_b.b4.data import GridBundle, GridPersonDataset, collate_grid
from training.path_b.b4.model import PatchCNNEncoder, ce_loss, masked_traj_mse
from training.path_b.b4.pcgrad import grad_cosine, pcgrad_step
from training.path_b.b4.uncertainty import UncertaintyWeights

Balancer = Literal["none", "pcgrad", "uncertainty"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _valid_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Observed bins: not pad. Prefer wear for attention when available."""
    pad = batch["pad_mask"].bool()
    wear = batch["wear_mask"].bool()
    # use wear ∪ (~pad & any channel) — plan: pad never observed; attend on wear
    return wear & ~pad


def _predict_loader(
    model: PatchCNNEncoder, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    ys, ps, pids = [], [], []
    sse = sae = 0.0
    n_el = 0
    sum_p = sum_t = sum_pp = sum_tt = sum_pt = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            vm = _valid_mask(batch).to(device)
            out = model(x, vm)
            proba = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            ys.append(batch["y"].numpy())
            ps.append(proba)
            pids.append(batch["pid"].numpy())

            pred = out["cgm_pred"].cpu().numpy()
            tgt = batch["cgm"].numpy()
            tm = batch["traj_mask"].numpy().astype(bool)
            if tm.any():
                d = pred[tm] - tgt[tm]
                sse += float(np.sum(d ** 2))
                sae += float(np.sum(np.abs(d)))
                n_el += int(d.size)
                p = pred[tm].astype(np.float64)
                t = tgt[tm].astype(np.float64)
                sum_p += float(p.sum())
                sum_t += float(t.sum())
                sum_pp += float((p * p).sum())
                sum_tt += float((t * t).sum())
                sum_pt += float((p * t).sum())
    if n_el > 0:
        mean_p = sum_p / n_el
        mean_t = sum_t / n_el
        var_p = max(sum_pp / n_el - mean_p ** 2, 0.0)
        var_t = max(sum_tt / n_el - mean_t ** 2, 0.0)
        cov = sum_pt / n_el - mean_p * mean_t
        pearson = float(cov / (np.sqrt(var_p * var_t) + 1e-12)) if var_p > 0 and var_t > 0 else float("nan")
        # baseline: predict train global mean in z-space (=0 after z-score)
        base_rmse = float(np.sqrt(sum_tt / n_el))  # tgt already z-scored; mean~0
        rmse = float(np.sqrt(sse / n_el))
        traj = {
            "traj_mse": sse / n_el,
            "traj_mae": sae / n_el,
            "traj_rmse": rmse,
            "traj_pearson": pearson,
            "traj_baseline_rmse_mean0": base_rmse,
            "traj_beats_mean": bool(rmse < base_rmse - 1e-6),
            "n_traj": int(n_el),
        }
    else:
        traj = {
            "traj_mse": float("nan"),
            "traj_mae": float("nan"),
            "traj_rmse": float("nan"),
            "traj_pearson": float("nan"),
            "traj_baseline_rmse_mean0": float("nan"),
            "traj_beats_mean": False,
            "n_traj": 0,
        }
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pids), traj


def extract_z(
    model: PatchCNNEncoder,
    bundle: GridBundle,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z [N,H], y, pids in bundle order."""
    model.eval()
    ds = GridPersonDataset(bundle, "train")  # placeholder; we iterate all via indices
    # manual all-split loader
    n = len(bundle.pids)
    zs = np.zeros((n, model.hidden), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(bundle.X[start:end]).to(device)
            pad = torch.from_numpy(bundle.pad_mask[start:end])
            wear = torch.from_numpy(bundle.wear_mask[start:end])
            vm = (wear & ~pad).to(device)
            out = model(x, vm)
            zs[start:end] = out["z"].cpu().numpy()
    return zs, bundle.y.copy(), bundle.pids.copy()


def _shared_encoder_params(model: PatchCNNEncoder) -> list[torch.nn.Parameter]:
    """Encoder trunk only (not class_head / traj_head)."""
    shared = list(model.input.parameters()) + list(model.attn.parameters())
    return shared


def train_one_lambda(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    lam: float,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
    balancer: Balancer = "none",
) -> dict[str, Any]:
    """Train class ± traj multi-task.

    balancer:
      none — plain L = CE + λ * traj (v1)
      pcgrad — PCGrad on shared encoder; heads get native grads; λ scales traj loss
      uncertainty — Kendall UW (ignores λ scale; both tasks always on)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    tcfg = cfg["train"]
    mcfg = cfg["model"]
    if quick:
        max_epochs = int(cfg["quick"]["max_epochs"])
        es_patience = int(cfg["quick"]["es_patience"])
        batch_size = int(cfg["quick"].get("batch_size", tcfg["batch_size"]))
    else:
        max_epochs = int(tcfg["max_epochs"])
        es_patience = int(tcfg["es_patience"])
        batch_size = int(tcfg["batch_size"])

    d_in = len(bundle.feature_cols)
    model = PatchCNNEncoder(
        d_in,
        hidden=int(mcfg["hidden"]),
        patch_size=int(mcfg["patch_size"]),
        patch_stride=int(mcfg["patch_stride"]),
        dropout=float(mcfg["dropout"]),
        n_classes=int(mcfg["n_classes"]),
    ).to(device)

    uw: UncertaintyWeights | None = None
    if balancer == "uncertainty":
        uw = UncertaintyWeights().to(device)

    class_w = torch.tensor(bundle.class_weights, dtype=torch.float32, device=device)
    train_loader = DataLoader(
        GridPersonDataset(bundle, "train"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_grid,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        GridPersonDataset(bundle, "val"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_grid,
    )
    test_loader = DataLoader(
        GridPersonDataset(bundle, "test"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_grid,
    )

    params = list(model.parameters()) + (list(uw.parameters()) if uw is not None else [])
    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=float(tcfg["plateau_factor"]),
        patience=int(tcfg["plateau_patience"]),
    )

    best_auc = -1.0
    best_state = None
    best_uw_state = None
    best_ep = -1
    bad = 0
    history: list[dict[str, Any]] = []
    cos_epoch_means: list[float] = []

    shared = _shared_encoder_params(model)
    head_params = list(model.class_head.parameters()) + list(model.traj_head.parameters())

    for ep in range(1, max_epochs + 1):
        model.train()
        if uw is not None:
            uw.train()
        total_loss = total_ce = total_tr = 0.0
        n_batches = 0
        cos_batches: list[float] = []
        n_conflicts = 0

        for batch in train_loader:
            x = batch["X"].to(device)
            y = batch["y"].to(device)
            vm = _valid_mask(batch).to(device)
            cgm = batch["cgm"].to(device)
            tm = batch["traj_mask"].to(device)

            out = model(x, vm)
            loss_ce = ce_loss(out["logits"], y, class_w)
            loss_tr = masked_traj_mse(out["cgm_pred"], cgm, tm)
            loss_tr_scaled = float(lam) * loss_tr

            opt.zero_grad(set_to_none=True)

            if balancer == "pcgrad" and float(lam) > 0:
                # PCGrad on shared encoder; separate head grads (do not clobber shared)
                diag = pcgrad_step([loss_ce, loss_tr_scaled], shared, retain_graph=True)
                if np.isfinite(diag["cos_mean"]):
                    cos_batches.append(diag["cos_mean"])
                n_conflicts += int(diag["n_conflicts"])
                loss_heads = loss_ce + loss_tr_scaled
                head_grads = torch.autograd.grad(
                    loss_heads,
                    head_params,
                    retain_graph=False,
                    allow_unused=True,
                )
                for p, g in zip(head_params, head_grads):
                    if g is not None:
                        p.grad = g
                loss = loss_heads.detach()
            elif balancer == "uncertainty":
                assert uw is not None
                # cos diagnostic (retain graph) then combined UW backward
                g_ce = torch.autograd.grad(
                    loss_ce, shared, retain_graph=True, allow_unused=True
                )
                g_tr = torch.autograd.grad(
                    loss_tr, shared, retain_graph=True, allow_unused=True
                )
                flat_ce = torch.cat(
                    [
                        gi.reshape(-1)
                        if gi is not None
                        else torch.zeros(p.numel(), device=device)
                        for gi, p in zip(g_ce, shared)
                    ]
                )
                flat_tr = torch.cat(
                    [
                        gi.reshape(-1)
                        if gi is not None
                        else torch.zeros(p.numel(), device=device)
                        for gi, p in zip(g_tr, shared)
                    ]
                )
                cos_batches.append(grad_cosine(flat_ce, flat_tr))
                loss = uw.combine(loss_ce, loss_tr)
                loss.backward()
            else:
                loss = loss_ce + loss_tr_scaled
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            if uw is not None:
                torch.nn.utils.clip_grad_norm_(uw.parameters(), float(tcfg["grad_clip"]))
            opt.step()

            total_loss += float(loss.item()) if torch.is_tensor(loss) else float(loss)
            total_ce += float(loss_ce.item())
            total_tr += float(loss_tr.item())
            n_batches += 1

        if len(val_loader.dataset) == 0:
            raise RuntimeError(
                "val split empty after T_min/grid filter — widen FE smoke pids or lower t_min"
            )
        yv, pv, _, traj_v = _predict_loader(model, val_loader, device)
        val_auc = float(macro_ovr_auc(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
        cos_mean_ep = float(np.nanmean(cos_batches)) if cos_batches else float("nan")
        if np.isfinite(cos_mean_ep):
            cos_epoch_means.append(cos_mean_ep)
        row = {
            "epoch": ep,
            "loss": total_loss / max(n_batches, 1),
            "ce": total_ce / max(n_batches, 1),
            "traj": total_tr / max(n_batches, 1),
            "val_4auc": val_auc,
            "val_traj": traj_v,
            "grad_cos_mean": cos_mean_ep,
            "n_conflicts": n_conflicts,
            "balancer": balancer,
        }
        if uw is not None:
            row["uw"] = uw.state_dict_small()
        history.append(row)
        print(
            f"  bal={balancer} λ={lam} ep{ep} loss={row['loss']:.4f} ce={row['ce']:.4f} "
            f"traj={row['traj']:.4f} val_auc={val_auc:.4f} "
            f"val_pear={traj_v.get('traj_pearson', float('nan')):.3f} "
            f"cos={cos_mean_ep:.3f}"
        )

        if np.isfinite(val_auc):
            sched.step(val_auc)
            if val_auc > best_auc + 1e-5:
                best_auc = val_auc
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if uw is not None:
                    best_uw_state = {k: v.detach().cpu().clone() for k, v in uw.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= es_patience:
                    print(f"  early stop ep{ep} best_ep={best_ep} best_auc={best_auc:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    if uw is not None and best_uw_state is not None:
        uw.load_state_dict(best_uw_state)

    # conflict diagnostic over early epochs (plan G3b)
    early = cos_epoch_means[:5] if cos_epoch_means else []
    early_cos_med = float(np.nanmedian(early)) if early else float("nan")
    no_conflict = bool(np.isfinite(early_cos_med) and early_cos_med > 0)

    # final eval
    metrics: dict[str, Any] = {
        "lambda": lam,
        "balancer": balancer,
        "best_epoch": best_ep,
        "best_val_4auc": best_auc,
        "early_grad_cos_median": early_cos_med,
        "no_conflict_early": no_conflict,
        "cos_epoch_means": cos_epoch_means,
    }
    if uw is not None:
        metrics["uw_final"] = uw.state_dict_small()

    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        if len(loader.dataset) == 0:
            # OOF fold bundles have no test split — skip cleanly
            metrics[name] = {"empty": True}
            continue
        yy, pp, ppid, traj = _predict_loader(model, loader, device)
        rep = full_report(yy, pp, tag=name)
        metrics[name] = {**rep, "traj": traj, "pids": ppid.tolist()}
        metrics[f"{name}_proba"] = pp
        metrics[f"{name}_y"] = yy
        metrics[f"{name}_pid"] = ppid

    z_all, y_all, pid_all = extract_z(model, bundle, device, batch_size=batch_size)
    split_s = np.asarray(bundle.split, dtype="U16")
    np.savez_compressed(
        out_dir / "embeddings.npz",
        z=z_all,
        y=y_all,
        pid=pid_all,
        split=split_s,
    )

    ckpt = {
        "model_state": model.state_dict(),
        "lambda": lam,
        "balancer": balancer,
        "feat_mean": bundle.feat_mean,
        "feat_std": bundle.feat_std,
        "cgm_mean": bundle.cgm_mean,
        "cgm_std": bundle.cgm_std,
        "feature_cols": bundle.feature_cols,
        "model_cfg": mcfg,
        "best_epoch": best_ep,
        "best_val_4auc": best_auc,
        "early_grad_cos_median": early_cos_med,
        "no_conflict_early": no_conflict,
    }
    if uw is not None and best_uw_state is not None:
        ckpt["uw_state"] = best_uw_state
    torch.save(ckpt, out_dir / "checkpoint.pt")
    slim = {
        k: v
        for k, v in metrics.items()
        if not k.endswith("_proba") and not k.endswith("_y") and not k.endswith("_pid")
    }
    for sp in ("train", "val", "test"):
        if sp in slim and isinstance(slim[sp], dict):
            slim[sp] = {kk: vv for kk, vv in slim[sp].items() if kk != "pids"}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(slim, f, indent=2, default=float)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    if "test_y" in metrics:
        np.savez_compressed(
            out_dir / "test_preds.npz",
            y=metrics["test_y"],
            proba=metrics["test_proba"],
            pid=metrics["test_pid"],
        )
    if "val_y" in metrics:
        np.savez_compressed(
            out_dir / "val_preds.npz",
            y=metrics["val_y"],
            proba=metrics["val_proba"],
            pid=metrics["val_pid"],
        )
    return metrics


def load_model_from_ckpt(path: Path, device: torch.device) -> tuple[PatchCNNEncoder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ckpt["model_cfg"]
    d_in = len(ckpt["feature_cols"])
    model = PatchCNNEncoder(
        d_in,
        hidden=int(mcfg["hidden"]),
        patch_size=int(mcfg["patch_size"]),
        patch_stride=int(mcfg["patch_stride"]),
        dropout=float(mcfg["dropout"]),
        n_classes=int(mcfg["n_classes"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt
