"""B4-B representation distillation (CGM-privileged teacher → wear student).

Teacher modes (no class head in teacher loss):
  - easy:     X ∥ cgm  (original; easy recon — may copy cgm channel)
  - cgm_only: cgm + tod_sin/cos only (H1 hard privilege; no wear)
  - wear_cgm: wear X only → traj MSE (H2 hard wear→glucose map)
Student: X only; L = CE + μ || stopgrad(z_T) − z_S ||²
  z_T loss only on train∩aux; never CGM at student infer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_b.b4.data import GridBundle, GridPersonDataset, collate_grid
from training.path_b.b4.model import PatchCNNEncoder, ce_loss, masked_traj_mse
from training.path_b.b4.train import _valid_mask, extract_z, set_seed

TeacherMode = Literal["easy", "cgm_only", "wear_cgm"]


def _tod_indices(feature_cols: list[str]) -> tuple[int, int]:
    try:
        return feature_cols.index("tod_sin"), feature_cols.index("tod_cos")
    except ValueError as e:
        raise ValueError("feature_cols must include tod_sin/tod_cos for cgm_only teacher") from e


def _teacher_input(
    x: torch.Tensor,
    cgm: torch.Tensor,
    traj_mask: torch.Tensor,
    *,
    mode: TeacherMode,
    feature_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher tensor + valid mask for attention.

    Returns (x_teacher [B,T,d_t], valid_mask [B,T]).
    """
    vm_wear = None  # filled by caller via _valid_mask for wear modes
    c = cgm.unsqueeze(-1) * traj_mask.unsqueeze(-1).to(cgm.dtype)
    if mode == "easy":
        # wear + cgm; attention on wear (caller passes wear valid)
        return torch.cat([x, c], dim=-1), traj_mask  # valid overwritten by caller
    if mode == "cgm_only":
        i_sin, i_cos = _tod_indices(feature_cols)
        tod = x[:, :, [i_sin, i_cos]]
        # only attend where traj_sup_valid (has real CGM)
        xt = torch.cat([c, tod], dim=-1)
        return xt, traj_mask.bool()
    if mode == "wear_cgm":
        # wear only — hard map; caller uses wear valid mask
        return x, traj_mask  # valid overwritten by caller
    raise ValueError(f"unknown teacher mode {mode}")


def _teacher_d_in(mode: TeacherMode, n_feat: int) -> int:
    if mode == "easy":
        return n_feat + 1
    if mode == "cgm_only":
        return 3  # cgm, tod_sin, tod_cos
    if mode == "wear_cgm":
        return n_feat
    raise ValueError(mode)


def train_teacher(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
    teacher_mode: TeacherMode = "easy",
) -> dict[str, Any]:
    """Train teacher on traj recon only (train∩aux bins via traj_mask)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]))
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg.get("distill") or {}
    if quick:
        max_epochs = int(cfg["quick"]["max_epochs"])
        es_patience = int(cfg["quick"]["es_patience"])
        batch_size = int(cfg["quick"].get("batch_size", tcfg["batch_size"]))
    else:
        max_epochs = int(dcfg.get("teacher_max_epochs", tcfg["max_epochs"]))
        es_patience = int(dcfg.get("teacher_es_patience", tcfg["es_patience"]))
        batch_size = int(tcfg["batch_size"])

    d_in = _teacher_d_in(teacher_mode, len(bundle.feature_cols))
    model = PatchCNNEncoder(
        d_in,
        hidden=int(mcfg["hidden"]),
        patch_size=int(mcfg["patch_size"]),
        patch_stride=int(mcfg["patch_stride"]),
        dropout=float(mcfg["dropout"]),
        n_classes=int(mcfg["n_classes"]),
    ).to(device)

    # Teacher train loader: all train (loss masks non-aux traj to 0)
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(dcfg.get("teacher_lr", tcfg["lr"])),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(tcfg["plateau_factor"]), patience=int(tcfg["plateau_patience"])
    )

    best_rmse = float("inf")
    best_state = None
    best_ep = -1
    bad = 0
    history: list[dict[str, Any]] = []

    def _pack(
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["X"].to(device)
        cgm = batch["cgm"].to(device)
        tm = batch["traj_mask"].to(device)
        xt, vm_hint = _teacher_input(
            x, cgm, tm, mode=teacher_mode, feature_cols=bundle.feature_cols
        )
        if teacher_mode == "cgm_only":
            vm = vm_hint.bool()
        else:
            # wear modes: attend on wear; traj loss still uses traj_mask
            vm = _valid_mask(batch).to(device)
        return xt, cgm, tm, vm

    for ep in range(1, max_epochs + 1):
        model.train()
        total = n_b = 0.0
        for batch in train_loader:
            xt, cgm, tm, vm = _pack(batch)
            if not tm.any():
                continue
            out = model(xt, vm)
            loss = masked_traj_mse(out["cgm_pred"], cgm, tm)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            total += float(loss.item())
            n_b += 1

        # val traj metrics (aux bins only via traj_mask)
        model.eval()
        sse = n_el = 0.0
        sum_p = sum_t = sum_pp = sum_tt = sum_pt = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if not batch["traj_mask"].any():
                    continue
                xt, cgm, tm, vm = _pack(batch)
                out = model(xt, vm)
                pred = out["cgm_pred"]
                m = tm.bool()
                d = (pred - cgm)[m]
                sse += float((d ** 2).sum().item())
                n_el += int(d.numel())
                p = pred[m].double()
                t = cgm[m].double()
                sum_p += float(p.sum())
                sum_t += float(t.sum())
                sum_pp += float((p * p).sum())
                sum_tt += float((t * t).sum())
                sum_pt += float((p * t).sum())
        if n_el > 0:
            rmse = float(np.sqrt(sse / n_el))
            mean_p, mean_t = sum_p / n_el, sum_t / n_el
            var_p = max(sum_pp / n_el - mean_p ** 2, 0.0)
            var_t = max(sum_tt / n_el - mean_t ** 2, 0.0)
            cov = sum_pt / n_el - mean_p * mean_t
            pear = float(cov / (np.sqrt(var_p * var_t) + 1e-12)) if var_p > 0 and var_t > 0 else float("nan")
        else:
            rmse, pear = float("nan"), float("nan")

        row = {
            "epoch": ep,
            "train_traj_mse": total / max(n_b, 1),
            "val_traj_rmse": rmse,
            "val_traj_pearson": pear,
        }
        history.append(row)
        print(
            f"  teacher[{teacher_mode}] ep{ep} train_mse={row['train_traj_mse']:.4f} "
            f"val_rmse={rmse:.4f} val_pear={pear:.3f}"
        )

        if np.isfinite(rmse):
            sched.step(rmse)
            if rmse < best_rmse - 1e-5:
                best_rmse = rmse
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= es_patience:
                    print(f"  teacher early stop ep{ep} best_ep={best_ep} best_rmse={best_rmse:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state": model.state_dict(),
        "role": f"teacher_{teacher_mode}",
        "teacher_mode": teacher_mode,
        "d_in": d_in,
        "feat_mean": bundle.feat_mean,
        "feat_std": bundle.feat_std,
        "cgm_mean": bundle.cgm_mean,
        "cgm_std": bundle.cgm_std,
        "feature_cols": bundle.feature_cols,
        "model_cfg": mcfg,
        "best_epoch": best_ep,
        "best_val_traj_rmse": best_rmse,
    }
    torch.save(ckpt, out_dir / "teacher.pt")
    with open(out_dir / "teacher_history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    with open(out_dir / "teacher_metrics.json", "w") as f:
        json.dump(
            {
                "teacher_mode": teacher_mode,
                "best_epoch": best_ep,
                "best_val_traj_rmse": best_rmse,
                "history_tail": history[-3:],
            },
            f,
            indent=2,
            default=float,
        )

    # cache z_T for all pids (metrics + student train uses train only in loss)
    model.eval()
    n = len(bundle.pids)
    z_t = np.zeros((n, model.hidden), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {
                "X": torch.from_numpy(bundle.X[start:end]),
                "cgm": torch.from_numpy(bundle.cgm[start:end]),
                "traj_mask": torch.from_numpy(bundle.traj_mask[start:end]),
                "pad_mask": torch.from_numpy(bundle.pad_mask[start:end]),
                "wear_mask": torch.from_numpy(bundle.wear_mask[start:end]),
            }
            xt, _, _, vm = _pack(batch)
            # cgm_only persons with zero traj bins: vm all-false — force first bin
            if teacher_mode == "cgm_only":
                none = ~vm.any(dim=1)
                if none.any():
                    vm = vm.clone()
                    vm[none, 0] = True
            out = model(xt, vm)
            z_t[start:end] = out["z"].cpu().numpy()
    np.savez_compressed(
        out_dir / "teacher_z.npz",
        z=z_t,
        pid=bundle.pids,
        y=bundle.y,
        split=np.asarray(bundle.split, dtype="U16"),
        aux=bundle.aux_eligible.astype(np.bool_),
    )
    return {"model": model, "best_val_traj_rmse": best_rmse, "z_t": z_t}


def train_student_distill(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    z_teacher: np.ndarray,
    mu: float,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
) -> dict[str, Any]:
    """Student: CE + μ MSE(z_S, stopgrad z_T) on train aux persons."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]) + int(mu * 1000))
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

    # map pid → row for z_T (bundle order assumed)
    if len(z_teacher) != len(bundle.pids):
        raise AssertionError("z_teacher length != bundle")
    z_t_all = torch.tensor(z_teacher, dtype=torch.float32, device=device)
    aux_all = torch.tensor(bundle.aux_eligible, dtype=torch.bool, device=device)
    # index by position in bundle for dataset indices
    pid_to_i = {int(p): i for i, p in enumerate(bundle.pids)}

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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=float(tcfg["plateau_factor"]), patience=int(tcfg["plateau_patience"])
    )

    best_auc = -1.0
    best_state = None
    best_ep = -1
    bad = 0
    history: list[dict[str, Any]] = []

    for ep in range(1, max_epochs + 1):
        model.train()
        tot = tot_ce = tot_d = n_b = 0.0
        for batch in train_loader:
            x = batch["X"].to(device)
            y = batch["y"].to(device)
            vm = _valid_mask(batch).to(device)
            pids = batch["pid"].cpu().numpy()
            idx = [pid_to_i[int(p)] for p in pids]
            z_t = z_t_all[idx]
            aux = aux_all[idx]

            out = model(x, vm)
            loss_ce = ce_loss(out["logits"], y, class_w)
            # distill only aux persons in batch
            if aux.any():
                z_s = out["z"][aux]
                z_tg = z_t[aux].detach()
                loss_d = F.mse_loss(z_s, z_tg)
            else:
                loss_d = out["z"].new_zeros(())
            loss = loss_ce + float(mu) * loss_d

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            tot += float(loss.item())
            tot_ce += float(loss_ce.item())
            tot_d += float(loss_d.item()) if torch.is_tensor(loss_d) else float(loss_d)
            n_b += 1

        # val AUC
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(device)
                vm = _valid_mask(batch).to(device)
                out = model(x, vm)
                ps.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
                ys.append(batch["y"].numpy())
        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        val_auc = float(macro_ovr_auc(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
        row = {
            "epoch": ep,
            "loss": tot / max(n_b, 1),
            "ce": tot_ce / max(n_b, 1),
            "distill": tot_d / max(n_b, 1),
            "val_4auc": val_auc,
        }
        history.append(row)
        print(
            f"  student μ={mu} ep{ep} loss={row['loss']:.4f} ce={row['ce']:.4f} "
            f"dist={row['distill']:.4f} val_auc={val_auc:.4f}"
        )

        if np.isfinite(val_auc):
            sched.step(val_auc)
            if val_auc > best_auc + 1e-5:
                best_auc = val_auc
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= es_patience:
                    print(f"  student early stop ep{ep} best_ep={best_ep} best_auc={best_auc:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    def eval_split(loader: DataLoader, tag: str) -> dict[str, Any]:
        model.eval()
        ys, ps, pids = [], [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["X"].to(device)
                vm = _valid_mask(batch).to(device)
                out = model(x, vm)
                ps.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
                ys.append(batch["y"].numpy())
                pids.append(batch["pid"].numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        pid = np.concatenate(pids)
        rep = full_report(y, p, tag=tag)
        return {"report": rep, "y": y, "proba": p, "pid": pid}

    metrics: dict[str, Any] = {"mu": mu, "best_epoch": best_ep, "best_val_4auc": best_auc}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        ev = eval_split(loader, name)
        metrics[name] = ev["report"]
        np.savez_compressed(
            out_dir / f"{name}_preds.npz", y=ev["y"], proba=ev["proba"], pid=ev["pid"]
        )

    z_all, y_all, pid_all = extract_z(model, bundle, device, batch_size=batch_size)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        z=z_all,
        y=y_all,
        pid=pid_all,
        split=np.asarray(bundle.split, dtype="U16"),
    )

    # distill alignment diagnostic on train aux
    aux_m = bundle.aux_eligible & (bundle.split == "train")
    if aux_m.any():
        zs = z_all[aux_m]
        zt = z_teacher[aux_m]
        # mean cosine
        zs_n = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-8)
        zt_n = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-8)
        cos = float(np.mean(np.sum(zs_n * zt_n, axis=1)))
        mse = float(np.mean((zs - zt) ** 2))
    else:
        cos, mse = float("nan"), float("nan")
    metrics["train_aux_z_cosine"] = cos
    metrics["train_aux_z_mse"] = mse

    torch.save(
        {
            "model_state": model.state_dict(),
            "role": "student_distill",
            "mu": mu,
            "feature_cols": bundle.feature_cols,
            "model_cfg": mcfg,
            "best_epoch": best_ep,
            "best_val_4auc": best_auc,
            "feat_mean": bundle.feat_mean,
            "feat_std": bundle.feat_std,
        },
        out_dir / "checkpoint.pt",
    )
    slim = {k: v for k, v in metrics.items() if k not in ()}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(slim, f, indent=2, default=float)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    return metrics


def run_distill_pipeline(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    device: torch.device,
    mus: list[float],
    quick: bool = False,
    resume: bool = False,
    teacher_mode: TeacherMode = "easy",
) -> dict[str, Any]:
    """Teacher once + students for each μ. Returns summary."""
    tdir = run_dir / "teacher"
    if resume and (tdir / "teacher_z.npz").exists():
        print(f"=== resume teacher [{teacher_mode}] ===")
        z_t = np.load(tdir / "teacher_z.npz")["z"]
        with open(tdir / "teacher_metrics.json") as f:
            tmet = json.load(f)
    else:
        print(f"=== train teacher mode={teacher_mode} ===")
        tres = train_teacher(
            bundle,
            cfg,
            out_dir=tdir,
            device=device,
            quick=quick,
            teacher_mode=teacher_mode,
        )
        z_t = tres["z_t"]
        tmet = {
            "teacher_mode": teacher_mode,
            "best_val_traj_rmse": tres["best_val_traj_rmse"],
        }

    summary: dict[str, Any] = {"teacher": tmet, "teacher_mode": teacher_mode, "students": {}}
    for mu in mus:
        sdir = run_dir / f"mu_{str(float(mu)).replace('.', 'p')}"
        if resume and (sdir / "metrics.json").exists():
            print(f"=== resume student μ={mu} ===")
            with open(sdir / "metrics.json") as f:
                summary["students"][str(mu)] = json.load(f)
            continue
        print(f"=== train student μ={mu} (teacher={teacher_mode}) ===")
        met = train_student_distill(
            bundle, cfg, z_teacher=z_t, mu=mu, out_dir=sdir, device=device, quick=quick
        )
        summary["students"][str(mu)] = {
            k: v
            for k, v in met.items()
            if k
            in (
                "mu",
                "best_epoch",
                "best_val_4auc",
                "train",
                "val",
                "test",
                "train_aux_z_cosine",
                "train_aux_z_mse",
            )
        }
    with open(run_dir / "distill_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
