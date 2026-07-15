"""B4-B / B4-V2 representation distillation (CGM-privileged teacher → wear student).

Teacher modes (no class head in teacher loss):
  - easy:     X ∥ cgm  (original; easy recon — may copy cgm channel)
  - cgm_only: cgm + tod_sin/cos only (H1 hard privilege; no wear)
  - wear_cgm: wear X only → traj MSE (H2 hard wear→glucose map)
Student: X only; L = CE + μ * distill(z_S, sg z_T)
  Distill objectives: l2 (v1 frozen) | rkd (V2 primary) | crd (V2 sensitivity)
  z_T loss only on train∩aux; never CGM at student infer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_b.b4.augment import augment_wear_batch
from training.path_b.b4.data import GridBundle, GridPersonDataset, collate_grid
from training.path_b.b4.losses_crd import MemoryBank, crd_nce_loss
from training.path_b.b4.losses_rkd import ProjectionHead, rkd_loss
from training.path_b.b4.model import PatchCNNEncoder, ce_loss, masked_traj_mse
from training.path_b.b4.train import _valid_mask, extract_z, set_seed

TeacherMode = Literal["easy", "cgm_only", "wear_cgm"]
DistillObjective = Literal["l2", "rkd", "crd"]


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

        # mean0 baseline RMSE in z-space (targets z-scored; mean≈0 → RMSE = sqrt(mean t²))
        base_rmse = float(np.sqrt(sum_tt / n_el)) if n_el > 0 else float("nan")
        beats_mean = bool(np.isfinite(rmse) and np.isfinite(base_rmse) and rmse < base_rmse - 1e-6)
        row = {
            "epoch": ep,
            "train_traj_mse": total / max(n_b, 1),
            "val_traj_rmse": rmse,
            "val_traj_pearson": pear,
            "val_traj_baseline_rmse_mean0": base_rmse,
            "val_traj_beats_mean": beats_mean,
        }
        history.append(row)
        print(
            f"  teacher[{teacher_mode}] ep{ep} train_mse={row['train_traj_mse']:.4f} "
            f"val_rmse={rmse:.4f} val_pear={pear:.3f} beats_mean={beats_mean}"
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
    # metrics at best-RMSE epoch (not last epoch — H-1)
    best_row = next((h for h in history if h["epoch"] == best_ep), history[-1] if history else {})
    with open(out_dir / "teacher_metrics.json", "w") as f:
        json.dump(
            {
                "teacher_mode": teacher_mode,
                "best_epoch": best_ep,
                "best_val_traj_rmse": best_rmse,
                "best_val_traj_pearson": best_row.get("val_traj_pearson"),
                "best_val_traj_beats_mean": best_row.get("val_traj_beats_mean"),
                "best_val_traj_baseline_rmse_mean0": best_row.get("val_traj_baseline_rmse_mean0"),
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
    return {
        "model": model,
        "best_val_traj_rmse": best_rmse,
        "best_val_traj_pearson": best_row.get("val_traj_pearson"),
        "best_val_traj_beats_mean": best_row.get("val_traj_beats_mean"),
        "z_t": z_t,
    }


def train_student_distill(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    z_teacher: np.ndarray,
    mu: float,
    out_dir: Path,
    device: torch.device,
    quick: bool = False,
    objective: DistillObjective = "l2",
    dist_ratio: float = 1.0,
    angle_ratio: float = 2.0,
    use_aug: bool = False,
) -> dict[str, Any]:
    """Student: CE + μ * distill(z_S, stopgrad z_T) on train∩aux persons.

    objective: l2 (v1) | rkd | crd. Projection heads used for rkd/crd.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]) + int(mu * 1000))
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg.get("distill") or {}
    if quick:
        max_epochs = int(cfg["quick"]["max_epochs"])
        es_patience = int(cfg["quick"]["es_patience"])
        batch_size = int(cfg["quick"].get("batch_size", tcfg["batch_size"]))
    else:
        max_epochs = int(tcfg["max_epochs"])
        es_patience = int(tcfg["es_patience"])
        batch_size = int(tcfg["batch_size"])

    d_in = len(bundle.feature_cols)
    hidden = int(mcfg["hidden"])
    proj_dim = int(dcfg.get("proj_dim", 128))
    model = PatchCNNEncoder(
        d_in,
        hidden=hidden,
        patch_size=int(mcfg["patch_size"]),
        patch_stride=int(mcfg["patch_stride"]),
        dropout=float(mcfg["dropout"]),
        n_classes=int(mcfg["n_classes"]),
    ).to(device)

    student_proj: ProjectionHead | None = None
    teacher_proj: ProjectionHead | None = None
    bank: MemoryBank | None = None
    if objective in ("rkd", "crd"):
        student_proj = ProjectionHead(hidden, proj_dim).to(device)
        teacher_proj = ProjectionHead(hidden, proj_dim).to(device)
    if objective == "crd":
        bank_size = int(dcfg.get("crd_bank_size", 1280))
        bank = MemoryBank(proj_dim, bank_size).to(device)

    if len(z_teacher) != len(bundle.pids):
        raise AssertionError("z_teacher length != bundle")
    z_t_all = torch.tensor(z_teacher, dtype=torch.float32, device=device)
    aux_all = torch.tensor(bundle.aux_eligible, dtype=torch.bool, device=device)
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

    # teacher_proj is stopgrad in loss — do not put in optimizer (N-10)
    params: list[nn.Parameter] = list(model.parameters())
    if student_proj is not None:
        params += list(student_proj.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=float(tcfg["plateau_factor"]), patience=int(tcfg["plateau_patience"])
    )

    best_auc = -1.0
    best_state = None
    best_proj_s = None
    best_proj_t = None
    best_ep = -1
    bad = 0
    history: list[dict[str, Any]] = []
    tau = float(dcfg.get("crd_temperature", 0.07))
    aug_p = float(dcfg.get("aug_p", 0.5)) if use_aug else 0.0

    for ep in range(1, max_epochs + 1):
        model.train()
        if student_proj is not None:
            student_proj.train()
            teacher_proj.train()  # type: ignore[union-attr]
        tot = tot_ce = tot_d = n_b = 0.0
        for batch in train_loader:
            x = batch["X"].to(device)
            y = batch["y"].to(device)
            vm = _valid_mask(batch).to(device)
            if aug_p > 0:
                x = augment_wear_batch(x, vm, p=aug_p)
            pids = batch["pid"].cpu().numpy()
            idx = [pid_to_i[int(p)] for p in pids]
            z_t = z_t_all[idx]
            aux = aux_all[idx]

            out = model(x, vm)
            loss_ce = ce_loss(out["logits"], y, class_w)
            # distill only aux persons in batch (train split already)
            pending_bank: list[torch.Tensor] = []
            if float(mu) > 0 and aux.any():
                z_s = out["z"][aux]
                z_tg = z_t[aux].detach()
                if objective == "l2":
                    loss_d = F.mse_loss(z_s, z_tg)
                elif objective == "rkd":
                    assert student_proj is not None and teacher_proj is not None
                    ps = student_proj(z_s)
                    with torch.no_grad():
                        pt = teacher_proj(z_tg)
                    loss_d = rkd_loss(
                        ps, pt, dist_ratio=dist_ratio, angle_ratio=angle_ratio
                    )
                elif objective == "crd":
                    assert student_proj is not None and teacher_proj is not None and bank is not None
                    ps = student_proj(z_s)
                    with torch.no_grad():
                        pt = teacher_proj(z_tg)
                    loss_d = crd_nce_loss(ps, pt, bank, temperature=tau)
                    non_aux = ~aux
                    with torch.no_grad():
                        pending_bank.append(ps.detach())
                        if non_aux.any():
                            pending_bank.append(student_proj(out["z"][non_aux]).detach())
                else:
                    raise ValueError(objective)
            else:
                loss_d = out["z"].new_zeros(())
                if (
                    float(mu) > 0
                    and objective == "crd"
                    and bank is not None
                    and student_proj is not None
                ):
                    with torch.no_grad():
                        pending_bank.append(student_proj(out["z"]).detach())
            loss = loss_ce + float(mu) * loss_d

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
            # CRD bank update only after backward (avoid inplace version bump on graph)
            if bank is not None and pending_bank:
                with torch.no_grad():
                    for t_ in pending_bank:
                        bank.enqueue(t_)
            tot += float(loss.item())
            tot_ce += float(loss_ce.item())
            tot_d += float(loss_d.item()) if torch.is_tensor(loss_d) else float(loss_d)
            n_b += 1

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
            "objective": objective,
        }
        history.append(row)
        print(
            f"  student obj={objective} μ={mu} ep{ep} loss={row['loss']:.4f} ce={row['ce']:.4f} "
            f"dist={row['distill']:.4f} val_auc={val_auc:.4f}"
        )

        if np.isfinite(val_auc):
            sched.step(val_auc)
            if val_auc > best_auc + 1e-5:
                best_auc = val_auc
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if student_proj is not None:
                    best_proj_s = {
                        k: v.detach().cpu().clone() for k, v in student_proj.state_dict().items()
                    }
                    best_proj_t = {
                        k: v.detach().cpu().clone()
                        for k, v in teacher_proj.state_dict().items()  # type: ignore[union-attr]
                    }
                bad = 0
            else:
                bad += 1
                if bad >= es_patience:
                    print(f"  student early stop ep{ep} best_ep={best_ep} best_auc={best_auc:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    if student_proj is not None and best_proj_s is not None:
        student_proj.load_state_dict(best_proj_s)
        teacher_proj.load_state_dict(best_proj_t)  # type: ignore[union-attr, arg-type]

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

    metrics: dict[str, Any] = {
        "mu": mu,
        "objective": objective,
        "dist_ratio": dist_ratio,
        "angle_ratio": angle_ratio,
        "best_epoch": best_ep,
        "best_val_4auc": best_auc,
    }
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

    aux_m = bundle.aux_eligible & (bundle.split == "train")
    if aux_m.any():
        zs = z_all[aux_m]
        zt = z_teacher[aux_m]
        zs_n = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-8)
        zt_n = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-8)
        cos = float(np.mean(np.sum(zs_n * zt_n, axis=1)))
        mse = float(np.mean((zs - zt) ** 2))
    else:
        cos, mse = float("nan"), float("nan")
    metrics["train_aux_z_cosine"] = cos
    metrics["train_aux_z_mse"] = mse

    ckpt: dict[str, Any] = {
        "model_state": model.state_dict(),
        "role": "student_distill",
        "mu": mu,
        "objective": objective,
        "feature_cols": bundle.feature_cols,
        "model_cfg": mcfg,
        "best_epoch": best_ep,
        "best_val_4auc": best_auc,
        "feat_mean": bundle.feat_mean,
        "feat_std": bundle.feat_std,
    }
    if best_proj_s is not None:
        ckpt["student_proj_state"] = best_proj_s
        ckpt["teacher_proj_state"] = best_proj_t
        ckpt["proj_dim"] = proj_dim
    torch.save(ckpt, out_dir / "checkpoint.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    return metrics


def probe_teacher_z(
    bundle: GridBundle,
    z_teacher: np.ndarray,
    *,
    out_dir: Path,
    seed: int = 42,
    auc_go_threshold: float = 0.55,
) -> dict[str, Any]:
    """Linear + MLP + 5-NN probes on teacher z (train∩aux fit, val∩aux eval).

    STOP/GO per PLAN_B4_V2 §3.3.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)
    train_m = (bundle.split == "train") & bundle.aux_eligible
    val_m = (bundle.split == "val") & bundle.aux_eligible
    if train_m.sum() < 20 or val_m.sum() < 5:
        result = {
            "go": False,
            "reason": "insufficient train∩aux or val∩aux",
            "n_train_aux": int(train_m.sum()),
            "n_val_aux": int(val_m.sum()),
        }
        with open(out_dir / "teacher_probe.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    Xtr, ytr = z_teacher[train_m], bundle.y[train_m]
    Xva, yva = z_teacher[val_m], bundle.y[val_m]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    def _ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(macro_ovr_auc(y_true, proba))

    # linear (sklearn ≥1.8 dropped multi_class; multinomial is default for multinomial loss)
    lin = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=seed, solver="lbfgs"
    )
    lin.fit(Xtr_s, ytr)
    lin_proba = lin.predict_proba(Xva_s)
    # align columns to 0..3 if needed
    lin_full = np.zeros((len(yva), 4), dtype=np.float64)
    for j, c in enumerate(lin.classes_):
        lin_full[:, int(c)] = lin_proba[:, j]
    lin_auc = _ovr_auc(yva, lin_full)

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        alpha=1e-3,
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(Xtr_s, ytr)
    mlp_proba = mlp.predict_proba(Xva_s)
    mlp_full = np.zeros((len(yva), 4), dtype=np.float64)
    for j, c in enumerate(mlp.classes_):
        mlp_full[:, int(c)] = mlp_proba[:, j]
    mlp_auc = _ovr_auc(yva, mlp_full)

    # 5-NN
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(Xtr_s, ytr)
    knn_proba = knn.predict_proba(Xva_s)
    knn_full = np.zeros((len(yva), 4), dtype=np.float64)
    for j, c in enumerate(knn.classes_):
        knn_full[:, int(c)] = knn_proba[:, j]
    knn_auc = _ovr_auc(yva, knn_full)
    knn_acc = float(knn.score(Xva_s, yva))

    # traj quality from teacher metrics file if present (caller may pass)
    go_probe = bool(
        (np.isfinite(mlp_auc) and mlp_auc > auc_go_threshold)
        or (np.isfinite(knn_auc) and knn_auc > auc_go_threshold)
    )
    result = {
        "n_train_aux": int(train_m.sum()),
        "n_val_aux": int(val_m.sum()),
        "linear_val_4auc": lin_auc,
        "mlp_val_4auc": mlp_auc,
        "knn5_val_4auc": knn_auc,
        "knn5_val_acc": knn_acc,
        "auc_go_threshold": auc_go_threshold,
        "probe_go": go_probe,
        # final go needs traj non-deg too — set by caller
        "go": go_probe,
        "note": "final GO = probe_go AND traj non-deg (caller)",
    }
    with open(out_dir / "teacher_probe.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(
        f"  teacher probe val∩aux: lin={lin_auc:.3f} mlp={mlp_auc:.3f} "
        f"knn={knn_auc:.3f} go={go_probe}"
    )
    return result


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
    objective: DistillObjective = "l2",
    dist_ratio: float = 1.0,
    angle_ratio: float = 2.0,
    use_aug: bool = False,
    run_probe: bool = False,
    force_students: bool = False,
) -> dict[str, Any]:
    """Teacher once + optional probe + students for each μ."""
    dcfg = cfg.get("distill") or {}
    auc_thr = float(dcfg.get("probe_auc_threshold", 0.55))
    pearson_min = float(dcfg.get("traj_pearson_min", 0.15))

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
            "best_val_traj_pearson": tres.get("best_val_traj_pearson"),
            "best_val_traj_beats_mean": tres.get("best_val_traj_beats_mean"),
            "best_epoch": None,  # filled from metrics file if needed
        }
        tm_path = tdir / "teacher_metrics.json"
        if tm_path.exists():
            tm_full = json.loads(tm_path.read_text())
            tmet["best_epoch"] = tm_full.get("best_epoch")
            tmet["best_val_traj_pearson"] = tm_full.get("best_val_traj_pearson")
            tmet["best_val_traj_beats_mean"] = tm_full.get("best_val_traj_beats_mean")

    # traj non-deg from best-RMSE epoch (not last) + beats mean-predictor
    traj_pearson = float("nan")
    traj_beats_mean = False
    if "best_val_traj_pearson" in tmet and tmet["best_val_traj_pearson"] is not None:
        traj_pearson = float(tmet["best_val_traj_pearson"])
        traj_beats_mean = bool(tmet.get("best_val_traj_beats_mean", False))
    else:
        tmet_path = tdir / "teacher_metrics.json"
        if tmet_path.exists():
            tm_full = json.loads(tmet_path.read_text())
            if tm_full.get("best_val_traj_pearson") is not None:
                traj_pearson = float(tm_full["best_val_traj_pearson"])
                traj_beats_mean = bool(tm_full.get("best_val_traj_beats_mean", False))
        if not np.isfinite(traj_pearson):
            # fallback: history row at best_epoch
            thist = tdir / "teacher_history.json"
            if thist.exists() and tmet.get("best_epoch") is not None:
                hist = json.loads(thist.read_text())
                best_ep = int(tmet["best_epoch"])
                row = next((h for h in hist if int(h.get("epoch", -1)) == best_ep), None)
                if row is not None:
                    traj_pearson = float(row.get("val_traj_pearson", float("nan")))
                    traj_beats_mean = bool(row.get("val_traj_beats_mean", False))
    tmet["val_traj_pearson_best"] = traj_pearson
    tmet["val_traj_beats_mean"] = traj_beats_mean
    traj_ok = bool(
        np.isfinite(traj_pearson)
        and traj_pearson >= pearson_min
        and traj_beats_mean
    )

    probe_res: dict[str, Any] | None = None
    go = True
    if run_probe:
        print(f"=== teacher probe (val∩aux) thr={auc_thr} ===")
        probe_res = probe_teacher_z(
            bundle, z_t, out_dir=tdir, seed=int(cfg["train"]["seed"]), auc_go_threshold=auc_thr
        )
        go = bool(probe_res.get("probe_go")) and traj_ok
        probe_res["traj_pearson"] = traj_pearson
        probe_res["traj_beats_mean"] = traj_beats_mean
        probe_res["traj_ok"] = traj_ok
        probe_res["go"] = go
        with open(tdir / "teacher_probe.json", "w") as f:
            json.dump(probe_res, f, indent=2, default=float)
        print(
            f"  STOP/GO final go={go} (probe={probe_res.get('probe_go')} "
            f"traj_ok={traj_ok} pear={traj_pearson:.3f} beats_mean={traj_beats_mean})"
        )

    summary: dict[str, Any] = {
        "teacher": tmet,
        "teacher_mode": teacher_mode,
        "objective": objective,
        "probe": probe_res,
        "go": go,
        "students": {},
    }

    if run_probe and not go and not force_students:
        print("=== STOP distill students (teacher probe/traj failed) ===")
        with open(run_dir / "distill_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return summary

    for mu in mus:
        sdir = run_dir / f"mu_{str(float(mu)).replace('.', 'p')}"
        if resume and (sdir / "metrics.json").exists():
            print(f"=== resume student μ={mu} ===")
            with open(sdir / "metrics.json") as f:
                summary["students"][str(mu)] = json.load(f)
            continue
        print(
            f"=== train student obj={objective} μ={mu} (teacher={teacher_mode}) ==="
        )
        met = train_student_distill(
            bundle,
            cfg,
            z_teacher=z_t,
            mu=mu,
            out_dir=sdir,
            device=device,
            quick=quick,
            objective=objective,
            dist_ratio=dist_ratio,
            angle_ratio=angle_ratio,
            use_aug=use_aug,
        )
        summary["students"][str(mu)] = {
            k: v
            for k, v in met.items()
            if k
            in (
                "mu",
                "objective",
                "best_epoch",
                "best_val_4auc",
                "train",
                "val",
                "test",
                "train_aux_z_cosine",
                "train_aux_z_mse",
                "dist_ratio",
                "angle_ratio",
            )
        }
    with open(run_dir / "distill_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
