"""MLP student with exact Hinton KD loss (N0 / Nα)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.metrics import full_report, macro_auprc, macro_ovr_auc
from training.path_b.b3.data import (
    assert_student_features,
    hard_class_weights,
    split_xy,
    temperature_scale_proba,
)


class StudentMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.1, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _impute_scale_fit(X: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    med = X.median(axis=0)
    Xf = X.fillna(med)
    mu = Xf.mean(axis=0).to_numpy(dtype=np.float64)
    sd = Xf.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (Xf.to_numpy(dtype=np.float64) - mu) / sd
    state = {
        "median": med.to_dict(),
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "columns": list(X.columns),
    }
    return Z, state


def _impute_scale_apply(X: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    cols = state["columns"]
    med = pd.Series(state["median"])
    Xf = X[cols].fillna(med)
    mu = np.asarray(state["mean"], dtype=np.float64)
    sd = np.asarray(state["std"], dtype=np.float64)
    return (Xf.to_numpy(dtype=np.float64) - mu) / sd


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _hinton_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    soft: torch.Tensor | None,
    has_soft: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
    class_w: torch.Tensor | None,
) -> torch.Tensor:
    """(1-α) CE + α T² KL(soft_T || student_T) on rows with soft labels."""
    ce = F.cross_entropy(logits, y, weight=class_w, reduction="none")
    if alpha <= 0 or soft is None or not has_soft.any():
        return ce.mean()

    t = float(temperature)
    log_p_s = F.log_softmax(logits / t, dim=-1)
    # soft already temperature-scaled probabilities
    kl = F.kl_div(log_p_s, soft, reduction="none").sum(dim=-1) * (t * t)

    # Strict Hinton: (1-α) CE on all rows; α T² KL only where soft labels exist.
    # Non-aux (no soft) must NOT get full CE while aux gets (1-α) CE.
    loss = (1.0 - alpha) * ce + torch.where(has_soft, alpha * kl, torch.zeros_like(ce))
    return loss.mean()


@torch.no_grad()
def _predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, bs: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=device)
        logits = model(xb)
        outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_mlp_student(
    df: pd.DataFrame,
    c1_cols: list[str],
    soft_by_pid: dict[int, np.ndarray],
    cfg: dict[str, Any],
    *,
    alpha: float,
    temperature: float | None = None,
    device: str | None = None,
    log=lambda msg: None,
    arm_name: str | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    hidden: int | None = None,
) -> dict[str, Any]:
    assert_student_features(c1_cols, cfg)
    temp = float(temperature if temperature is not None else cfg["run"]["temperature"])
    splits = split_xy(df, c1_cols, pool="core")

    Ztr, scale_state = _impute_scale_fit(splits["X_train"])
    Zva = _impute_scale_apply(splits["X_val"], scale_state)
    Zte = _impute_scale_apply(splits["X_test"], scale_state)

    ytr = splits["y_train"]
    yva = splits["y_val"]
    yte = splits["y_test"]
    pid_tr = splits["pid_train"]

    # soft matrix aligned to train rows (temperature-scaled); zeros if no soft
    n_classes = 4
    soft = np.zeros((len(ytr), n_classes), dtype=np.float64)
    has_soft = np.zeros(len(ytr), dtype=bool)
    for i, p in enumerate(pid_tr):
        p = int(p)
        if p in soft_by_pid and float(alpha) > 0:
            soft[i] = temperature_scale_proba(soft_by_pid[p], temp)
            has_soft[i] = True

    cw_np = hard_class_weights(ytr, n_classes=n_classes)
    dev = _device(device)
    if dev.type == "cuda":
        # ROCm MIOpen reduction quirk seen on B1
        try:
            torch.backends.cudnn.enabled = False
        except Exception:
            pass

    hidden = int(hidden if hidden is not None else cfg["run"]["mlp_hidden"])
    dropout = float(cfg["run"]["mlp_dropout"])
    epochs = int(cfg["run"]["mlp_epochs"])
    patience = int(cfg["run"]["mlp_patience"])
    lr = float(lr if lr is not None else cfg["run"]["mlp_lr"])
    wd = float(weight_decay if weight_decay is not None else cfg["run"]["mlp_weight_decay"])
    bs = int(cfg["run"]["mlp_batch_size"])
    seed = int(cfg["run"]["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = StudentMLP(Ztr.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    class_w = torch.tensor(cw_np, dtype=torch.float32, device=dev)

    ds = TensorDataset(
        torch.tensor(Ztr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.long),
        torch.tensor(soft, dtype=torch.float32),
        torch.tensor(has_soft, dtype=torch.bool),
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    best_state = None
    best_auc = -1.0
    bad = 0
    history: list[dict[str, float]] = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, sb, hb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            sb = sb.to(dev)
            hb = hb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = _hinton_loss(
                logits,
                yb,
                sb,
                hb,
                alpha=float(alpha),
                temperature=temp,
                class_w=class_w,
            )
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        proba_va = _predict_proba(model, Zva, dev)
        auc = float(macro_ovr_auc(yva, proba_va))
        history.append({"epoch": ep, "loss": total / max(n, 1), "val_auc": auc})
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev)

    proba_val = _predict_proba(model, Zva, dev)
    proba_test = _predict_proba(model, Zte, dev)

    cal = fit_calibrators(
        proba_val,
        yva,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test_cal = cal["primary"].transform(proba_test)

    name = arm_name or (f"N_a={alpha}" if alpha != 0 else "N0")
    log(
        f"{name} best_val_auc={best_auc:.4f} epochs={len(history)} "
        f"n_soft={int(has_soft.sum())} device={dev}"
    )
    return {
        "arm": name,
        "alpha": float(alpha),
        "temperature": temp,
        "family": "mlp",
        "params": {
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": wd,
            "batch_size": bs,
        },
        "best_epoch": int(np.argmax([h["val_auc"] for h in history])) if history else 0,
        "val_macro_ovr_auc": float(macro_ovr_auc(yva, proba_val)),
        "val_macro_auprc": float(macro_auprc(yva, proba_val)),
        "model": model.cpu(),
        "scale_state": scale_state,
        "proba_val": proba_val,
        "proba_test": proba_test,
        "proba_test_cal": proba_test_cal,
        "calibrator": cal,
        "metrics": {
            "val_raw": full_report(yva, proba_val, tag="val_raw"),
            "test_raw": full_report(yte, proba_test, tag="test_raw"),
            "test_cal": full_report(yte, proba_test_cal, tag="test_cal_primary"),
        },
        "feature_cols": list(c1_cols),
        "pool": "core",
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_test": splits["n_test"],
        "pid_test": splits["pid_test"],
        "y_test": yte,
        "deployable": True,
        "oracle": False,
        "history": history,
        "n_soft_train": int(has_soft.sum()),
    }
