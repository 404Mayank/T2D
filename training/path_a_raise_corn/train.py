"""Train / HPO loop for CORN, CE, CORAL tabular MLPs."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.path_a_raise_corn.losses_proba import (
    balanced_person_weights,
    corn_conditional_to_proba,
    corn_hard_label_from_logits,
    corn_loss_unweighted,
    corn_loss_weighted,
    coral_levels_to_proba,
)
from training.path_a_raise_corn.model import build_model
from training.path_a_watch.metrics import macro_auprc, macro_ovr_auc

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _device(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba_arm(
    model: torch.nn.Module,
    X: np.ndarray,
    arm: str,
    device: torch.device,
    bs: int = 256,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    arm = arm.lower()
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=device)
        logits = model(xb)
        if arm == "corn":
            p = corn_conditional_to_proba(logits)
        elif arm == "coral":
            p = coral_levels_to_proba(logits)
        elif arm in ("ce", "ce_mlp", "softmax"):
            p = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(arm)
        outs.append(p.detach().cpu().numpy())
    proba = np.concatenate(outs, axis=0)
    # contract: column 0 is P0
    if proba.shape[1] != 4:
        raise AssertionError(f"proba shape {proba.shape}")
    s = proba.sum(axis=1)
    if not np.allclose(s, 1.0, atol=1e-4):
        proba = proba / s[:, None]
    return proba.astype(np.float64)


@torch.no_grad()
def predict_hard_arm(
    model: torch.nn.Module,
    X: np.ndarray,
    arm: str,
    device: torch.device,
    bs: int = 256,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    arm = arm.lower()
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i : i + bs], dtype=torch.float32, device=device)
        logits = model(xb)
        if arm == "corn":
            lab = corn_hard_label_from_logits(logits)
        elif arm in ("ce", "ce_mlp", "softmax"):
            lab = logits.argmax(dim=-1)
        else:
            # coral: threshold survival
            u = torch.sigmoid(logits)
            lab = torch.sum(u > 0.5, dim=1)
        outs.append(lab.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.int64)


def _batch_loss(
    logits: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor | None,
    arm: str,
    n_classes: int,
    class_w: torch.Tensor | None,
    imbalance: str,
) -> torch.Tensor:
    arm = arm.lower()
    if arm == "corn":
        if imbalance == "weighted_corn" and wb is not None:
            return corn_loss_weighted(logits, yb, n_classes, wb)
        return corn_loss_unweighted(logits, yb, n_classes)
    if arm == "coral":
        # levels: for label k, first k entries are 1
        # build levels on the fly
        levels = torch.zeros(len(yb), n_classes - 1, device=logits.device, dtype=logits.dtype)
        for i, yi in enumerate(yb.tolist()):
            if yi > 0:
                levels[i, : int(yi)] = 1.0
        # unweighted coral-style BCE
        log_sig = F.logsigmoid(logits)
        term = log_sig * levels + (log_sig - logits) * (1.0 - levels)
        per = -term.sum(dim=1)
        if wb is not None and imbalance == "weighted_corn":
            return (per * wb).mean()
        return per.mean()
    # CE
    return F.cross_entropy(logits, yb, weight=class_w)


def train_one(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    arm: str,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta_auc: float,
    imbalance: str,
    n_classes: int = 4,
    seed: int = 42,
    device: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    log = log or (lambda _m: None)
    dev = _device(device)
    if dev.type == "cuda":
        try:
            torch.backends.cudnn.enabled = False
        except Exception:
            pass

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(arm, d_in=Z_train.shape[1], n_classes=n_classes, hidden=hidden, dropout=dropout)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    w_np = balanced_person_weights(y_train, n_classes=n_classes)
    class_w_np = np.zeros(n_classes, dtype=np.float64)
    for k in range(n_classes):
        m = y_train == k
        if m.any():
            class_w_np[k] = float(w_np[m][0])
    class_w = torch.tensor(class_w_np, dtype=torch.float32, device=dev)

    ds = TensorDataset(
        torch.tensor(Z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(w_np, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_auc = -1.0
    best_auprc = -1.0
    best_ep = -1
    bad = 0
    history: list[dict[str, float]] = []

    for ep in range(max_epochs):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, wb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            wb = wb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = _batch_loss(logits, yb, wb, arm, n_classes, class_w, imbalance)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)

        proba_va = predict_proba_arm(model, Z_val, arm, dev)
        auc = float(macro_ovr_auc(y_val, proba_va))
        auprc = float(macro_auprc(y_val, proba_va))
        history.append({"epoch": ep, "loss": total / max(n, 1), "val_auc": auc, "val_auprc": auprc})

        improved = auc > best_auc + min_delta_auc or (
            abs(auc - best_auc) <= min_delta_auc and auprc > best_auprc + 1e-6
        )
        if improved:
            best_auc = auc
            best_auprc = auprc
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev)

    proba_val = predict_proba_arm(model, Z_val, arm, dev)
    hard_val = predict_hard_arm(model, Z_val, arm, dev)

    return {
        "arm": arm,
        "family": f"mlp_{arm}",
        "model": model.cpu(),
        "device": str(dev),
        "params": {
            "hidden": int(hidden),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "imbalance": imbalance,
        },
        "best_epoch": int(best_ep),
        "epochs_ran": int(len(history)),
        "val_macro_ovr_auc": float(macro_ovr_auc(y_val, proba_val)),
        "val_macro_auprc": float(macro_auprc(y_val, proba_val)),
        "proba_val": proba_val,
        "hard_val": hard_val,
        "history": history,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }


def hpo_arm(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    arm: str,
    n_trials: int,
    seed: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    log = log or (lambda _m: None)
    seed = int(seed if seed is not None else cfg["run"]["seed"])
    hpo = cfg["hpo"]
    run = cfg["run"]

    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_categorical("hidden", list(hpo["hidden"]))
        dropout = trial.suggest_categorical("dropout", list(hpo["dropout"]))
        batch_size = trial.suggest_categorical("batch_size", list(hpo["batch_size"]))
        lr = trial.suggest_float("lr", float(hpo["lr_log_low"]), float(hpo["lr_log_high"]), log=True)
        wd = trial.suggest_float(
            "weight_decay",
            float(hpo["weight_decay_log_low"]),
            float(hpo["weight_decay_log_high"]),
            log=True,
        )
        pack = train_one(
            Z_train,
            y_train,
            Z_val,
            y_val,
            arm=arm,
            hidden=int(hidden),
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(wd),
            batch_size=int(batch_size),
            max_epochs=int(run["max_epochs"]),
            patience=int(run["patience"]),
            min_delta_auc=float(run["min_delta_auc"]),
            imbalance=str(run["imbalance"]),
            n_classes=int(cfg["data"]["n_classes"]),
            seed=seed + trial.number,  # vary init slightly across trials; selection still val-based
            device=run.get("device"),
        )
        trial.set_user_attr("val_macro_auprc", pack["val_macro_auprc"])
        trial.set_user_attr("params_full", pack["params"])
        trial.set_user_attr("best_epoch", pack["best_epoch"])
        # store pack lightly? keep only metrics; retrain winner below
        return float(pack["val_macro_ovr_auc"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    # select: max AUC, tie-break AUPRC within eps
    eps = float(run["auc_tie_eps"])
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        raise RuntimeError("no complete HPO trials")
    best_auc = max(t.value for t in trials if t.value is not None)
    pool = [t for t in trials if t.value is not None and t.value >= best_auc - eps]
    winner = max(pool, key=lambda t: float(t.user_attrs.get("val_macro_auprc", -1.0)))

    p = winner.params
    log(
        f"HPO {arm}: best_trial={winner.number} val_auc={winner.value:.4f} "
        f"auprc={winner.user_attrs.get('val_macro_auprc', float('nan')):.4f} params={p}"
    )

    # retrain winner with seed=run seed for reproducibility of freeze artifact
    pack = train_one(
        Z_train,
        y_train,
        Z_val,
        y_val,
        arm=arm,
        hidden=int(p["hidden"]),
        dropout=float(p["dropout"]),
        lr=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        batch_size=int(p["batch_size"]),
        max_epochs=int(run["max_epochs"]),
        patience=int(run["patience"]),
        min_delta_auc=float(run["min_delta_auc"]),
        imbalance=str(run["imbalance"]),
        n_classes=int(cfg["data"]["n_classes"]),
        seed=seed,
        device=run.get("device"),
        log=log,
    )
    pack["trial_number"] = int(winner.number)
    pack["hpo_best_value"] = float(winner.value) if winner.value is not None else None
    pack["n_trials"] = int(n_trials)
    pack["study_best_trials"] = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "val_macro_auprc": t.user_attrs.get("val_macro_auprc"),
        }
        for t in sorted(trials, key=lambda x: -(x.value or -1))[:5]
    ]
    return pack


def train_fixed(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    *,
    arm: str,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Smoke / fixed-hparams path (no Optuna)."""
    run = cfg["run"]
    return train_one(
        Z_train,
        y_train,
        Z_val,
        y_val,
        arm=arm,
        hidden=int(run["mlp_hidden_default"]),
        dropout=float(run["mlp_dropout_default"]),
        lr=float(run["mlp_lr_default"]),
        weight_decay=float(run["mlp_weight_decay_default"]),
        batch_size=int(run["batch_size"]),
        max_epochs=int(run.get("smoke_max_epochs", run["max_epochs"])),
        patience=int(run.get("smoke_patience", run["patience"])),
        min_delta_auc=float(run["min_delta_auc"]),
        imbalance=str(run["imbalance"]),
        n_classes=int(cfg["data"]["n_classes"]),
        seed=int(run["seed"]),
        device=run.get("device"),
        log=log,
    )
