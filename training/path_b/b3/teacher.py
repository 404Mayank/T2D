"""Privileged teacher: Path A family on C1+true CGM; OOF soft labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.hpo import pick_family, tune_catboost, tune_lightgbm
from training.path_a_watch.metrics import full_report, macro_ovr_auc, multiclass_brier
from training.path_a_watch.models import predict_proba, resolve_lgbm_device
from training.path_b.b3.data import assert_teacher_features, soft_label_diagnostics, split_xy


def _ece_multiclass(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Simple confidence ECE (max-prob bins)."""
    y = np.asarray(y, dtype=np.int64).ravel()
    p = np.asarray(proba, dtype=float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def train_hard_gbm(
    splits: dict[str, Any],
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    log=lambda msg: None,
    families: str = "both",
) -> dict[str, Any]:
    """Tune LGBM (+ optional CatBoost); return selected pack + metrics."""
    if n_trials is not None:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = int(n_trials)

    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xva, yva = splits["X_val"], splits["y_val"]
    Xte, yte = splits["X_test"], splits["y_test"]

    packs: list[dict[str, Any]] = []
    lgbm_pack = tune_lightgbm(Xtr, ytr, Xva, yva, cfg, device=device)
    packs.append(lgbm_pack)
    log(
        f"  LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f} "
        f"auprc={lgbm_pack['val_macro_auprc']:.4f}"
    )
    cat_pack = None
    if families == "both":
        cat_pack = tune_catboost(Xtr, ytr, Xva, yva, cfg)
        packs.append(cat_pack)
        log(
            f"  Cat  val_auc={cat_pack['val_macro_ovr_auc']:.4f} "
            f"auprc={cat_pack['val_macro_auprc']:.4f}"
        )

    selected = pick_family(packs, eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"  SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    model = selected["model"]
    proba_val = predict_proba(model, Xva)
    proba_test = predict_proba(model, Xte)
    proba_train = predict_proba(model, Xtr)

    cal = fit_calibrators(
        proba_val,
        yva,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test_cal = cal["primary"].transform(proba_test)

    raw_val = full_report(yva, proba_val, tag="val_raw")
    raw_test = full_report(yte, proba_test, tag="test_raw")
    cal_test = full_report(yte, proba_test_cal, tag="test_cal_primary")

    return {
        "family": selected["family"],
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "source": selected.get("source"),
        "val_macro_ovr_auc": float(selected["val_macro_ovr_auc"]),
        "val_macro_auprc": float(selected["val_macro_auprc"]),
        "model": model,
        "lgbm_pack": lgbm_pack,
        "cat_pack": cat_pack,
        "lgbm_params": lgbm_pack["params"],
        "proba_train": proba_train,
        "proba_val": proba_val,
        "proba_test": proba_test,
        "proba_test_cal": proba_test_cal,
        "calibrator": cal,
        "metrics": {
            "val_raw": raw_val,
            "test_raw": raw_test,
            "test_cal": cal_test,
            "teacher_val_brier": float(multiclass_brier(yva, proba_val)),
            "teacher_val_ece": _ece_multiclass(yva, proba_val),
        },
        "feature_cols": list(splits["feature_cols"]),
        "pool": splits["pool"],
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_test": splits["n_test"],
        "pid_test": splits["pid_test"],
        "y_test": yte,
        "pid_val": splits["pid_val"],
        "y_val": yva,
        "pid_train": splits["pid_train"],
        "y_train": ytr,
    }


def fit_lgbm_fixed(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: str | None = None,
    class_weight: str | dict | None = "balanced",
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit LightGBM with fixed hyperparams (G0 pin / OOF fold teachers)."""
    from training.path_a_watch.models import best_iteration, fit_lgbm, make_lgbm

    device = device or resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    n_estimators = int(cfg["run"]["n_estimators_max"])
    es_rounds = int(cfg["run"]["es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    seed = int(cfg["run"]["seed"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4

    model = make_lgbm(
        params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        n_estimators=n_estimators,
        class_weight=class_weight,
    )
    # sample_weight support: monkey via fit kwargs
    import lightgbm as lgb

    callbacks = [
        lgb.early_stopping(es_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "multi_logloss",
        "callbacks": callbacks,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    proba_val = predict_proba(model, X_val)
    return {
        "family": "lightgbm",
        "params": dict(params),
        "best_iteration": best_iteration(model),
        "val_macro_ovr_auc": float(macro_ovr_auc(y_val, proba_val)),
        "model": model,
        "device": device,
    }


def build_teacher_and_oof(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: dict[str, Any],
    *,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> dict[str, Any]:
    """Full teacher (Tch) on aux + OOF soft labels for train∩aux + D1a metrics."""
    c1 = list(groups["c1"])
    true_cols = list(groups["true_cols"])
    teacher_feats = c1 + true_cols
    assert_teacher_features(teacher_feats, true_cols, cfg)

    # --- Tch / D1a on aux pool ---
    splits_tch = split_xy(df, teacher_feats, pool="aux")
    log(
        f"teacher Tch n_train={splits_tch['n_train']} "
        f"n_val={splits_tch['n_val']} n_test={splits_tch['n_test']}"
    )
    tch = train_hard_gbm(splits_tch, cfg, n_trials=n_trials, log=log, families="both")
    tch["arm"] = "Tch"
    tch["deployable"] = False
    tch["oracle"] = True

    splits_d1a = split_xy(df, c1, pool="aux")
    log(f"teacher D1a (matched) n_train={splits_d1a['n_train']}")
    d1a = train_hard_gbm(splits_d1a, cfg, n_trials=n_trials, log=log, families="both")
    d1a["arm"] = "D1a"
    d1a["deployable"] = True
    d1a["oracle"] = False

    # --- OOF soft labels on train ∩ aux ---
    # Prefer selected teacher family params when LGBM; else use LGBM pack params
    # for OOF (student soft labels need stable proba; LGBM always available).
    oof_params = dict(tch["lgbm_params"])
    device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    k = int(cfg["run"]["oof_folds"])
    seed = int(cfg["run"]["seed"])

    tr = df[(df["recommended_split"] == "train") & df["aux_eligible"].astype(bool)].copy()
    X_all = tr[teacher_feats]
    y_all = tr["label"].to_numpy(dtype=np.int64)
    pid_all = tr["person_id"].to_numpy()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    soft_by_pid: dict[int, np.ndarray] = {}
    fold_label_counts: list[dict[str, Any]] = []
    fold_val_aucs: list[float] = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        y_va_fold = y_all[va_idx]
        counts = {int(c): int((y_va_fold == c).sum()) for c in range(4)}
        fold_label_counts.append({"fold": fold_i, "label_counts": counts})
        if counts.get(3, 0) == 0:
            log(f"WARNING OOF fold {fold_i}: zero insulin in holdout")

        Xtr_f = X_all.iloc[tr_idx]
        ytr_f = y_all[tr_idx]
        Xva_f = X_all.iloc[va_idx]
        yva_f = y_all[va_idx]
        pid_va = pid_all[va_idx]

        pack = fit_lgbm_fixed(
            Xtr_f,
            ytr_f,
            Xva_f,
            yva_f,
            oof_params,
            cfg,
            device=device,
            class_weight=cfg["class_weights"]["lightgbm"],
        )
        proba = predict_proba(pack["model"], Xva_f)
        auc = float(macro_ovr_auc(yva_f, proba))
        fold_val_aucs.append(auc)
        log(f"  OOF fold {fold_i}: holdout_auc={auc:.4f} counts={counts}")
        for j, p_i in enumerate(pid_va):
            soft_by_pid[int(p_i)] = proba[j].astype(np.float64)

    if len(soft_by_pid) != len(tr):
        raise AssertionError(
            f"OOF soft labels {len(soft_by_pid)} != aux-train {len(tr)}"
        )

    oof_mean = float(np.mean(fold_val_aucs)) if fold_val_aucs else float("nan")
    oof_std = float(np.std(fold_val_aucs)) if fold_val_aucs else float("nan")
    d1a_val = float(d1a["val_macro_ovr_auc"])
    margin = float(cfg["decision_bars"]["oof_teacher_margin"])
    oof_gate = bool(oof_mean > d1a_val + margin)

    temp = float(cfg["run"]["temperature"])
    soft_diag = soft_label_diagnostics(soft_by_pid, temperature=temp)

    log(
        f"OOF teacher mean_val_auc={oof_mean:.4f}±{oof_std:.4f} "
        f"D1a_val={d1a_val:.4f} gate_pass={oof_gate}"
    )

    return {
        "tch": tch,
        "d1a": d1a,
        "soft_by_pid": soft_by_pid,
        "oof": {
            "fold_label_counts": fold_label_counts,
            "fold_val_aucs": fold_val_aucs,
            "mean_val_auc": oof_mean,
            "std_val_auc": oof_std,
            "d1a_val_auc": d1a_val,
            "gate_margin": margin,
            "gate_pass": oof_gate,
            "n_soft_pids": len(soft_by_pid),
            "oof_params": oof_params,
        },
        "soft_diagnostics": soft_diag,
        "teacher_feats": teacher_feats,
        "c1_feats": c1,
    }
