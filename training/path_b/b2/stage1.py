"""Stage-1 multi-output glucose emulator: 8× LightGBM (PLAN_B2 primary)."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from training.path_b.b2.data import Stage1Predictions, pred_col_names

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _lgbm_reg(
    params: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
    n_estimators: int,
    device: str,
) -> LGBMRegressor:
    p = dict(params)
    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        device=device,
        verbosity=-1,
        **p,
    )


def _fit_reg(
    model: LGBMRegressor,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    *,
    es_rounds: int,
) -> LGBMRegressor:
    if X_va is not None and y_va is not None and len(y_va) > 0:
        callbacks = [
            lgb.early_stopping(es_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_tr, y_tr)
    return model


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "n": int(mask.sum())}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(mean_squared_error(yt, yp) ** 0.5),
        "n": int(mask.sum()),
    }


def _space(trial: optuna.Trial, cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["hpo"]["lightgbm"]
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", s["learning_rate"][0], s["learning_rate"][1], log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", s["num_leaves"][0], s["num_leaves"][1]),
        "max_depth": trial.suggest_int("max_depth", s["max_depth"][0], s["max_depth"][1]),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", s["min_child_samples"][0], s["min_child_samples"][1]
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain", s["min_split_gain"][0], s["min_split_gain"][1]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", s["colsample_bytree"][0], s["colsample_bytree"][1]
        ),
        "subsample": trial.suggest_float("subsample", s["subsample"][0], s["subsample"][1]),
        "subsample_freq": trial.suggest_int(
            "subsample_freq", s["subsample_freq"][0], s["subsample_freq"][1]
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", s["reg_alpha"][0], s["reg_alpha"][1], log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", s["reg_lambda"][0], s["reg_lambda"][1], log=True
        ),
    }


def tune_one_target(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int,
    seed: int,
    study_name: str,
) -> dict[str, Any]:
    n_est = int(cfg["run"]["stage1_n_estimators"])
    es = int(cfg["run"]["stage1_es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4
    packs: list[dict[str, Any]] = []

    def _fit(params: dict[str, Any], dev: str) -> dict[str, Any]:
        m = _lgbm_reg(params, seed=seed, n_jobs=n_jobs, n_estimators=n_est, device=dev)
        _fit_reg(m, X_tr, y_tr, X_va, y_va, es_rounds=es)
        pred = m.predict(X_va)
        met = _r2_rmse(y_va, pred)
        return {"params": params, "model": m, "val_r2": met["r2"], "val_rmse": met["rmse"], "device": dev}

    def objective(trial: optuna.Trial) -> float:
        params = _space(trial, cfg)
        try:
            pack = _fit(params, device)
        except Exception:
            try:
                pack = _fit(params, "cpu")
            except Exception:
                return -1e9
        packs.append(pack)
        return float(pack["val_r2"]) if np.isfinite(pack["val_r2"]) else -1e9

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # baseline
    base_params = dict(cfg["stage1_default"])
    try:
        base = _fit(base_params, device)
        packs.append(base)
    except Exception:
        base = _fit(base_params, "cpu")
        packs.append(base)

    if not packs:
        raise RuntimeError(f"stage1 {study_name}: no packs")
    best = max(packs, key=lambda p: (p["val_r2"] if np.isfinite(p["val_r2"]) else -1e9))
    return best


def fit_multi_target(
    X_tr: pd.DataFrame,
    Y_tr: pd.DataFrame,
    X_va: pd.DataFrame | None,
    Y_va: pd.DataFrame | None,
    params_per_target: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    *,
    device: str,
    seed: int,
) -> dict[str, LGBMRegressor]:
    n_est = int(cfg["run"]["stage1_n_estimators"])
    es = int(cfg["run"]["stage1_es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4
    models: dict[str, LGBMRegressor] = {}
    for col in Y_tr.columns:
        params = params_per_target[col]
        m = _lgbm_reg(params, seed=seed, n_jobs=n_jobs, n_estimators=n_est, device=device)
        yv = Y_va[col].to_numpy() if Y_va is not None else None
        Xv = X_va if Y_va is not None else None
        try:
            _fit_reg(m, X_tr, Y_tr[col].to_numpy(), Xv, yv, es_rounds=es)
        except Exception as e:
            # keep original error context if CPU also fails
            try:
                m = _lgbm_reg(
                    params, seed=seed, n_jobs=n_jobs, n_estimators=n_est, device="cpu"
                )
                _fit_reg(m, X_tr, Y_tr[col].to_numpy(), Xv, yv, es_rounds=es)
            except Exception as e2:
                raise RuntimeError(
                    f"stage1 fit failed target={col} device={device}: {type(e).__name__}: {e}; "
                    f"cpu fallback: {type(e2).__name__}: {e2}"
                ) from e2
        models[col] = m
    return models


def predict_multi(
    models: dict[str, LGBMRegressor],
    X: pd.DataFrame,
    target_cols: list[str],
) -> np.ndarray:
    cols = []
    for c in target_cols:
        cols.append(models[c].predict(X))
    return np.column_stack(cols)


def metrics_multi(
    Y_true: pd.DataFrame | np.ndarray,
    Y_pred: np.ndarray,
    target_cols: list[str],
) -> dict[str, Any]:
    if isinstance(Y_true, pd.DataFrame):
        yt = Y_true[target_cols].to_numpy(dtype=float)
    else:
        yt = np.asarray(Y_true, dtype=float)
    per = {}
    r2s = []
    for i, c in enumerate(target_cols):
        met = _r2_rmse(yt[:, i], Y_pred[:, i])
        per[c] = met
        if np.isfinite(met["r2"]):
            r2s.append(met["r2"])
    return {
        "per_target": per,
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "n_targets": len(target_cols),
    }


def run_stage1(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> Stage1Predictions:
    """Full Stage-1 under PLAN_B2 OOF + non-aux rules.

    - Tune params on train∩aux → val∩aux (fixed recommended_split).
    - OOF K-fold on train∩aux for train Ŷ (aux holdouts).
    - Non-aux train Ŷ = mean of K fold-model predictions.
    - Final model on full train∩aux → val/test Ŷ for all core pids.
    """
    seed = int(cfg["run"]["seed"])
    k_folds = int(cfg["run"]["oof_folds"])
    n_trials = int(n_trials if n_trials is not None else cfg["run"]["stage1_n_trials"])
    w0 = groups["w0"]
    true_cols = groups["true_cols"]
    glu_targets = groups["glu_targets"]  # unprefixed names matching Y columns conceptually
    # map true_cols order == glu_targets order
    pred_cols = pred_col_names(cfg, glu_targets)

    train = df[df["recommended_split"] == "train"].copy()
    val = df[df["recommended_split"] == "val"].copy()
    test = df[df["recommended_split"] == "test"].copy()

    aux_tr = train[train["aux_eligible"].astype(bool)].copy()
    nonaux_tr = train[~train["aux_eligible"].astype(bool)].copy()
    aux_va = val[val["aux_eligible"].astype(bool)].copy()

    X_tr = aux_tr[w0]
    Y_tr = aux_tr[true_cols].copy()
    Y_tr.columns = glu_targets  # work with short names internally
    X_va = aux_va[w0]
    Y_va = aux_va[true_cols].copy()
    Y_va.columns = glu_targets

    log(f"stage1 tune pool train_aux={len(aux_tr)} val_aux={len(aux_va)} n_trials={n_trials}")

    best_params: dict[str, dict[str, Any]] = {}
    for tcol in glu_targets:
        log(f"  tune target={tcol}")
        pack = tune_one_target(
            X_tr,
            Y_tr[tcol].to_numpy(),
            X_va,
            Y_va[tcol].to_numpy(),
            cfg,
            device=device,
            n_trials=n_trials,
            seed=seed,
            study_name=f"s1_{tcol}",
        )
        best_params[tcol] = dict(pack["params"])
        log(f"    val_r2={pack['val_r2']:.4f} rmse={pack['val_rmse']:.4f}")

    # --- OOF on train∩aux ---
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    y_lab = aux_tr["label"].to_numpy(dtype=np.int64)
    oof = np.full((len(aux_tr), len(glu_targets)), np.nan, dtype=float)
    nonaux_acc = np.zeros((len(nonaux_tr), len(glu_targets)), dtype=float)
    fold_counts: list[dict[str, Any]] = []

    X_non = nonaux_tr[w0] if len(nonaux_tr) else None
    aux_index = aux_tr.index.to_numpy()

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(aux_tr)), y_lab)):
        y_hold = y_lab[va_idx]
        counts = {int(k): int(v) for k, v in zip(*np.unique(y_hold, return_counts=True))}
        fold_counts.append({"fold": fold_i, "n": int(len(va_idx)), "label_counts": counts})
        log(f"  oof fold={fold_i} n_hold={len(va_idx)} labels={counts}")

        X_ftr = X_tr.iloc[tr_idx]
        Y_ftr = Y_tr.iloc[tr_idx]
        X_fva = X_tr.iloc[va_idx]
        models = fit_multi_target(
            X_ftr,
            Y_ftr,
            None,
            None,
            best_params,
            cfg,
            device=device,
            seed=seed + fold_i,
        )
        oof[va_idx] = predict_multi(models, X_fva, glu_targets)
        if X_non is not None and len(nonaux_tr):
            nonaux_acc += predict_multi(models, X_non, glu_targets)

    if np.isnan(oof).any():
        raise AssertionError("OOF has NaNs — stratified fold coverage failed")

    if len(nonaux_tr):
        nonaux_pred = nonaux_acc / float(k_folds)
    else:
        nonaux_pred = np.zeros((0, len(glu_targets)))

    # train Ŷ frame (aux OOF + non-aux mean-K)
    yhat_tr_parts = []
    aux_yhat = pd.DataFrame(oof, columns=pred_cols)
    aux_yhat.insert(0, "person_id", aux_tr["person_id"].to_numpy())
    yhat_tr_parts.append(aux_yhat)
    if len(nonaux_tr):
        na = pd.DataFrame(nonaux_pred, columns=pred_cols)
        na.insert(0, "person_id", nonaux_tr["person_id"].to_numpy())
        yhat_tr_parts.append(na)
    yhat_train = pd.concat(yhat_tr_parts, axis=0, ignore_index=True)

    # --- final model: full train∩aux, early-stop on val∩aux ---
    final_models = fit_multi_target(
        X_tr,
        Y_tr,
        X_va,
        Y_va,
        best_params,
        cfg,
        device=device,
        seed=seed,
    )

    def _predict_split(split_df: pd.DataFrame) -> pd.DataFrame:
        pred = predict_multi(final_models, split_df[w0], glu_targets)
        out = pd.DataFrame(pred, columns=pred_cols)
        out.insert(0, "person_id", split_df["person_id"].to_numpy())
        return out

    yhat_val = _predict_split(val)
    yhat_test = _predict_split(test)

    # metrics on aux val/test only (true CGM available); align by person_id
    va_m = yhat_val.merge(
        val.loc[val["aux_eligible"].astype(bool), ["person_id"] + true_cols],
        on="person_id",
        how="inner",
    )
    te_m = yhat_test.merge(
        test.loc[test["aux_eligible"].astype(bool), ["person_id"] + true_cols],
        on="person_id",
        how="inner",
    )
    Yva_true = va_m[true_cols].copy()
    Yva_true.columns = glu_targets
    Yte_true = te_m[true_cols].copy()
    Yte_true.columns = glu_targets
    s1_val = metrics_multi(Yva_true, va_m[pred_cols].to_numpy(), glu_targets)
    s1_test = metrics_multi(Yte_true, te_m[pred_cols].to_numpy(), glu_targets)
    log(f"stage1 val mean_r2={s1_val['mean_r2']:.4f} test mean_r2={s1_test['mean_r2']:.4f}")

    # gate targets
    gate_targets = list(cfg["decision_bars"]["stage1_r2_targets"])
    gate = {t: s1_val["per_target"].get(t, {}).get("r2", float("nan")) for t in gate_targets}
    s1_val["gate_targets_r2"] = gate
    s1_val["gate_pass"] = bool(
        all(np.isfinite(v) and v > 0 for v in gate.values())
    )
    log(f"stage1 smoke gate (val R²>0 on {gate_targets}): {s1_val['gate_pass']} {gate}")

    return Stage1Predictions(
        yhat_train=yhat_train,
        yhat_val=yhat_val,
        yhat_test=yhat_test,
        fold_label_counts=fold_counts,
        stage1_val_metrics=s1_val,
        stage1_test_metrics=s1_test,
        best_params_per_target=best_params,
    )
