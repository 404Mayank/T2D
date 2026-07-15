"""Stage-1 daily quantile LGBM emulator + person aggregation (PLAN_B2_V2)."""

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

from training.path_b.b2v2.data import (
    Stage1Predictions,
    aggregate_day_preds_to_person,
    apply_impute,
    fit_impute_medians,
    handoff_col_names,
    short_targets,
    short_to_person,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# quantile key → column prefix on day predictions
Q_MID = 0.5
Q_LO = 0.1
Q_HI = 0.9


def _lgbm_regressor(
    params: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
    n_estimators: int,
    device: str,
    alpha: float | None = None,
) -> LGBMRegressor:
    """Mid uses MSE regression (better R²); tails use quantile pinball."""
    p = dict(params)
    if alpha is None:
        return LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=n_jobs,
            device=device,
            verbosity=-1,
            **p,
        )
    return LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
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
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    else:
        model.fit(X_tr, y_tr)
    return model


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


def _person_agg_r2_from_days(
    day_hold: pd.DataFrame,
    mid_pred: np.ndarray,
    *,
    y_col: str,
) -> float:
    """Person-mean of day mid vs person-mean of true y (holdout supervised days)."""
    tmp = day_hold[["person_id", y_col]].copy()
    tmp["p"] = mid_pred
    g = tmp.groupby("person_id", sort=False).agg(y=(y_col, "mean"), p=("p", "mean"))
    if len(g) < 2:
        return float("nan")
    return float(r2_score(g["y"].to_numpy(), g["p"].to_numpy()))


def tune_mid_target(
    day_tr: pd.DataFrame,
    day_va: pd.DataFrame,
    x_cols: list[str],
    short: str,
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """HPO maximizing val **person-agg** R² on α=0.5 quantile."""
    y_col = f"y_{short}"
    n_est = int(cfg["run"]["stage1_n_estimators"])
    es = int(cfg["run"]["stage1_es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4
    packs: list[dict[str, Any]] = []

    X_tr = day_tr[x_cols]
    y_tr = day_tr[y_col].to_numpy(dtype=float)
    X_va = day_va[x_cols]
    y_va = day_va[y_col].to_numpy(dtype=float)

    def _fit(params: dict[str, Any], dev: str) -> dict[str, Any]:
        # Mid = MSE regression (quantile α=0.5 underperformed R² on this data)
        m = _lgbm_regressor(
            params, seed=seed, n_jobs=n_jobs, n_estimators=n_est, device=dev, alpha=None
        )
        _fit_reg(m, X_tr, y_tr, X_va, y_va, es_rounds=es)
        pred = m.predict(X_va)
        person_r2 = _person_agg_r2_from_days(day_va, pred, y_col=y_col)
        day_met = _r2_rmse(y_va, pred)
        return {
            "params": params,
            "model": m,
            "val_person_r2": person_r2,
            "val_day_r2": day_met["r2"],
            "val_day_rmse": day_met["rmse"],
            "device": dev,
        }

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
        v = pack["val_person_r2"]
        return float(v) if np.isfinite(v) else -1e9

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name=f"s1_mid_{short}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    base_params = dict(cfg["stage1_default"])
    try:
        packs.append(_fit(base_params, device))
    except Exception:
        packs.append(_fit(base_params, "cpu"))

    if not packs:
        raise RuntimeError(f"stage1 mid {short}: no packs")
    best = max(
        packs,
        key=lambda p: (
            p["val_person_r2"] if np.isfinite(p["val_person_r2"]) else -1e9
        ),
    )
    return best


def fit_quantile_heads(
    day_tr: pd.DataFrame,
    day_va: pd.DataFrame | None,
    x_cols: list[str],
    short: str,
    params: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: str,
    seed: int,
    alphas: list[float],
) -> dict[float, LGBMRegressor]:
    y_col = f"y_{short}"
    n_est = int(cfg["run"]["stage1_n_estimators"])
    es = int(cfg["run"]["stage1_es_rounds"])
    n_jobs = int(cfg["run"]["n_jobs"])
    if device == "gpu" and n_jobs == -1:
        n_jobs = 4
    X_tr = day_tr[x_cols]
    y_tr = day_tr[y_col].to_numpy(dtype=float)
    X_va = day_va[x_cols] if day_va is not None else None
    y_va = day_va[y_col].to_numpy(dtype=float) if day_va is not None else None
    models: dict[float, LGBMRegressor] = {}
    for a in alphas:
        # mid (0.5): MSE regression; lo/hi: quantile
        use_alpha = None if abs(float(a) - Q_MID) < 1e-9 else float(a)
        try:
            m = _lgbm_regressor(
                params,
                seed=seed,
                n_jobs=n_jobs,
                n_estimators=n_est,
                device=device,
                alpha=use_alpha,
            )
            _fit_reg(m, X_tr, y_tr, X_va, y_va, es_rounds=es)
        except Exception as e:
            try:
                m = _lgbm_regressor(
                    params,
                    seed=seed,
                    n_jobs=n_jobs,
                    n_estimators=n_est,
                    device="cpu",
                    alpha=use_alpha,
                )
                _fit_reg(m, X_tr, y_tr, X_va, y_va, es_rounds=es)
            except Exception as e2:
                raise RuntimeError(
                    f"stage1 fit failed short={short} alpha={a}: {type(e).__name__}: {e}; "
                    f"cpu: {type(e2).__name__}: {e2}"
                ) from e2
        models[float(a)] = m
    return models


def predict_day_quantiles(
    models_by_short: dict[str, dict[float, LGBMRegressor]],
    day_df: pd.DataFrame,
    x_cols: list[str],
    shorts: list[str],
) -> pd.DataFrame:
    """Return person_id + mid/lo/hi per short for each day row (same index as day_df)."""
    out = pd.DataFrame({"person_id": day_df["person_id"].to_numpy()})
    X = day_df[x_cols]
    for s in shorts:
        mids = models_by_short[s]
        out[f"mid_{s}"] = mids[Q_MID].predict(X)
        out[f"lo_{s}"] = mids[Q_LO].predict(X)
        out[f"hi_{s}"] = mids[Q_HI].predict(X)
        # enforce lo <= mid <= hi softly via sort (quantile crossing fix)
        lo = out[f"lo_{s}"].to_numpy()
        mid = out[f"mid_{s}"].to_numpy()
        hi = out[f"hi_{s}"].to_numpy()
        stacked = np.column_stack([lo, mid, hi])
        stacked.sort(axis=1)
        out[f"lo_{s}"] = stacked[:, 0]
        out[f"mid_{s}"] = stacked[:, 1]
        out[f"hi_{s}"] = stacked[:, 2]
    return out


def _metrics_person(
    yhat_person: pd.DataFrame,
    person_true: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """yhat_person has yhat_{s}_mid; person_true has ytrue_ person cols + person_id."""
    shorts = short_targets(cfg)
    s2p = short_to_person(cfg)
    tpfx = cfg["data"]["true_prefix"]
    pfx = cfg["data"]["pred_prefix"]
    m = yhat_person.merge(person_true, on="person_id", how="inner")
    per: dict[str, Any] = {}
    r2s = []
    coverages = []
    for s in shorts:
        yt = m[f"{tpfx}{s2p[s]}"].to_numpy(dtype=float)
        yp = m[f"{pfx}{s}_mid"].to_numpy(dtype=float)
        met = _r2_rmse(yt, yp)
        per[s] = met
        if np.isfinite(met["r2"]):
            r2s.append(met["r2"])
        # coverage: true in [lo, hi] after person-agg of day lo/hi
        # we store person-level lo/hi as mean of day lo/hi via spread reconstruction:
        # person mid ± isn't stored; recompute from spread? Plan: person lo/hi = mean day lo/hi
        if f"{pfx}{s}_spread" in m.columns:
            # approximate: coverage using mid ± spread/2 is wrong; need explicit person lo/hi
            pass
    return {
        "per_target": per,
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "n_targets": len(shorts),
        "n_persons": int(len(m)),
    }


def _person_lo_hi_from_day(
    day_pred: pd.DataFrame,
    shorts: list[str],
) -> pd.DataFrame:
    """Mean of day lo/hi → person lo/hi for coverage."""
    rows = []
    for pid, g in day_pred.groupby("person_id", sort=False):
        rec: dict[str, Any] = {"person_id": pid}
        for s in shorts:
            rec[f"lo_{s}"] = float(np.nanmean(g[f"lo_{s}"].to_numpy(dtype=float)))
            rec[f"hi_{s}"] = float(np.nanmean(g[f"hi_{s}"].to_numpy(dtype=float)))
            rec[f"mid_{s}"] = float(np.nanmean(g[f"mid_{s}"].to_numpy(dtype=float)))
        rows.append(rec)
    return pd.DataFrame(rows)


def _coverage_table(
    person_bounds: pd.DataFrame,
    person_true: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    shorts = short_targets(cfg)
    s2p = short_to_person(cfg)
    tpfx = cfg["data"]["true_prefix"]
    m = person_bounds.merge(person_true, on="person_id", how="inner")
    per = {}
    ok_count = 0
    bars = cfg["decision_bars"]
    lo_b, hi_b = float(bars["coverage_lo"]), float(bars["coverage_hi"])
    for s in shorts:
        yt = m[f"{tpfx}{s2p[s]}"].to_numpy(dtype=float)
        lo = m[f"lo_{s}"].to_numpy(dtype=float)
        hi = m[f"hi_{s}"].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
        if mask.sum() == 0:
            cov = float("nan")
        else:
            cov = float(np.mean((yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])))
        per[s] = cov
        if np.isfinite(cov) and lo_b <= cov <= hi_b:
            ok_count += 1
    return {
        "per_target": per,
        "n_targets_in_band": int(ok_count),
        "min_required": int(bars["coverage_min_targets"]),
        "band": [lo_b, hi_b],
        "pass": ok_count >= int(bars["coverage_min_targets"]),
    }


def _n_days_buckets(yhat: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    if "n_days_agg" not in yhat.columns:
        return {}
    pfx = cfg["data"]["pred_prefix"]
    shorts = short_targets(cfg)
    out: dict[str, Any] = {}
    n = yhat["n_days_agg"].to_numpy()
    for label, mask in (
        ("n1", n == 1),
        ("n2_7", (n >= 2) & (n <= 7)),
        ("n8p", n >= 8),
    ):
        sub = yhat.loc[mask]
        row: dict[str, Any] = {"n_persons": int(len(sub))}
        if len(sub) == 0:
            out[label] = row
            continue
        for s in shorts:
            for suf in ("spread", "daysd"):
                c = f"{pfx}{s}_{suf}"
                if c in sub.columns:
                    row[f"{s}_{suf}_p50"] = float(sub[c].median())
        out[label] = row
    return out


def run_stage1(
    person_df: pd.DataFrame,
    day_df: pd.DataFrame,
    x_cols: list[str],
    groups: dict[str, list[str]],
    cfg: dict[str, Any],
    *,
    device: str,
    n_trials: int | None = None,
    log=lambda msg: None,
) -> Stage1Predictions:
    """Daily quantile Stage-1 under PLAN_B2_V2 OOF + non-aux rules."""
    seed = int(cfg["run"]["seed"])
    k_folds = int(cfg["run"]["oof_folds"])
    n_trials = int(n_trials if n_trials is not None else cfg["run"]["stage1_n_trials"])
    shorts = short_targets(cfg)
    alphas = [float(a) for a in cfg["run"]["quantiles"]]
    assert Q_LO in alphas and Q_MID in alphas and Q_HI in alphas
    pfx = cfg["data"]["pred_prefix"]
    var_cols = handoff_col_names(cfg, "var")

    # --- impute: train∩aux supervised days ---
    train_sup = (
        (day_df["recommended_split"] == "train")
        & day_df["supervised"]
    )
    if not train_sup.any():
        raise AssertionError("no train supervised days for impute")
    meds = fit_impute_medians(day_df, x_cols, train_mask=train_sup)
    day = apply_impute(day_df, x_cols, meds)

    # person splits
    aux_tr_persons = person_df[
        (person_df["recommended_split"] == "train")
        & person_df["aux_eligible"].astype(bool)
    ]["person_id"].to_numpy()
    nonaux_tr_persons = person_df[
        (person_df["recommended_split"] == "train")
        & ~person_df["aux_eligible"].astype(bool)
    ]["person_id"].to_numpy()
    aux_va_persons = person_df[
        (person_df["recommended_split"] == "val")
        & person_df["aux_eligible"].astype(bool)
    ]["person_id"].to_numpy()

    day_tr_sup = day[
        day["person_id"].isin(set(aux_tr_persons)) & day["supervised"]
    ].copy()
    day_va_sup = day[
        day["person_id"].isin(set(aux_va_persons)) & day["supervised"]
    ].copy()
    if len(day_tr_sup) < 50 or len(day_va_sup) < 10:
        raise AssertionError(
            f"insufficient supervised days train={len(day_tr_sup)} val={len(day_va_sup)}"
        )

    log(
        f"stage1 days train_sup={len(day_tr_sup)} val_sup={len(day_va_sup)} "
        f"n_x={len(x_cols)} n_trials={n_trials} green_fuse={cfg['run'].get('green_fuse')}"
    )

    # --- HPO mid per target ---
    best_params: dict[str, dict[str, Any]] = {}
    for s in shorts:
        log(f"  tune mid target={s}")
        pack = tune_mid_target(
            day_tr_sup,
            day_va_sup,
            x_cols,
            s,
            cfg,
            device=device,
            n_trials=n_trials,
            seed=seed,
        )
        best_params[s] = dict(pack["params"])
        log(
            f"    val_person_r2={pack['val_person_r2']:.4f} "
            f"day_r2={pack['val_day_r2']:.4f}"
        )

    # --- OOF on train∩aux persons (stratified by label) ---
    aux_tr_df = person_df[
        (person_df["recommended_split"] == "train")
        & person_df["aux_eligible"].astype(bool)
    ].copy()
    y_lab = aux_tr_df["label"].to_numpy(dtype=np.int64)
    pids_aux = aux_tr_df["person_id"].to_numpy()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # collect day preds for OOF aux; non-aux = mean of K *person-agg* packs (plan §3.4)
    oof_day_parts: list[pd.DataFrame] = []
    nonaux_day = day[
        day["person_id"].isin(set(nonaux_tr_persons)) & day["infer_ok"]
    ].copy()
    nonaux_person_fold_parts: list[pd.DataFrame] = []

    fold_counts: list[dict[str, Any]] = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(pids_aux)), y_lab)):
        hold_pids = set(pids_aux[va_idx].tolist())
        fit_pids = set(pids_aux[tr_idx].tolist())
        y_hold = y_lab[va_idx]
        counts = {int(k): int(v) for k, v in zip(*np.unique(y_hold, return_counts=True))}
        fold_counts.append({"fold": fold_i, "n": int(len(va_idx)), "label_counts": counts})
        log(f"  oof fold={fold_i} n_hold_persons={len(va_idx)} labels={counts}")

        d_fit = day[day["person_id"].isin(fit_pids) & day["supervised"]]
        d_hold = day[day["person_id"].isin(hold_pids) & day["infer_ok"]]
        if len(d_fit) == 0:
            raise AssertionError(f"fold {fold_i}: empty fit days")

        models_by_short: dict[str, dict[float, LGBMRegressor]] = {}
        for s in shorts:
            models_by_short[s] = fit_quantile_heads(
                d_fit,
                None,
                x_cols,
                s,
                best_params[s],
                cfg,
                device=device,
                seed=seed + fold_i,
                alphas=alphas,
            )
        if len(d_hold):
            pred_hold = predict_day_quantiles(models_by_short, d_hold, x_cols, shorts)
            oof_day_parts.append(pred_hold)
        if len(nonaux_day):
            # person-agg *per fold*, then mean across K (preserves daysd/spread scale)
            pred_na = predict_day_quantiles(models_by_short, nonaux_day, x_cols, shorts)
            na_person = aggregate_day_preds_to_person(
                pred_na, shorts=shorts, pred_prefix=pfx
            )
            na_person["_fold"] = fold_i
            nonaux_person_fold_parts.append(na_person)

    if not oof_day_parts:
        raise AssertionError("OOF produced no day predictions")
    oof_days = pd.concat(oof_day_parts, axis=0, ignore_index=True)
    yhat_train_parts = [
        aggregate_day_preds_to_person(oof_days, shorts=shorts, pred_prefix=pfx)
    ]
    if nonaux_person_fold_parts:
        na_all = pd.concat(nonaux_person_fold_parts, axis=0, ignore_index=True)
        pack_cols = [
            c
            for c in na_all.columns
            if c.startswith(pfx) or c == "n_days_agg"
        ]
        na_mean = (
            na_all.groupby("person_id", sort=False)[pack_cols]
            .mean()
            .reset_index()
        )
        yhat_train_parts.append(na_mean)
    yhat_train = pd.concat(yhat_train_parts, axis=0, ignore_index=True)
    n_fallback_train = 0
    # every train person must have Ŷ (aux OOF + non-aux mean-K)
    train_pids = set(
        person_df.loc[person_df["recommended_split"] == "train", "person_id"]
    )
    missing_tr = train_pids - set(yhat_train["person_id"])
    if missing_tr:
        n_fallback_train = len(missing_tr)
        log(
            f"WARNING: {n_fallback_train} train pids with 0 watch-valid days — "
            "median-fill handoff from OOF aux"
        )
        med_row = {
            c: float(yhat_train[c].median())
            for c in handoff_col_names(cfg, "var")
            if c in yhat_train.columns
        }
        med_row["n_days_agg"] = 0
        fill = pd.DataFrame([{"person_id": p, **med_row} for p in missing_tr])
        yhat_train = pd.concat([yhat_train, fill], axis=0, ignore_index=True)

    # --- final model: full train∩aux supervised, early-stop on val∩aux ---
    final_models: dict[str, dict[float, LGBMRegressor]] = {}
    for s in shorts:
        final_models[s] = fit_quantile_heads(
            day_tr_sup,
            day_va_sup,
            x_cols,
            s,
            best_params[s],
            cfg,
            device=device,
            seed=seed,
            alphas=alphas,
        )

    n_fallback_by_split: dict[str, int] = {"train": int(n_fallback_train)}

    def _predict_split_persons(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        pids = person_df.loc[
            person_df["recommended_split"] == split_name, "person_id"
        ]
        d = day[day["person_id"].isin(set(pids)) & day["infer_ok"]]
        if len(d) == 0:
            raise AssertionError(f"no infer days for split {split_name}")
        day_pred = predict_day_quantiles(final_models, d, x_cols, shorts)
        person = aggregate_day_preds_to_person(day_pred, shorts=shorts, pred_prefix=pfx)
        # ensure all persons present (fallback train-aux median handoff)
        missing = set(pids) - set(person["person_id"])
        n_fallback_by_split[split_name] = int(len(missing))
        if missing:
            log(
                f"WARNING: {len(missing)} pids in {split_name} with 0 watch-valid days "
                "— median fill"
            )
            med_row = {
                c: float(yhat_train[c].median()) for c in var_cols if c in yhat_train
            }
            med_row["n_days_agg"] = 0
            fill = pd.DataFrame([{"person_id": p, **med_row} for p in missing])
            person = pd.concat([person, fill], axis=0, ignore_index=True)
        return person, day_pred

    yhat_val, day_pred_val = _predict_split_persons("val")
    yhat_test, day_pred_test = _predict_split_persons("test")

    # metrics on aux only
    true_cols = groups["true_cols"]
    val_true = person_df.loc[
        (person_df["recommended_split"] == "val")
        & person_df["aux_eligible"].astype(bool),
        ["person_id"] + true_cols,
    ]
    test_true = person_df.loc[
        (person_df["recommended_split"] == "test")
        & person_df["aux_eligible"].astype(bool),
        ["person_id"] + true_cols,
    ]
    s1_val = _metrics_person(yhat_val, val_true, cfg)
    s1_test = _metrics_person(yhat_test, test_true, cfg)

    # day-level R² on val aux supervised (secondary)
    day_val_sup_pred = predict_day_quantiles(final_models, day_va_sup, x_cols, shorts)
    day_r2 = {}
    for s in shorts:
        day_r2[s] = _r2_rmse(
            day_va_sup[f"y_{s}"].to_numpy(dtype=float),
            day_val_sup_pred[f"mid_{s}"].to_numpy(dtype=float),
        )
    s1_val["day_level"] = day_r2

    # coverage
    bounds_val = _person_lo_hi_from_day(
        day_pred_val[day_pred_val["person_id"].isin(set(val_true["person_id"]))],
        shorts,
    )
    coverage = _coverage_table(bounds_val, val_true, cfg)
    s1_val["coverage"] = coverage
    s1_val["coverage_pass"] = bool(coverage["pass"])

    # gates: smoke = mean R² > 0 only (PLAN_B2_V2 §3.10); 3-target is diagnostic
    gate_targets = list(cfg["decision_bars"]["stage1_r2_targets"])
    gate = {t: s1_val["per_target"].get(t, {}).get("r2", float("nan")) for t in gate_targets}
    s1_val["gate_targets_r2"] = gate
    mean_r2_smoke = s1_val["per_target"].get("mean", {}).get("r2", float("nan"))
    s1_val["smoke_mean_r2"] = mean_r2_smoke
    s1_val["gate_pass"] = bool(np.isfinite(mean_r2_smoke) and mean_r2_smoke > 0)
    s1_val["gate_pass_three_target_diagnostic"] = bool(
        all(np.isfinite(v) and v > 0 for v in gate.values())
    )
    # Early-kill uses mean R² on {mean,sd,tar} per PLAN_B2_V2 §3.10 (not all 4 shorts)
    ek_targets = list(cfg["decision_bars"]["stage1_r2_targets"])
    ek_r2s = [
        s1_val["per_target"].get(t, {}).get("r2", float("nan")) for t in ek_targets
    ]
    ek_r2s_f = [v for v in ek_r2s if np.isfinite(v)]
    mean_r2_ek = float(np.mean(ek_r2s_f)) if ek_r2s_f else float("nan")
    any_hi = any(
        np.isfinite(s1_val["per_target"].get(t, {}).get("r2", float("nan")))
        and s1_val["per_target"][t]["r2"]
        >= float(cfg["decision_bars"]["stage1_early_kill_any_r2"])
        for t in shorts
    )
    early_kill = bool(
        (
            not np.isfinite(mean_r2_ek)
            or mean_r2_ek < float(cfg["decision_bars"]["stage1_early_kill_mean_r2"])
        )
        and not any_hi
    )
    s1_val["early_kill"] = early_kill
    s1_val["early_kill_mean_r2"] = mean_r2_ek
    s1_val["early_kill_mean_r2_threshold"] = float(
        cfg["decision_bars"]["stage1_early_kill_mean_r2"]
    )

    log(
        f"stage1 val mean_r2={s1_val['mean_r2']:.4f} test mean_r2={s1_test['mean_r2']:.4f} "
        f"gate={s1_val['gate_pass']} coverage_pass={s1_val['coverage_pass']} "
        f"early_kill={early_kill}"
    )
    log(f"  coverage={coverage['per_target']}")

    # n_days diagnostics on train Ŷ (aux OOF vs non-aux mean-K)
    aux_tr_set = set(aux_tr_persons.tolist())
    nonaux_tr_set = set(nonaux_tr_persons.tolist())
    yhat_tr_aux = yhat_train[yhat_train["person_id"].isin(aux_tr_set)]
    yhat_tr_nonaux = yhat_train[yhat_train["person_id"].isin(nonaux_tr_set)]
    n_days_diag = {
        "train": _n_days_buckets(yhat_train, cfg),
        "train_aux_oof": _n_days_buckets(yhat_tr_aux, cfg),
        "train_nonaux": _n_days_buckets(yhat_tr_nonaux, cfg),
        "val": _n_days_buckets(yhat_val, cfg),
        "test": _n_days_buckets(yhat_test, cfg),
        "n_days_summary": {
            "train_min": int(yhat_train["n_days_agg"].min()) if len(yhat_train) else None,
            "train_p50": float(yhat_train["n_days_agg"].median()) if len(yhat_train) else None,
            "train_max": int(yhat_train["n_days_agg"].max()) if len(yhat_train) else None,
            "train_aux_n": int(len(yhat_tr_aux)),
            "train_nonaux_n": int(len(yhat_tr_nonaux)),
        },
        "n_fallback_persons": n_fallback_by_split,
    }

    # drop n_days_agg from Stage-2 handoff (coverage leakage cousin) — keep in diag only
    def _strip_ndays(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "n_days_agg"]
        return df[cols].copy()

    return Stage1Predictions(
        yhat_train=_strip_ndays(yhat_train),
        yhat_val=_strip_ndays(yhat_val),
        yhat_test=_strip_ndays(yhat_test),
        fold_label_counts=fold_counts,
        stage1_val_metrics=s1_val,
        stage1_test_metrics=s1_test,
        best_params_per_target=best_params,
        impute_medians=meds,
        x_cols=list(x_cols),
        diagnostics={
            "n_days": n_days_diag,
            "coverage": coverage,
            "n_train_sup_days": int(len(day_tr_sup)),
            "n_val_sup_days": int(len(day_va_sup)),
        },
    )
