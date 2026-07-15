"""Load person C1/W0 + daily watch/CGM; handoff packs; leakage guards (PLAN_B2_V2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import load_watch_onboarding_mood

DENY = {
    "label",
    "recommended_split",
    "clinical_site",
    "person_id",
    "study_group",
    "wearable_core",
    "wearable_core_strict",
    "aux_eligible",
    "age_discrepancy",
    "day_local",
    "infer_ok",
    "supervised",
}


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def load_path_a_blocks_cfg(repo: Path, path: str) -> dict[str, Any]:
    with _resolve(repo, path).open() as f:
        return yaml.safe_load(f)


def load_yaml(repo: Path, path: str) -> dict[str, Any]:
    with _resolve(repo, path).open() as f:
        return yaml.safe_load(f)


def pin_hpo_from_frozen_b2(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Overwrite cfg hpo/class_weights/calibration from frozen b2 config if present."""
    ref_path = cfg["paths"].get("frozen_b2_config")
    if not ref_path:
        return cfg
    p = _resolve(repo, ref_path)
    if not p.exists():
        return cfg
    ref = load_yaml(repo, ref_path)
    out = dict(cfg)
    out["hpo"] = ref.get("hpo", cfg.get("hpo"))
    for k in ("class_weights", "calibration"):
        if k in ref:
            out[k] = ref[k]
    return out


def hpo_space_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "hpo": cfg.get("hpo"),
        "class_weights": cfg.get("class_weights"),
        "calibration": cfg.get("calibration"),
        "source": cfg["paths"].get("frozen_b2_config", "inline"),
    }


def short_targets(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["data"]["glu_short"])


def daily_targets(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["data"]["glu_targets_daily"])


def person_targets(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["data"]["glu_targets_person"])


def short_to_person(cfg: dict[str, Any]) -> dict[str, str]:
    return dict(zip(short_targets(cfg), person_targets(cfg), strict=True))


def short_to_daily(cfg: dict[str, Any]) -> dict[str, str]:
    return dict(zip(short_targets(cfg), daily_targets(cfg), strict=True))


def handoff_col_names(cfg: dict[str, Any], pack: str) -> list[str]:
    """pack: 'point' | 'var' | 'true'."""
    pfx = cfg["data"]["pred_prefix"]
    tpfx = cfg["data"]["true_prefix"]
    shorts = short_targets(cfg)
    s2p = short_to_person(cfg)
    if pack == "point":
        return [f"{pfx}{s}_mid" for s in shorts]
    if pack == "var":
        cols: list[str] = []
        for s in shorts:
            cols.extend(
                [
                    f"{pfx}{s}_mid",
                    f"{pfx}{s}_spread",
                    f"{pfx}{s}_daysd",
                ]
            )
        return cols
    if pack == "true":
        return [f"{tpfx}{s2p[s]}" for s in shorts]
    raise ValueError(pack)


def load_person_frame(
    repo: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Wearable_core person frame: W0, C1, true reduced person CGM, pool flags."""
    pa = load_path_a_blocks_cfg(repo, cfg["paths"]["path_a_blocks_config"])
    onboard_keep = list(pa["data"]["onboarding_keep"])
    mood_scores = list(pa["data"]["mood_scores"])

    df, watch_cols, onboard_cols, mood_cols, c1_cols = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=onboard_keep,
        mood_cols=mood_scores,
        expected_n=int(cfg["data"]["expected_n_core"]),
    )
    if "aux_eligible" not in df.columns:
        raise AssertionError("aux_eligible missing after merge")

    w0 = list(watch_cols)
    if len(w0) != 30:
        raise AssertionError(f"expected 30 GREEN cols, got {len(w0)}")
    expected_c1_n = int(cfg["data"]["expected_c1_n_feat"])
    if len(c1_cols) != expected_c1_n:
        raise AssertionError(
            f"C1 n_feat={len(c1_cols)} != expected {expected_c1_n}: {c1_cols}"
        )

    cgm = pd.read_parquet(_resolve(repo, cfg["paths"]["cgm_person"]))
    person_tgts = person_targets(cfg)
    forbid = set(cfg["data"]["glu_forbid"])
    bad = [c for c in person_tgts if c in forbid]
    if bad:
        raise AssertionError(f"forbidden glu targets: {bad}")
    miss = [c for c in person_tgts if c not in cgm.columns]
    if miss:
        raise ValueError(f"cgm_person missing {miss}")

    true_prefix = cfg["data"]["true_prefix"]
    keep_cgm = ["person_id"] + person_tgts
    cgm = cgm[keep_cgm].copy()
    rename = {c: f"{true_prefix}{c}" for c in person_tgts}
    cgm = cgm.rename(columns=rename)
    true_cols = [rename[c] for c in person_tgts]

    n_before = len(df)
    df = df.merge(cgm, on="person_id", how="left")
    if len(df) != n_before:
        raise AssertionError("cgm_person merge changed row count")

    aux = df["aux_eligible"].astype(bool)
    if df.loc[aux, true_cols].isna().any().any():
        n_bad = int(df.loc[aux, true_cols].isna().any(axis=1).sum())
        raise AssertionError(f"aux pids with null true CGM: {n_bad}")

    n_aux = int(aux.sum())
    exp_aux = int(cfg["data"]["expected_n_aux"])
    if n_aux != exp_aux:
        raise AssertionError(f"aux_eligible n={n_aux} != {exp_aux}")

    groups = {
        "w0": w0,
        "onboarding": list(onboard_cols),
        "mood": list(mood_cols),
        "c1": list(c1_cols),
        "glu_short": short_targets(cfg),
        "glu_daily": daily_targets(cfg),
        "glu_person": person_tgts,
        "true_cols": true_cols,
        "point_cols": handoff_col_names(cfg, "point"),
        "var_cols": handoff_col_names(cfg, "var"),
    }
    return df, groups


def build_day_table(
    repo: Path,
    cfg: dict[str, Any],
    person_df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Person×day table for Stage-1: watch days + optional CGM + GREEN fuse.

    Returns (day_df, x_cols, impute_medians_placeholder empty — filled later).
    """
    day_key = cfg["data"]["day_key"]
    wd = pd.read_parquet(_resolve(repo, cfg["paths"]["watch_daily"]))
    cgm = pd.read_parquet(_resolve(repo, cfg["paths"]["cgm_daily"]))

    day_feats = list(cfg["data"]["day_watch_feats"])
    miss_w = [c for c in day_feats if c not in wd.columns]
    if miss_w:
        raise ValueError(f"watch_daily missing {miss_w}")
    d_tgts = daily_targets(cfg)
    miss_c = [c for c in d_tgts if c not in cgm.columns]
    if miss_c:
        raise ValueError(f"cgm_daily missing {miss_c}")

    # Keep core persons only
    core_pids = set(person_df["person_id"].tolist())
    wd = wd[wd["person_id"].isin(core_pids)].copy()
    cgm = cgm[cgm["person_id"].isin(core_pids)].copy()

    # Exact join key assert material: both must use day_local strings
    if day_key not in wd.columns or day_key not in cgm.columns:
        raise AssertionError(f"missing day key {day_key}")

    keep_wd = (
        ["person_id", day_key, "watch_day_valid"]
        + day_feats
    )
    wd = wd[keep_wd]
    keep_cgm = ["person_id", day_key, "cgm_day_valid"] + d_tgts
    cgm = cgm[keep_cgm]

    # Exact key uniqueness before join (no ordinal day math)
    if wd.duplicated(["person_id", day_key]).any():
        raise AssertionError("duplicate (person_id, day_local) in watch_daily")
    if cgm.duplicated(["person_id", day_key]).any():
        raise AssertionError("duplicate (person_id, day_local) in cgm_daily")
    # Watch base for inference; CGM left-joined on exact (person_id, day_local)
    day = wd.merge(cgm, on=["person_id", day_key], how="left")
    if day.duplicated(["person_id", day_key]).any():
        raise AssertionError("duplicate (person_id, day_local) after watch/cgm join")

    # Person meta
    meta_cols = [
        "person_id",
        "label",
        "recommended_split",
        "aux_eligible",
    ]
    day = day.merge(person_df[meta_cols], on="person_id", how="inner")
    if day["person_id"].nunique() != person_df["person_id"].nunique():
        # non-core watch rows dropped ok; every core pid should have some days
        missing = set(person_df["person_id"]) - set(day["person_id"])
        if missing:
            raise AssertionError(
                f"{len(missing)} core pids have zero watch_daily rows"
            )

    green_fuse = bool(cfg["run"].get("green_fuse", True))
    x_cols = list(day_feats)
    if green_fuse:
        gpfx = cfg["data"]["green_prefix"]
        w0 = groups["w0"]
        g = person_df[["person_id"] + w0].copy()
        g = g.rename(columns={c: f"{gpfx}{c}" for c in w0})
        g_cols = [f"{gpfx}{c}" for c in w0]
        day = day.merge(g, on="person_id", how="left")
        x_cols = x_cols + g_cols

    # short target columns on day table
    s2d = short_to_daily(cfg)
    for s, dcol in s2d.items():
        day[f"y_{s}"] = day[dcol]

    day["watch_day_valid"] = day["watch_day_valid"].astype(bool)
    day["cgm_day_valid"] = day["cgm_day_valid"].fillna(False).astype(bool)
    day["supervised"] = (
        day["watch_day_valid"]
        & day["cgm_day_valid"]
        & day["aux_eligible"].astype(bool)
    )
    day["infer_ok"] = day["watch_day_valid"]

    return day, x_cols, {}


def fit_impute_medians(
    day_df: pd.DataFrame,
    x_cols: list[str],
    *,
    train_mask: pd.Series,
) -> dict[str, float]:
    """Medians on train supervised (or train infer) days only."""
    sub = day_df.loc[train_mask, x_cols]
    meds: dict[str, float] = {}
    for c in x_cols:
        v = sub[c]
        m = float(v.median()) if v.notna().any() else 0.0
        if not np.isfinite(m):
            m = 0.0
        meds[c] = m
    return meds


def apply_impute(
    day_df: pd.DataFrame,
    x_cols: list[str],
    meds: dict[str, float],
) -> pd.DataFrame:
    out = day_df.copy()
    for c in x_cols:
        fill = meds.get(c, 0.0)
        out[c] = out[c].fillna(fill)
    return out


def assert_no_leakage(feature_cols: list[str], cfg: dict[str, Any]) -> None:
    forbid = set(DENY) | set(cfg["data"]["glu_forbid"])
    forbid |= set(cfg["data"]["glu_targets_daily"])
    forbid |= set(cfg["data"]["glu_targets_person"])
    forbid |= {f"y_{s}" for s in short_targets(cfg)}
    forbid |= {"n_days_agg", "n_days"}
    bad = [
        c
        for c in feature_cols
        if c in forbid or c.startswith("ytrue_") or c.startswith("y_")
    ]
    if bad:
        raise AssertionError(f"leakage/deny in features: {bad}")


def assert_oracle_features(feature_cols: list[str], true_cols: list[str]) -> None:
    for c in true_cols:
        if c not in feature_cols:
            raise AssertionError(f"oracle missing {c}")
    if any(c.startswith("yhat_") for c in feature_cols):
        raise AssertionError("oracle arm must not include yhat_ predicted CGM")


@dataclass
class Stage1Predictions:
    yhat_train: pd.DataFrame
    yhat_val: pd.DataFrame
    yhat_test: pd.DataFrame
    fold_label_counts: list[dict[str, Any]]
    stage1_val_metrics: dict[str, Any]
    stage1_test_metrics: dict[str, Any]
    best_params_per_target: dict[str, dict[str, Any]]
    impute_medians: dict[str, float] = field(default_factory=dict)
    x_cols: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def merge_yhat_into_df(
    df: pd.DataFrame,
    preds: Stage1Predictions,
    handoff_cols: list[str],
) -> pd.DataFrame:
    parts = [preds.yhat_train, preds.yhat_val, preds.yhat_test]
    yhat = pd.concat(parts, axis=0, ignore_index=True)
    if yhat["person_id"].duplicated().any():
        raise AssertionError("duplicate person_id in concatenated Ŷ")
    out = df.merge(yhat, on="person_id", how="left")
    if out[handoff_cols].isna().any().any():
        n = int(out[handoff_cols].isna().any(axis=1).sum())
        raise AssertionError(f"missing Ŷ after merge for {n} pids")
    return out


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    pool: str,
    id_col: str = "person_id",
    label_col: str = "label",
    split_col: str = "recommended_split",
) -> dict[str, Any]:
    d = df
    if pool == "aux":
        d = df[df["aux_eligible"].astype(bool)].copy()
    elif pool != "core":
        raise ValueError(pool)

    def _part(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        m = d[split_col] == name
        if not m.any():
            raise AssertionError(f"no rows for {pool}/{name}")
        X = d.loc[m, feature_cols].copy()
        y = d.loc[m, label_col].to_numpy(dtype=np.int64)
        pid = d.loc[m, id_col].to_numpy()
        return X, y, pid

    Xtr, ytr, pid_tr = _part("train")
    Xva, yva, pid_va = _part("val")
    Xte, yte, pid_te = _part("test")
    return {
        "pool": pool,
        "feature_cols": list(feature_cols),
        "X_train": Xtr,
        "y_train": ytr,
        "pid_train": pid_tr,
        "X_val": Xva,
        "y_val": yva,
        "pid_val": pid_va,
        "X_test": Xte,
        "y_test": yte,
        "pid_test": pid_te,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
    }


def yhat_drift_table(
    df: pd.DataFrame,
    cols: list[str],
    *,
    split: str = "test",
) -> dict[str, Any]:
    d = df[df["recommended_split"] == split]
    out: dict[str, Any] = {"split": split, "dims": {}}
    for c in cols:
        rows = {}
        for name, mask in (
            ("aux", d["aux_eligible"].astype(bool)),
            ("non_aux", ~d["aux_eligible"].astype(bool)),
        ):
            s = d.loc[mask, c]
            if len(s) == 0:
                rows[name] = {"n": 0}
            else:
                rows[name] = {
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "p10": float(s.quantile(0.10)),
                    "p50": float(s.quantile(0.50)),
                    "p90": float(s.quantile(0.90)),
                }
        out["dims"][c] = rows
    return out


def subsample_train_for_smoke(
    person_df: pd.DataFrame,
    day_df: pd.DataFrame,
    *,
    n_train: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all val/test persons; subsample train stratified by label."""
    tr = person_df[person_df["recommended_split"] == "train"]
    other = person_df[person_df["recommended_split"] != "train"]
    if len(tr) <= n_train:
        return person_df.copy(), day_df.copy()
    parts = []
    rng = np.random.default_rng(seed)
    for _lab, g in tr.groupby("label"):
        frac = len(g) / len(tr)
        k = max(1, int(round(n_train * frac)))
        k = min(k, len(g))
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        parts.append(person_df.loc[idx])
    tr_s = pd.concat(parts, axis=0)
    if len(tr_s) > n_train:
        tr_s = tr_s.sample(n=n_train, random_state=seed)
    person_out = pd.concat([tr_s, other], axis=0).sort_index().reset_index(drop=True)
    keep = set(person_out["person_id"].tolist())
    day_out = day_df[day_df["person_id"].isin(keep)].copy().reset_index(drop=True)
    return person_out, day_out


def aggregate_day_preds_to_person(
    day_pred: pd.DataFrame,
    *,
    shorts: list[str],
    pred_prefix: str = "yhat_",
) -> pd.DataFrame:
    """day_pred: person_id + for each short: mid/lo/hi columns (day-level).

    Returns person_id + yhat_{s}_mid/spread/daysd + n_days_agg.
    """
    if day_pred.empty:
        cols = ["person_id", "n_days_agg"]
        for s in shorts:
            cols.extend(
                [
                    f"{pred_prefix}{s}_mid",
                    f"{pred_prefix}{s}_spread",
                    f"{pred_prefix}{s}_daysd",
                ]
            )
        return pd.DataFrame(columns=cols)

    rows = []
    for pid, g in day_pred.groupby("person_id", sort=False):
        rec: dict[str, Any] = {"person_id": pid, "n_days_agg": int(len(g))}
        for s in shorts:
            mid = g[f"mid_{s}"].to_numpy(dtype=float)
            lo = g[f"lo_{s}"].to_numpy(dtype=float)
            hi = g[f"hi_{s}"].to_numpy(dtype=float)
            finite = np.isfinite(mid)
            if not finite.any():
                rec[f"{pred_prefix}{s}_mid"] = np.nan
                rec[f"{pred_prefix}{s}_spread"] = np.nan
                rec[f"{pred_prefix}{s}_daysd"] = np.nan
                continue
            m = mid[finite]
            spread = (hi - lo)[finite]
            rec[f"{pred_prefix}{s}_mid"] = float(np.mean(m))
            rec[f"{pred_prefix}{s}_spread"] = float(np.mean(spread))
            rec[f"{pred_prefix}{s}_daysd"] = (
                float(np.std(m, ddof=0)) if len(m) >= 2 else 0.0
            )
        rows.append(rec)
    return pd.DataFrame(rows)
