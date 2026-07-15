"""Stage-2 GBM: (z ∥ C1) and matched D1 for B4 ambition bar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import load_watch_onboarding_mood
from training.path_b.b1.metrics import paired_bootstrap_delta_auc
from training.path_b.b2.stage2 import train_stage2
from training.path_b.b4.data import GridBundle


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def _path_a_run_cfg(repo: Path, b4_cfg: dict[str, Any]) -> dict[str, Any]:
    """Minimal cfg for path_a_watch HPO helpers (reuse path_a_blocks)."""
    pa_path = _resolve(repo, b4_cfg["paths"]["path_a_blocks_config"])
    with open(pa_path) as f:
        pa = yaml.safe_load(f)
    # stage2 needs run/calibration/hpo keys from path_a
    cfg = dict(pa)
    cfg["run"] = dict(pa.get("run") or {})
    cfg["run"]["n_trials"] = int(b4_cfg.get("stage2", {}).get("n_trials", 50))
    cfg["run"]["lgbm_device"] = b4_cfg.get("stage2", {}).get("lgbm_device", "auto")
    cfg["run"]["auc_tie_eps"] = float(b4_cfg.get("stage2", {}).get("auc_tie_eps", 0.001))
    if "calibration" not in cfg:
        cfg["calibration"] = dict(b4_cfg.get("calibration") or {})
    return cfg


def _make_splits(
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    pids: np.ndarray,
    feature_cols: list[str],
    pool: str = "wearable_core",
) -> dict[str, Any]:
    def take(s: str):
        m = split == s
        return X[m], y[m], pids[m]

    Xtr, ytr, pid_tr = take("train")
    Xva, yva, pid_va = take("val")
    Xte, yte, pid_te = take("test")
    return {
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xva,
        "y_val": yva,
        "X_test": Xte,
        "y_test": yte,
        "pid_train": pid_tr,
        "pid_val": pid_va,
        "pid_test": pid_te,
        "feature_cols": feature_cols,
        "pool": pool,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
    }


def load_c1_matrix(
    repo: Path,
    b4_cfg: dict[str, Any],
    pids: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return C1 X aligned to pids order, c1_cols, y, split from full core frame."""
    paths = b4_cfg["paths"]
    pa_path = _resolve(repo, paths["path_a_blocks_config"])
    with open(pa_path) as f:
        pa = yaml.safe_load(f)
    df, watch_cols, onboard_cols, mood_cols, c1_cols = load_watch_onboarding_mood(
        repo,
        watch_green=paths["watch_green"],
        onboarding=paths["onboarding"],
        mood=paths["mood"],
        pool_masks=paths["pool_masks"],
        onboarding_keep=list(pa["data"]["onboarding_keep"]),
        mood_cols=list(pa["data"]["mood_scores"]),
        expected_n=int(b4_cfg["data"]["expected_core_n"]),
    )
    df = df.set_index("person_id")
    rows = []
    ys = []
    splits = []
    for pid in pids:
        pid = int(pid)
        if pid not in df.index:
            raise KeyError(pid)
        r = df.loc[pid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        rows.append(pd.to_numeric(r[c1_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
        ys.append(int(r["label"]))
        splits.append(str(r["recommended_split"]))
    X = np.stack(rows, axis=0)
    return X, list(c1_cols), np.asarray(ys, dtype=np.int64), np.asarray(splits, dtype=object)


def run_d1(
    repo: Path,
    b4_cfg: dict[str, Any],
    out_dir: Path,
    *,
    n_trials: int | None = None,
    pid_allow: np.ndarray | list[int] | None = None,
    log=print,
) -> dict[str, Any]:
    """Re-fit matched Path A C1 GBM.

    Default: full wearable_core. If pid_allow is set (sequence-surviving /
    smoke pool), restrict so S*+C1−D1 pairs on the same people.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = b4_cfg["paths"]
    pa_path = _resolve(repo, paths["path_a_blocks_config"])
    with open(pa_path) as f:
        pa = yaml.safe_load(f)
    df, _, _, _, c1_cols = load_watch_onboarding_mood(
        repo,
        watch_green=paths["watch_green"],
        onboarding=paths["onboarding"],
        mood=paths["mood"],
        pool_masks=paths["pool_masks"],
        onboarding_keep=list(pa["data"]["onboarding_keep"]),
        mood_cols=list(pa["data"]["mood_scores"]),
        expected_n=int(b4_cfg["data"]["expected_core_n"]),
    )
    if pid_allow is not None:
        allow = {int(p) for p in np.asarray(pid_allow).tolist()}
        df = df[df["person_id"].astype(int).isin(allow)].copy()
        if len(df) != len(allow):
            missing = allow - set(df["person_id"].astype(int))
            raise AssertionError(
                f"D1 pid_allow missing from C1 frame: {sorted(missing)[:10]}"
            )
    X = df[c1_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    split = df["recommended_split"].astype(str).to_numpy()
    pids = df["person_id"].to_numpy(dtype=np.int64)
    splits = _make_splits(X, y, split, pids, list(c1_cols))
    pa_run = _path_a_run_cfg(repo, b4_cfg)
    if n_trials is not None:
        pa_run["run"]["n_trials"] = int(n_trials)
    result = train_stage2(splits, pa_run, n_trials=n_trials, log=log)
    result["pid_test"] = splits["pid_test"]
    result["y_test"] = splits["y_test"]
    freeze = float(b4_cfg["data"]["frozen_c1_4auc"])
    tol = float(b4_cfg["data"]["d1_drift_tol"])
    test_auc = float(result["metrics"]["test_raw"]["macro_ovr_auc"])
    result["freeze_c1_4auc"] = freeze
    result["d1_vs_freeze_delta"] = test_auc - freeze
    # freeze parity only meaningful on full core re-fit
    full_core = pid_allow is None and len(pids) == int(b4_cfg["data"]["expected_core_n"])
    result["d1_matches_freeze"] = bool(full_core and abs(test_auc - freeze) <= tol)
    result["pid_allow_restricted"] = pid_allow is not None
    with open(out_dir / "d1_metrics.json", "w") as f:
        json.dump(
            {
                "family": result["family"],
                "val_macro_ovr_auc": result["val_macro_ovr_auc"],
                "test_raw": result["metrics"]["test_raw"],
                "test_cal": result["metrics"]["test_cal"],
                "d1_vs_freeze_delta": result["d1_vs_freeze_delta"],
                "d1_matches_freeze": result["d1_matches_freeze"],
                "n_feat": len(c1_cols),
                "feature_cols": list(c1_cols),
                "n_persons": int(len(pids)),
                "pid_allow_restricted": pid_allow is not None,
                "full_core_freeze_check": full_core,
            },
            f,
            indent=2,
            default=float,
        )
    np.savez_compressed(
        out_dir / "d1_test_preds.npz",
        y=result["y_test"],
        proba=result["proba_test"],
        pid=result["pid_test"],
    )
    return result


def run_z_c1_arm(
    repo: Path,
    b4_cfg: dict[str, Any],
    *,
    z: np.ndarray,
    pids: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    arm_name: str,
    out_dir: Path,
    n_trials: int | None = None,
    log=print,
) -> dict[str, Any]:
    """GBM on concat(z, C1) for persons in arrays (same order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X_c1, c1_cols, y_c1, split_c1 = load_c1_matrix(repo, b4_cfg, pids)
    # y/split from C1 frame should match
    if not np.array_equal(y, y_c1):
        # allow if only ordering — we rebuilt from pids
        y = y_c1
        split = split_c1
    z_cols = [f"z_{i}" for i in range(z.shape[1])]
    X = np.concatenate([z.astype(np.float32), X_c1], axis=1)
    feat_cols = z_cols + c1_cols
    splits = _make_splits(X, y, split, pids, feat_cols)
    pa_run = _path_a_run_cfg(repo, b4_cfg)
    if n_trials is not None:
        pa_run["run"]["n_trials"] = int(n_trials)
    log(f"=== hybrid arm {arm_name} n_feat={len(feat_cols)} (z={z.shape[1]}+c1={len(c1_cols)}) ===")
    result = train_stage2(splits, pa_run, n_trials=n_trials, log=log)
    result["arm"] = arm_name
    result["pid_test"] = splits["pid_test"]
    result["y_test"] = splits["y_test"]
    with open(out_dir / f"{arm_name}_metrics.json", "w") as f:
        json.dump(
            {
                "arm": arm_name,
                "family": result["family"],
                "val_macro_ovr_auc": result["val_macro_ovr_auc"],
                "test_raw": result["metrics"]["test_raw"],
                "test_cal": result["metrics"]["test_cal"],
                "n_feat": len(feat_cols),
                "feature_cols": feat_cols,
            },
            f,
            indent=2,
            default=float,
        )
    np.savez_compressed(
        out_dir / f"{arm_name}_test_preds.npz",
        y=result["y_test"],
        proba=result["proba_test"],
        pid=result["pid_test"],
    )
    return result


def pair_boot_from_preds(
    path_a: Path,
    path_b: Path,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Join two test_preds.npz on pid; paired ΔAUC (b - a)."""
    a = np.load(path_a)
    b = np.load(path_b)
    da = pd.DataFrame({"pid": a["pid"], "y": a["y"]})
    for i in range(a["proba"].shape[1]):
        da[f"pa{i}"] = a["proba"][:, i]
    db = pd.DataFrame({"pid": b["pid"]})
    for i in range(b["proba"].shape[1]):
        db[f"pb{i}"] = b["proba"][:, i]
    m = da.merge(db, on="pid", how="inner")
    if len(m) != len(da) or len(m) != len(db):
        raise AssertionError(
            f"pid mismatch for bootstrap: |a|={len(da)} |b|={len(db)} |join|={len(m)}"
        )
    y = m["y"].to_numpy(dtype=int)
    pa = m[[c for c in m.columns if c.startswith("pa")]].to_numpy(dtype=float)
    pb = m[[c for c in m.columns if c.startswith("pb")]].to_numpy(dtype=float)
    return paired_bootstrap_delta_auc(y, pa, pb, n_boot=n_boot, seed=seed)


def hybrid_from_bundle_embeddings(
    repo: Path,
    b4_cfg: dict[str, Any],
    bundle: GridBundle,
    emb_path: Path,
    arm_name: str,
    out_dir: Path,
    *,
    n_trials: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Load embeddings.npz (bundle order) and run z∥C1 GBM."""
    # allow_pickle only if needed for legacy object `split`; z/y/pid are numeric.
    try:
        emb = np.load(emb_path, allow_pickle=False)
        _ = emb["z"]
        split_from_emb = None
        if "split" in emb.files:
            try:
                split_from_emb = np.asarray(emb["split"]).astype(str)
            except ValueError:
                split_from_emb = None
    except ValueError:
        emb = np.load(emb_path, allow_pickle=True)
        split_from_emb = None
        if "split" in emb.files:
            try:
                split_from_emb = np.asarray(emb["split"]).astype(str)
            except Exception:
                split_from_emb = None
    z = emb["z"]
    pids = emb["pid"]
    y = emb["y"]
    split = split_from_emb if split_from_emb is not None else bundle.split
    # align: embeddings saved in bundle order
    if not np.array_equal(pids, bundle.pids):
        order = {int(p): i for i, p in enumerate(pids)}
        idx = [order[int(p)] for p in bundle.pids]
        z = z[idx]
        pids = bundle.pids
        y = bundle.y
        split = bundle.split
    elif len(y) != len(bundle.pids):
        y = bundle.y
        split = bundle.split
    return run_z_c1_arm(
        repo,
        b4_cfg,
        z=z,
        pids=pids,
        y=y,
        split=split,
        arm_name=arm_name,
        out_dir=out_dir,
        n_trials=n_trials,
        log=log,
    )
