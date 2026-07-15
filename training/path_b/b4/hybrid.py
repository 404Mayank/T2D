"""Stage-2 GBM: (z ∥ C1) and matched D1 for B4 ambition bar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
    # Freeze parity: meaningful when re-fit covers (near) full core.
    # pid_allow is often set to sequence survivors even on full claim — treat as
    # full-core-eligible when n ≈ expected_core_n (≤ T_min drops).
    expected_n = int(b4_cfg["data"]["expected_core_n"])
    n_persons = int(len(pids))
    full_core_like = n_persons >= expected_n - 5  # allow few T_min drops
    # Unrestricted path (pid_allow is None) is always a freeze check candidate
    freeze_check_eligible = pid_allow is None or full_core_like
    result["d1_matches_freeze"] = bool(
        freeze_check_eligible and abs(test_auc - freeze) <= tol
    )
    result["pid_allow_restricted"] = pid_allow is not None
    result["freeze_check_eligible"] = freeze_check_eligible
    result["fair_bar_note"] = (
        None
        if result["d1_matches_freeze"]
        else (
            f"D1 test 4-AUC {test_auc:.4f} vs freeze {freeze:.4f} "
            f"(Δ={test_auc - freeze:+.4f}); fair bar = re-fit D1 only"
            if freeze_check_eligible
            else f"restricted pool n={n_persons} (smoke/subsample); fair bar = re-fit D1 only"
        )
    )
    with open(out_dir / "d1_metrics.json", "w") as f:
        json.dump(
            {
                "family": result["family"],
                "val_macro_ovr_auc": result["val_macro_ovr_auc"],
                "test_raw": result["metrics"]["test_raw"],
                "test_cal": result["metrics"]["test_cal"],
                "d1_vs_freeze_delta": result["d1_vs_freeze_delta"],
                "d1_matches_freeze": result["d1_matches_freeze"],
                "freeze_check_eligible": freeze_check_eligible,
                "fair_bar_note": result["fair_bar_note"],
                "n_feat": len(c1_cols),
                "feature_cols": list(c1_cols),
                "n_persons": n_persons,
                "pid_allow_restricted": pid_allow is not None,
                "full_core_like": full_core_like,
                "expected_core_n": expected_n,
            },
            f,
            indent=2,
            default=float,
        )
    if result["fair_bar_note"]:
        log(f"  D1 fair-bar note: {result['fair_bar_note']}")
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


def _load_emb(emb_path: Path, bundle: GridBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings.npz aligned to bundle order → z, pids, y, split."""
    try:
        emb = np.load(emb_path, allow_pickle=False)
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
    return z, pids, y, np.asarray(split)


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
    """Load embeddings.npz (bundle order) and run z∥C1 GBM (F2 frozen-z recipe)."""
    z, pids, y, split = _load_emb(emb_path, bundle)
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


def run_z_only_arm(
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
    """F0b: GBM on z only (no C1) — orthogonal-signal probe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    z_cols = [f"z_{i}" for i in range(z.shape[1])]
    splits = _make_splits(z.astype(np.float32), y, split, pids, z_cols)
    pa_run = _path_a_run_cfg(repo, b4_cfg)
    if n_trials is not None:
        pa_run["run"]["n_trials"] = int(n_trials)
    log(f"=== z-only arm {arm_name} n_feat={z.shape[1]} ===")
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
                "n_feat": int(z.shape[1]),
                "feature_cols": z_cols,
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


def z_only_from_embeddings(
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
    z, pids, y, split = _load_emb(emb_path, bundle)
    return run_z_only_arm(
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


def build_oof_embeddings(
    bundle: GridBundle,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    out_path: Path,
    n_folds: int = 5,
    quick: bool = False,
    balancer: str = "none",
    lam: float = 0.0,
    log=print,
) -> Path:
    """K-fold OOF z on train; val/test z from full-train student (PLAN_B4_V2 F1).

    **Claim scope (V2 v1-impl):** class-only encoder (lam=0) OOF.
    This answers "does OOF-z reduce frozen-z dilution vs F2?" for the **μ=0 / CE**
    student — NOT the RKD-μ fusion ambition. Label artifacts accordingly.
    Per-fold RKD student OOF is a separate extension (not this function).
    """
    from sklearn.model_selection import StratifiedKFold

    from training.path_b.b4.train import train_one_lambda

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(bundle.pids)
    z_oof = np.zeros((n, int(cfg["model"]["hidden"])), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)

    train_idx = np.where(bundle.split == "train")[0]
    y_train = bundle.y[train_idx]
    # stratified folds on train labels (cap folds by rarest class count)
    from collections import Counter

    min_class = min(Counter(y_train.tolist()).values()) if len(y_train) else 1
    n_splits = max(2, min(n_folds, len(train_idx), min_class))
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg["train"]["seed"]),
    )
    fold_root = out_path.parent / f"{out_path.stem}_folds"
    fold_root.mkdir(parents=True, exist_ok=True)

    for fold, (tr_rel, va_rel) in enumerate(skf.split(train_idx, y_train)):
        tr_abs = train_idx[tr_rel]
        va_abs = train_idx[va_rel]
        log(f"  OOF fold {fold}: train_n={len(tr_abs)} oof_n={len(va_abs)}")
        # subset bundle for fold train by remapping split: keep only tr_abs as train;
        # use a tiny val from fold for ES (va_abs as val) — val labels used only for ES,
        # OOF z for va_abs taken from this fold model (held out of train).
        fold_bundle = _subset_bundle_for_oof(bundle, tr_abs, va_abs)
        fdir = fold_root / f"fold{fold}"
        met = train_one_lambda(
            fold_bundle,
            cfg,
            lam=float(lam),
            out_dir=fdir,
            device=device,
            quick=quick,
            balancer=balancer,  # type: ignore[arg-type]
        )
        # map embeddings back: fold_bundle.pids order
        emb = np.load(fdir / "embeddings.npz")
        z_f = emb["z"]
        pids_f = emb["pid"]
        # OOF rows = va_abs persons
        pid_to_zf = {int(p): z_f[i] for i, p in enumerate(pids_f)}
        for j in va_abs:
            pid = int(bundle.pids[j])
            if pid in pid_to_zf:
                z_oof[j] = pid_to_zf[pid]
                filled[j] = True

    # full-train student for val/test z
    log("  OOF full-train student for val/test z")
    full_dir = fold_root / "full_train"
    train_one_lambda(
        bundle,
        cfg,
        lam=float(lam),
        out_dir=full_dir,
        device=device,
        quick=quick,
        balancer=balancer,  # type: ignore[arg-type]
    )
    emb_full = np.load(full_dir / "embeddings.npz")
    z_full = emb_full["z"]
    # assume bundle order
    if not np.array_equal(emb_full["pid"], bundle.pids):
        order = {int(p): i for i, p in enumerate(emb_full["pid"])}
        z_full = z_full[[order[int(p)] for p in bundle.pids]]
    for j in range(n):
        if bundle.split[j] in ("val", "test"):
            z_oof[j] = z_full[j]
            filled[j] = True
        elif not filled[j]:
            # train pid missing from folds (shouldn't) — fall back full-train
            z_oof[j] = z_full[j]
            filled[j] = True

    meta = {
        "oof_scope": "class_only_lam0",
        "claim_note": (
            "F1 OOF is μ=0/CE encoder only — not RKD-μ distill OOF. "
            "Ambition bar on distill fusion remains F2 (frozen z∥C1) until "
            "per-fold RKD OOF is implemented."
        ),
        "n_folds": int(n_splits),
        "lam": float(lam),
        "balancer": balancer,
    }
    np.savez_compressed(
        out_path,
        z=z_oof,
        y=bundle.y,
        pid=bundle.pids,
        split=np.asarray(bundle.split, dtype="U16"),
        filled=filled,
        n_folds=n_splits,
        oof_scope=np.asarray(["class_only_lam0"]),
    )
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  wrote OOF embeddings → {out_path} (scope=class_only_lam0)")
    return out_path


def _subset_bundle_for_oof(
    bundle: GridBundle,
    train_abs: np.ndarray,
    val_abs: np.ndarray,
) -> GridBundle:
    """Build a GridBundle whose train/val are the fold indices; drop test (empty OK).

    Uses same tensors; remaps split labels. Persons outside train∪val get split='drop'
    and are excluded from Dataset via split filter — we only keep train∪val rows.
    """
    keep = np.concatenate([train_abs, val_abs])
    # preserve order: all train then val
    keep = np.unique(keep)
    # rebuild split array for kept rows
    train_set = set(int(i) for i in train_abs)
    val_set = set(int(i) for i in val_abs)

    def take(arr):
        return arr[keep]

    new_split = np.array(
        ["train" if int(i) in train_set else "val" for i in keep], dtype=object
    )
    # if overlap (shouldn't), prefer train
    for k, i in enumerate(keep):
        if int(i) in train_set:
            new_split[k] = "train"
        elif int(i) in val_set:
            new_split[k] = "val"

    return GridBundle(
        pids=take(bundle.pids),
        y=take(bundle.y),
        split=new_split,
        aux_eligible=take(bundle.aux_eligible),
        X=take(bundle.X),
        pad_mask=take(bundle.pad_mask),
        wear_mask=take(bundle.wear_mask),
        cgm=take(bundle.cgm),
        traj_mask=take(bundle.traj_mask),
        feature_cols=bundle.feature_cols,
        feat_mean=bundle.feat_mean,
        feat_std=bundle.feat_std,
        cgm_mean=bundle.cgm_mean,
        cgm_std=bundle.cgm_std,
        class_weights=bundle.class_weights,
        wear_valid_in_window=take(bundle.wear_valid_in_window),
        pad_frac=take(bundle.pad_frac),
        sub_start_idx=take(bundle.sub_start_idx),
        c1=take(bundle.c1) if bundle.c1 is not None else None,
        c1_cols=bundle.c1_cols,
        w0_cols=bundle.w0_cols,
        dropped_pids=list(bundle.dropped_pids or []),
    )
