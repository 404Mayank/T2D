"""Load watch_daily ⋈ cgm_daily ⋈ pool_masks → person sequences for B1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _resolve(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _truncate_days(
    days: pd.DataFrame,
    max_len: int,
    *,
    cgm_valid_col: str = "cgm_day_valid",
) -> pd.DataFrame:
    """Prefer cgm-valid days, earliest tie-break; never last-N only."""
    if len(days) <= max_len:
        return days.sort_values("day_local").reset_index(drop=True)
    d = days.sort_values("day_local").reset_index(drop=True)
    has_cgm = d[cgm_valid_col].fillna(False).astype(bool) if cgm_valid_col in d.columns else pd.Series(False, index=d.index)
    cgm_rows = d.loc[has_cgm]
    other = d.loc[~has_cgm]
    take = cgm_rows.head(max_len)
    if len(take) < max_len:
        need = max_len - len(take)
        take = pd.concat([take, other.head(need)], ignore_index=True)
    return take.sort_values("day_local").reset_index(drop=True)


@dataclass
class SequenceBundle:
    """In-memory person sequences + fit artifacts (impute / zscore / weights)."""

    pids: np.ndarray
    y: np.ndarray
    X: list[np.ndarray]  # each [T, d]
    watch_mask: list[np.ndarray]
    glu_y: list[np.ndarray]
    glu_mask: list[np.ndarray]
    feature_cols: list[str]
    glu_cols: list[str]
    split: np.ndarray
    aux_eligible: np.ndarray
    impute_values: dict[str, float]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    glu_mean: np.ndarray
    glu_std: np.ndarray
    class_weights: np.ndarray
    # Optional person-level GREEN late-fusion (aligned to pids)
    green: np.ndarray | None = None  # [N, g] or None
    green_cols: list[str] | None = None
    green_mean: np.ndarray | None = None
    green_std: np.ndarray | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    import yaml

    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_sequences(
    repo_root: Path,
    cfg: dict[str, Any],
    *,
    max_participants: int | None = None,
) -> SequenceBundle:
    paths = cfg["paths"]
    feat_cols: list[str] = list(cfg["feature_cols"])
    glu_cols: list[str] = list(cfg["glu_cols"])
    fill_zero = set(cfg.get("fill_zero") or [])
    max_len = int(cfg["data"]["max_len"])
    t_min = int(cfg["data"]["t_min_valid_watch"])
    std_floor = float(cfg["data"]["glu_std_floor"])
    expected_n = int(cfg["data"]["expected_core_n"])

    wd = pd.read_parquet(_resolve(repo_root, paths["watch_daily"]))
    cgm = pd.read_parquet(_resolve(repo_root, paths["cgm_daily"]))
    meta = pd.read_parquet(_resolve(repo_root, paths["pool_masks"]))
    gcfg = cfg.get("green_fusion") or {}
    green_enabled = bool(gcfg.get("enabled"))
    green_df: pd.DataFrame | None = None
    green_cols: list[str] = []
    if green_enabled:
        gpath = paths.get("watch_green")
        if not gpath:
            raise ValueError("green_fusion.enabled but paths.watch_green missing")
        green_df = pd.read_parquet(_resolve(repo_root, gpath))
        green_df["person_id"] = green_df["person_id"].astype(int)
        if gcfg.get("cols"):
            green_cols = list(gcfg["cols"])
        else:
            green_cols = [
                c
                for c in green_df.columns
                if c != "person_id" and pd.api.types.is_numeric_dtype(green_df[c])
            ]
        miss_g = [c for c in green_cols if c not in green_df.columns]
        if miss_g:
            raise ValueError(f"watch_green missing cols {miss_g}")

    need_meta = [
        "person_id",
        "label",
        "recommended_split",
        "wearable_core",
        "aux_eligible",
    ]
    missing = [c for c in need_meta if c not in meta.columns]
    if missing:
        raise ValueError(f"pool_masks missing {missing}")

    core = meta.loc[meta["wearable_core"].astype(bool), need_meta].copy()
    core["person_id"] = core["person_id"].astype(int)
    if max_participants is not None:
        # Stratify by recommended_split (keep train/val/test present) and prefer aux.
        # Avoid taking only early person_ids (drops classes on small smoke sets).
        n_take = int(max_participants)
        parts = []
        splits = ["train", "val", "test"]
        # proportional share of n_take by full-core split sizes
        sizes = {
            s: int((core["recommended_split"] == s).sum()) for s in splits
        }
        total = sum(sizes.values()) or 1
        alloc = {s: max(1, int(round(n_take * sizes[s] / total))) for s in splits}
        # fix rounding to exact n_take
        while sum(alloc.values()) > n_take:
            for s in sorted(splits, key=lambda x: alloc[x], reverse=True):
                if alloc[s] > 1 and sum(alloc.values()) > n_take:
                    alloc[s] -= 1
        while sum(alloc.values()) < n_take:
            for s in splits:
                if sum(alloc.values()) < n_take:
                    alloc[s] += 1
        for s in splits:
            sub = core.loc[core["recommended_split"] == s].copy()
            sub = sub.sort_values(
                ["aux_eligible", "label", "person_id"],
                ascending=[False, True, True],
            )
            # round-robin labels so all classes appear when possible
            picked = []
            by_lab = {
                int(lb): g.sort_values("person_id")
                for lb, g in sub.groupby(sub["label"].astype(int))
            }
            labs = sorted(by_lab.keys())
            idx = {lb: 0 for lb in labs}
            while len(picked) < alloc[s] and labs:
                progress = False
                for lb in labs:
                    rows = by_lab[lb]
                    i = idx[lb]
                    if i < len(rows):
                        picked.append(rows.iloc[i])
                        idx[lb] = i + 1
                        progress = True
                        if len(picked) >= alloc[s]:
                            break
                if not progress:
                    break
            if picked:
                parts.append(pd.DataFrame(picked))
        core = pd.concat(parts, ignore_index=True) if parts else core.head(n_take)
        print(
            f"  subset max_participants={n_take} → n={len(core)} "
            f"alloc={alloc} labels={core['label'].astype(int).value_counts().to_dict()}",
            flush=True,
        )
    elif len(core) != expected_n:
        print(f"  warn: wearable_core n={len(core)} != {expected_n}")

    pids = core["person_id"].astype(int).tolist()
    pid_set = set(pids)

    wd = wd[wd["person_id"].isin(pid_set)].copy()
    cgm = cgm[cgm["person_id"].isin(pid_set)].copy()
    wd["person_id"] = wd["person_id"].astype(int)
    cgm["person_id"] = cgm["person_id"].astype(int)

    for c in feat_cols:
        if c not in wd.columns:
            raise ValueError(f"watch_daily missing feature {c}")
    for c in glu_cols:
        if c not in cgm.columns:
            raise ValueError(f"cgm_daily missing {c}")

    # Outer join day index per pid later; start with full day union tables
    cgm_keep = ["person_id", "day_local"] + glu_cols + ["cgm_day_valid"]
    days = wd.merge(cgm[cgm_keep], on=["person_id", "day_local"], how="outer")
    days = days.merge(core, on="person_id", how="inner")

    # After outer join, watch_day_valid may be NA for cgm-only days
    if "watch_day_valid" not in days.columns:
        days["watch_day_valid"] = False
    days["watch_day_valid"] = days["watch_day_valid"].fillna(False).astype(bool)
    days["cgm_day_valid"] = days["cgm_day_valid"].fillna(False).astype(bool)
    days["aux_eligible"] = days["aux_eligible"].astype(bool)

    # Feature NaN policy (numeric; fill_zero applied before impute)
    for c in feat_cols:
        days[c] = pd.to_numeric(days[c], errors="coerce")

    # Train median impute (valid watch days only)
    train_pids = set(
        core.loc[core["recommended_split"] == "train", "person_id"].astype(int)
    )
    train_valid = days[
        days["person_id"].isin(train_pids) & days["watch_day_valid"]
    ].copy()
    # Pre-fill observed mask for non-fill_zero cols (z-stats use observed-only).
    observed: dict[str, pd.Series] = {
        c: train_valid[c].notna() for c in feat_cols if c not in fill_zero
    }

    for c in feat_cols:
        if c in fill_zero:
            days[c] = days[c].fillna(0.0)
            train_valid[c] = train_valid[c].fillna(0.0)

    impute_values: dict[str, float] = {}
    for c in feat_cols:
        med = (
            float(train_valid[c].median())
            if c in train_valid and train_valid[c].notna().any()
            else 0.0
        )
        if not np.isfinite(med):
            med = 0.0
        impute_values[c] = med
        days[c] = days[c].fillna(med)

    # Train-only feature z-score (C2). fill_zero cols: post-fill rows;
    # other cols: observed-only (avoids median-mass compressing sleep std).
    feat_eps = float(cfg["data"].get("feat_std_floor", std_floor))
    feat_mean = np.zeros(len(feat_cols), dtype=np.float64)
    feat_std = np.ones(len(feat_cols), dtype=np.float64)
    for i, c in enumerate(feat_cols):
        if c in fill_zero:
            x = train_valid[c].to_numpy(dtype=float)
        else:
            mask = observed[c]
            x = train_valid.loc[mask, c].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            feat_mean[i] = float(x.mean())
            s = float(x.std(ddof=0))
            feat_std[i] = max(s, feat_eps)
        else:
            feat_mean[i] = 0.0
            feat_std[i] = 1.0

    for i, c in enumerate(feat_cols):
        days[c] = (days[c].astype(float) - feat_mean[i]) / feat_std[i]

    # Glu z-score on train ∩ aux ∩ cgm_day_valid ∩ watch_day_valid
    glu_fit = days[
        days["person_id"].isin(train_pids)
        & days["aux_eligible"]
        & days["cgm_day_valid"]
        & days["watch_day_valid"]
    ]
    glu_mean = np.zeros(len(glu_cols), dtype=np.float64)
    glu_std = np.ones(len(glu_cols), dtype=np.float64)
    for i, c in enumerate(glu_cols):
        x = pd.to_numeric(glu_fit[c], errors="coerce").dropna().to_numpy(dtype=float)
        if x.size:
            glu_mean[i] = float(x.mean())
            s = float(x.std(ddof=0))
            glu_std[i] = max(s, std_floor)
        else:
            glu_mean[i] = 0.0
            glu_std[i] = 1.0

    for i, c in enumerate(glu_cols):
        days[c] = (pd.to_numeric(days[c], errors="coerce") - glu_mean[i]) / glu_std[i]

    # Class weights from train labels
    y_train = core.loc[core["recommended_split"] == "train", "label"].astype(int).to_numpy()
    n_classes = int(cfg["model"]["n_classes"])
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    inv = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        inv[k] = 1.0 / counts[k] if counts[k] > 0 else 0.0
    if inv.sum() > 0:
        inv = inv / inv.sum()
    else:
        inv = np.ones(n_classes) / n_classes
    class_weights = inv

    # Person GREEN matrix (optional late-fusion) — train-only impute + z-score
    green_mean: np.ndarray | None = None
    green_std: np.ndarray | None = None
    green_by_pid: dict[int, np.ndarray] = {}
    if green_enabled and green_df is not None and green_cols:
        gmat = green_df.set_index("person_id")[green_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        train_idx = [
            int(p)
            for p in core.loc[core["recommended_split"] == "train", "person_id"]
            if int(p) in gmat.index
        ]
        g_train = gmat.loc[train_idx] if train_idx else gmat.iloc[0:0]
        green_mean = np.zeros(len(green_cols), dtype=np.float64)
        green_std = np.ones(len(green_cols), dtype=np.float64)
        for i, c in enumerate(green_cols):
            x = g_train[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size:
                green_mean[i] = float(x.mean())
                green_std[i] = max(float(x.std(ddof=0)), std_floor)
            med = float(np.nanmedian(g_train[c].to_numpy(dtype=float))) if len(g_train) else 0.0
            if not np.isfinite(med):
                med = 0.0
            gmat[c] = gmat[c].fillna(med)
        for pid in gmat.index.astype(int):
            v = gmat.loc[pid, green_cols].to_numpy(dtype=np.float64)
            green_by_pid[int(pid)] = ((v - green_mean) / green_std).astype(np.float32)

    # Build per-person sequences
    pids_out: list[int] = []
    y_out: list[int] = []
    X_out: list[np.ndarray] = []
    wm_out: list[np.ndarray] = []
    gy_out: list[np.ndarray] = []
    gm_out: list[np.ndarray] = []
    split_out: list[str] = []
    aux_out: list[bool] = []
    green_out: list[np.ndarray] = []

    meta_by_pid = core.set_index("person_id")
    g_dim = len(green_cols)
    zero_green = np.zeros(g_dim, dtype=np.float32) if g_dim else None
    for pid, g in days.groupby("person_id", sort=True):
        pid = int(pid)
        if pid not in meta_by_pid.index:
            continue
        row = meta_by_pid.loc[pid]
        # Backbone only on valid watch days (contiguous after sort; pack_padded-safe).
        g = g.loc[g["watch_day_valid"]].copy()
        if len(g) < t_min:
            continue
        g = _truncate_days(g, max_len)
        X = np.ascontiguousarray(g[feat_cols].to_numpy(dtype=np.float32))
        wm = np.ones(len(g), dtype=bool)  # all kept rows are watch-valid
        gy = np.ascontiguousarray(
            np.nan_to_num(g[glu_cols].to_numpy(dtype=np.float32), nan=0.0)
        )
        gm = g["cgm_day_valid"].to_numpy(dtype=bool) & bool(row["aux_eligible"])
        if not bool(row["aux_eligible"]):
            gm = np.zeros(len(g), dtype=bool)
        gm = np.ascontiguousarray(gm)

        pids_out.append(pid)
        y_out.append(int(row["label"]))
        X_out.append(X)
        wm_out.append(wm)
        gy_out.append(gy)
        gm_out.append(gm)
        split_out.append(str(row["recommended_split"]))
        aux_out.append(bool(row["aux_eligible"]))
        if green_enabled and g_dim:
            green_out.append(green_by_pid.get(pid, zero_green.copy()))

    pids_arr = np.asarray(pids_out, dtype=np.int64)
    y_arr = np.asarray(y_out, dtype=np.int64)
    split_arr = np.asarray(split_out)
    aux_arr = np.asarray(aux_out, dtype=bool)
    green_arr = (
        np.ascontiguousarray(np.stack(green_out, axis=0))
        if green_out
        else None
    )

    # Split disjoint assert
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa = set(pids_arr[split_arr == a])
        sb = set(pids_arr[split_arr == b])
        if sa & sb:
            raise AssertionError(f"pid overlap {a}/{b}: {sa & sb}")

    # Glu mask never for non-aux
    for i, aux in enumerate(aux_arr):
        if not aux and gm_out[i].any():
            raise AssertionError(f"glu mask on non-aux pid {pids_arr[i]}")

    return SequenceBundle(
        pids=pids_arr,
        y=y_arr,
        X=X_out,
        watch_mask=wm_out,
        glu_y=gy_out,
        glu_mask=gm_out,
        feature_cols=feat_cols,
        glu_cols=glu_cols,
        split=split_arr,
        aux_eligible=aux_arr,
        impute_values=impute_values,
        feat_mean=feat_mean,
        feat_std=feat_std,
        glu_mean=glu_mean,
        glu_std=glu_std,
        class_weights=class_weights,
        green=green_arr,
        green_cols=list(green_cols) if green_enabled else None,
        green_mean=green_mean,
        green_std=green_std,
    )


class PersonSeqDataset(Dataset):
    def __init__(self, bundle: SequenceBundle, split: str):
        idx = np.where(bundle.split == split)[0]
        if len(idx) == 0:
            raise ValueError(f"no persons in split={split}")
        self.idx = idx
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, Any]:
        j = int(self.idx[i])
        item = {
            "pid": int(self.bundle.pids[j]),
            "y": int(self.bundle.y[j]),
            "X": self.bundle.X[j],
            "watch_mask": self.bundle.watch_mask[j],
            "glu_y": self.bundle.glu_y[j],
            "glu_mask": self.bundle.glu_mask[j],
            "aux": bool(self.bundle.aux_eligible[j]),
        }
        if self.bundle.green is not None:
            item["green"] = self.bundle.green[j]
        return item


def collate_persons(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    d = batch[0]["X"].shape[1]
    lengths = [int(b["X"].shape[0]) for b in batch]
    t_max = max(lengths)
    X = torch.zeros(bsz, t_max, d, dtype=torch.float32)
    wm = torch.zeros(bsz, t_max, dtype=torch.bool)
    gy = torch.zeros(bsz, t_max, batch[0]["glu_y"].shape[1], dtype=torch.float32)
    gm = torch.zeros(bsz, t_max, dtype=torch.bool)
    y = torch.zeros(bsz, dtype=torch.long)
    pid = torch.zeros(bsz, dtype=torch.long)
    has_green = "green" in batch[0]
    if has_green:
        gdim = int(np.asarray(batch[0]["green"]).shape[0])
        green = torch.zeros(bsz, gdim, dtype=torch.float32)
    else:
        green = None
    for i, b in enumerate(batch):
        t = b["X"].shape[0]
        X[i, :t] = torch.from_numpy(np.asarray(b["X"], dtype=np.float32))
        wm[i, :t] = torch.from_numpy(np.asarray(b["watch_mask"], dtype=bool))
        gy[i, :t] = torch.from_numpy(np.asarray(b["glu_y"], dtype=np.float32))
        gm[i, :t] = torch.from_numpy(np.asarray(b["glu_mask"], dtype=bool))
        y[i] = int(b["y"])
        pid[i] = int(b["pid"])
        if has_green:
            green[i] = torch.from_numpy(np.asarray(b["green"], dtype=np.float32))
    out = {
        "X": X,
        "watch_mask": wm,
        "glu_y": gy,
        "glu_mask": gm,
        "y": y,
        "pid": pid,
    }
    if green is not None:
        out["green"] = green
    return out


def subset_bundle(bundle: SequenceBundle, split: str) -> dict[str, Any]:
    """Counts / glu coverage diagnostics."""
    idx = np.where(bundle.split == split)[0]
    n_glu_days = sum(int(bundle.glu_mask[i].sum()) for i in idx)
    n_aux = int(bundle.aux_eligible[idx].sum())
    return {
        "n": int(len(idx)),
        "n_aux": n_aux,
        "n_glu_days": n_glu_days,
        "label_counts": {
            int(k): int(v)
            for k, v in zip(*np.unique(bundle.y[idx], return_counts=True))
        },
    }
