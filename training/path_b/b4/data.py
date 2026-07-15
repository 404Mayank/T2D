"""Load 5-min grid → fixed-length person tensors (CGM-free subwindow)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from pipeline.fe.grid_5min import choose_subwindow_start
from training.path_a_blocks.data_blocks import load_watch_onboarding_mood


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class GridBundle:
    pids: np.ndarray
    y: np.ndarray
    split: np.ndarray
    aux_eligible: np.ndarray
    # [N, T, d] wear features (padded); pad_mask True = pad
    X: np.ndarray
    pad_mask: np.ndarray  # [N, T] bool True=pad
    wear_mask: np.ndarray  # [N, T] bool True=wear_valid (and not pad)
    cgm: np.ndarray  # [N, T] z-scored targets (0 where invalid)
    traj_mask: np.ndarray  # [N, T] bool supervision mask
    feature_cols: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    cgm_mean: float
    cgm_std: float
    class_weights: np.ndarray
    # subwindow diagnostics
    wear_valid_in_window: np.ndarray
    pad_frac: np.ndarray
    sub_start_idx: np.ndarray
    # optional C1 static aligned to pids [N, c1]
    c1: np.ndarray | None = None
    c1_cols: list[str] | None = None
    w0_cols: list[str] | None = None
    # full core C1 frame for Stage-2 (may include dropped-seq pids? no — same pids)
    # person quality
    dropped_pids: list[int] | None = None


def _load_grid_table(repo: Path, cfg: dict[str, Any], pids: list[int]) -> pd.DataFrame:
    paths = cfg["paths"]
    single = _resolve(repo, paths["grid_5min"])
    if single.exists():
        df = pd.read_parquet(single)
        df["person_id"] = df["person_id"].astype(int)
        return df[df["person_id"].isin(pids)].copy()

    idx_path = _resolve(repo, paths.get("grid_5min_index", "data/processed/features/grid_5min_index.parquet"))
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Missing {single} and {idx_path} — run: python -m pipeline.run_fe --blocks grid_5min"
        )
    idx = pd.read_parquet(idx_path)
    feat_dir = idx_path.parent
    parts = []
    for row in idx.itertuples(index=False):
        pid = int(row.person_id)
        if pid not in set(pids):
            continue
        p = feat_dir / str(row.path)
        parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError("no grid shards for requested pids")
    df = pd.concat(parts, ignore_index=True)
    df["person_id"] = df["person_id"].astype(int)
    return df


def build_grid_bundle(
    repo: Path,
    cfg: dict[str, Any],
    *,
    max_participants: int | None = None,
    load_c1: bool = True,
) -> GridBundle:
    paths = cfg["paths"]
    feat_cols: list[str] = list(cfg["feature_cols"])
    fill_zero = set(cfg.get("fill_zero") or [])
    t_bins = int(cfg["data"]["t_bins"])
    t_min = int(cfg["data"]["t_min"])
    feat_floor = float(cfg["data"]["feat_std_floor"])
    cgm_floor = float(cfg["data"]["cgm_std_floor"])
    expected_n = int(cfg["data"]["expected_core_n"])

    meta = pd.read_parquet(_resolve(repo, paths["pool_masks"]))
    need = ["person_id", "label", "recommended_split", "wearable_core", "aux_eligible"]
    miss = [c for c in need if c not in meta.columns]
    if miss:
        raise ValueError(f"pool_masks missing {miss}")
    core = meta.loc[meta["wearable_core"].astype(bool), need].copy()
    core["person_id"] = core["person_id"].astype(int)
    if max_participants is None and len(core) != expected_n:
        raise AssertionError(f"wearable_core n={len(core)} != {expected_n}")

    if max_participants is not None:
        n_take = int(max_participants)
        splits = ["train", "val", "test"]
        sizes = {s: int((core["recommended_split"] == s).sum()) for s in splits}
        total = sum(sizes.values()) or 1
        alloc = {s: max(1, int(round(n_take * sizes[s] / total))) for s in splits}
        while sum(alloc.values()) > n_take:
            for s in sorted(splits, key=lambda x: alloc[x], reverse=True):
                if alloc[s] > 1 and sum(alloc.values()) > n_take:
                    alloc[s] -= 1
        while sum(alloc.values()) < n_take:
            for s in splits:
                if sum(alloc.values()) < n_take:
                    alloc[s] += 1
        parts = []
        for s in splits:
            sub = core.loc[core["recommended_split"] == s].copy()
            # prefer aux
            sub = sub.sort_values(["aux_eligible", "person_id"], ascending=[False, True])
            parts.append(sub.head(alloc[s]))
        core = pd.concat(parts, ignore_index=True)

    pids_list = core["person_id"].astype(int).tolist()
    grid = _load_grid_table(repo, cfg, pids_list)
    if grid.empty:
        raise RuntimeError("empty grid for selected pids")
    grid_pids = set(grid["person_id"].astype(int).unique())
    missing_grid = [p for p in pids_list if p not in grid_pids]
    if missing_grid:
        # Footgun: smoke FE (e.g. 20 pids) left on disk while train asks for more.
        if max_participants is None:
            raise RuntimeError(
                f"grid_5min missing {len(missing_grid)}/{len(pids_list)} core pids "
                f"(e.g. {missing_grid[:5]}). Re-run full: "
                f"python -m pipeline.run_fe --blocks grid_5min"
            )
        # quick/smoke: shrink to intersection (logged via dropped + counts)
        core = core[core["person_id"].isin(grid_pids)].copy()
        pids_list = core["person_id"].astype(int).tolist()
        if not pids_list:
            raise RuntimeError("no overlap between selected pids and grid_5min")

    # optional C1 static
    c1_mat = None
    c1_cols: list[str] | None = None
    w0_cols: list[str] | None = None
    if load_c1:
        pa_cfg_path = paths.get("path_a_blocks_config", "training/path_a_blocks/config.yaml")
        with open(_resolve(repo, pa_cfg_path)) as f:
            pa = yaml.safe_load(f)
        # Always load full core C1 (expected_n=1824); filter to selected pids below.
        df_c1, watch_cols, onboard_cols, mood_cols, c1_list = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=list(pa["data"]["onboarding_keep"]),
            mood_cols=list(pa["data"]["mood_scores"]),
            expected_n=int(cfg["data"]["expected_core_n"]),
        )
        c1_cols = list(c1_list)
        w0_cols = list(watch_cols)
        exp_c1 = int(cfg["data"]["expected_c1_n_feat"])
        if len(c1_cols) != exp_c1:
            raise AssertionError(f"C1 n_feat={len(c1_cols)} != {exp_c1}")
        df_c1 = df_c1.set_index("person_id")

    # first pass: build raw windows per pid; collect train stats
    raw: dict[int, dict[str, Any]] = {}
    dropped: list[int] = []
    for pid, g in grid.groupby("person_id", sort=True):
        pid = int(pid)
        if pid not in set(pids_list):
            continue
        g = g.sort_values("bin_start_utc").reset_index(drop=True)
        wear = g["wear_bin_valid"].fillna(False).astype(bool).to_numpy()
        s, e = choose_subwindow_start(wear, t_bins)
        win = g.iloc[s:e].reset_index(drop=True)
        n_w = int(win["wear_bin_valid"].fillna(False).astype(bool).sum()) if len(win) else 0
        if n_w < t_min:
            dropped.append(pid)
            continue
        raw[pid] = {"win": win, "start": s, "n_wear": n_w}

    core = core[core["person_id"].isin(raw.keys())].copy()
    if core.empty:
        raise RuntimeError(f"all pids dropped by T_min={t_min}; check grid FE")

    # impute + scale on train only
    train_pids = set(
        core.loc[core["recommended_split"] == "train", "person_id"].astype(int)
    )
    # collect observed values for feat mean/std and cgm
    feat_vals = {c: [] for c in feat_cols}
    cgm_vals: list[float] = []
    for pid, rec in raw.items():
        if pid not in train_pids:
            continue
        win = rec["win"]
        wear_m = win["wear_bin_valid"].fillna(False).astype(bool).to_numpy()
        for c in feat_cols:
            x = pd.to_numeric(win[c], errors="coerce").to_numpy(dtype=float)
            if c in fill_zero:
                x = np.where(np.isfinite(x), x, 0.0)
            obs = x[wear_m & np.isfinite(x)]
            if obs.size:
                feat_vals[c].append(obs)
        # traj: wear & cgm & aux
        aux = bool(core.loc[core["person_id"] == pid, "aux_eligible"].iloc[0])
        if aux:
            cgm_m = win["cgm_bin_valid"].fillna(False).astype(bool).to_numpy()
            tm = wear_m & cgm_m
            cg = pd.to_numeric(win["cgm"], errors="coerce").to_numpy(dtype=float)
            obs = cg[tm & np.isfinite(cg)]
            if obs.size:
                cgm_vals.append(obs)

    feat_mean = np.zeros(len(feat_cols), dtype=np.float64)
    feat_std = np.ones(len(feat_cols), dtype=np.float64)
    for i, c in enumerate(feat_cols):
        if feat_vals[c]:
            v = np.concatenate(feat_vals[c])
            feat_mean[i] = float(np.mean(v))
            feat_std[i] = float(max(np.std(v, ddof=0), feat_floor))
        else:
            feat_mean[i] = 0.0
            feat_std[i] = 1.0
    if cgm_vals:
        cv = np.concatenate(cgm_vals)
        cgm_mean = float(np.mean(cv))
        cgm_std = float(max(np.std(cv, ddof=0), cgm_floor))
    else:
        cgm_mean, cgm_std = 0.0, 1.0

    # tensorize
    N = len(core)
    d = len(feat_cols)
    X = np.zeros((N, t_bins, d), dtype=np.float32)
    pad_mask = np.ones((N, t_bins), dtype=bool)  # True=pad
    wear_mask = np.zeros((N, t_bins), dtype=bool)
    cgm_arr = np.zeros((N, t_bins), dtype=np.float32)
    traj_mask = np.zeros((N, t_bins), dtype=bool)
    y = np.zeros(N, dtype=np.int64)
    split = np.empty(N, dtype=object)
    aux_elig = np.zeros(N, dtype=bool)
    pids = np.zeros(N, dtype=np.int64)
    wear_in_win = np.zeros(N, dtype=np.int32)
    pad_frac = np.zeros(N, dtype=np.float32)
    sub_start = np.zeros(N, dtype=np.int32)
    c1_mat = None
    if load_c1 and c1_cols is not None:
        c1_mat = np.zeros((N, len(c1_cols)), dtype=np.float32)

    core = core.sort_values("person_id").reset_index(drop=True)
    for i, row in core.iterrows():
        pid = int(row["person_id"])
        rec = raw[pid]
        win = rec["win"]
        L = len(win)
        pids[i] = pid
        y[i] = int(row["label"])
        split[i] = str(row["recommended_split"])
        aux_elig[i] = bool(row["aux_eligible"])
        wear_in_win[i] = int(rec["n_wear"])
        sub_start[i] = int(rec["start"])
        pad_frac[i] = float(max(t_bins - L, 0) / t_bins)

        wear_m = win["wear_bin_valid"].fillna(False).astype(bool).to_numpy()
        cgm_m = win["cgm_bin_valid"].fillna(False).astype(bool).to_numpy()
        feats = np.zeros((L, d), dtype=np.float32)
        for j, c in enumerate(feat_cols):
            x = pd.to_numeric(win[c], errors="coerce").to_numpy(dtype=float)
            if c in fill_zero:
                x = np.where(np.isfinite(x), x, 0.0)
            # median/mean fill with train mean for non-fill_zero
            x = np.where(np.isfinite(x), x, feat_mean[j])
            x = (x - feat_mean[j]) / feat_std[j]
            # Zero non-wear for physiological channels; keep ToD always
            # (plan: tod observed whenever bin exists / not pad).
            if c not in ("tod_sin", "tod_cos"):
                x = np.where(wear_m, x, 0.0)
            feats[:, j] = x.astype(np.float32)

        cg = pd.to_numeric(win["cgm"], errors="coerce").to_numpy(dtype=float)
        cg_z = (cg - cgm_mean) / cgm_std
        cg_z = np.where(np.isfinite(cg_z), cg_z, 0.0).astype(np.float32)
        tm = wear_m & cgm_m & bool(row["aux_eligible"])

        X[i, :L, :] = feats
        pad_mask[i, :L] = False
        wear_mask[i, :L] = wear_m
        cgm_arr[i, :L] = cg_z
        traj_mask[i, :L] = tm

        if c1_mat is not None and c1_cols is not None:
            if pid not in df_c1.index:
                raise KeyError(f"pid {pid} missing from C1 frame")
            vals = df_c1.loc[pid, c1_cols]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[0]
            c1_mat[i] = pd.to_numeric(vals, errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float32
            )

    # class weights on train
    y_tr = y[split == "train"]
    counts = np.bincount(y_tr, minlength=4).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (1.0 / counts)
    w = w / w.sum() * 4.0
    class_weights = w.astype(np.float32)

    return GridBundle(
        pids=pids,
        y=y,
        split=split,
        aux_eligible=aux_elig,
        X=X,
        pad_mask=pad_mask,
        wear_mask=wear_mask,
        cgm=cgm_arr,
        traj_mask=traj_mask,
        feature_cols=feat_cols,
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        cgm_mean=cgm_mean,
        cgm_std=cgm_std,
        class_weights=class_weights,
        wear_valid_in_window=wear_in_win,
        pad_frac=pad_frac,
        sub_start_idx=sub_start,
        c1=c1_mat,
        c1_cols=c1_cols,
        w0_cols=w0_cols,
        dropped_pids=dropped,
    )


def subset_counts(bundle: GridBundle) -> dict[str, Any]:
    out = {}
    for s in ("train", "val", "test"):
        m = bundle.split == s
        out[s] = {
            "n": int(m.sum()),
            "n_aux": int((m & bundle.aux_eligible).sum()),
            "label_counts": {
                int(k): int(v)
                for k, v in zip(*np.unique(bundle.y[m], return_counts=True))
            },
            "mean_pad_frac": float(bundle.pad_frac[m].mean()) if m.any() else None,
            "mean_wear_in_window": float(bundle.wear_valid_in_window[m].mean())
            if m.any()
            else None,
            "traj_bins": int(bundle.traj_mask[m].sum()) if m.any() else 0,
        }
    out["n_dropped_tmin"] = len(bundle.dropped_pids or [])
    return out


class GridPersonDataset(Dataset):
    def __init__(self, bundle: GridBundle, split: str):
        self.bundle = bundle
        self.idx = np.where(bundle.split == split)[0]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, Any]:
        j = int(self.idx[i])
        out = {
            "X": torch.from_numpy(self.bundle.X[j]),
            "pad_mask": torch.from_numpy(self.bundle.pad_mask[j]),
            "wear_mask": torch.from_numpy(self.bundle.wear_mask[j]),
            "cgm": torch.from_numpy(self.bundle.cgm[j]),
            "traj_mask": torch.from_numpy(self.bundle.traj_mask[j]),
            "y": torch.tensor(self.bundle.y[j], dtype=torch.long),
            "pid": torch.tensor(self.bundle.pids[j], dtype=torch.long),
        }
        if self.bundle.c1 is not None:
            out["c1"] = torch.from_numpy(self.bundle.c1[j])
        return out


def collate_grid(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = batch[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
