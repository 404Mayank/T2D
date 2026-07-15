"""B3 data: C1 + true CGM teacher cols, soft-label expansion, leakage guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.path_b.b2.data import (
    DENY,
    load_b2_frame,
    split_xy,
    subsample_train_for_smoke,
)

__all__ = [
    "DENY",
    "load_b3_frame",
    "split_xy",
    "subsample_train_for_smoke",
    "assert_student_features",
    "assert_teacher_features",
    "temperature_scale_proba",
    "hard_class_weights",
    "expand_soft_rows",
    "soft_label_diagnostics",
]


def load_b3_frame(repo: Path, cfg: dict[str, Any]):
    """Reuse B2 frame loader (C1 + ytrue_* CGM daymeans + pools)."""
    return load_b2_frame(repo, cfg)


def assert_student_features(feature_cols: list[str], cfg: dict[str, Any]) -> None:
    forbid = set(DENY) | set(cfg["data"]["glu_forbid"]) | set(cfg["data"]["glu_targets"])
    bad = [
        c
        for c in feature_cols
        if c in forbid or c.startswith("ytrue_") or c.startswith("yhat_")
    ]
    if bad:
        raise AssertionError(f"student leakage/deny features: {bad}")


def assert_teacher_features(
    feature_cols: list[str], true_cols: list[str], cfg: dict[str, Any]
) -> None:
    for c in true_cols:
        if c not in feature_cols:
            raise AssertionError(f"teacher missing {c}")
    forbid = set(DENY) | set(cfg["data"]["glu_forbid"])
    bad = [c for c in feature_cols if c in forbid or c.startswith("yhat_")]
    if bad:
        raise AssertionError(f"teacher forbid features: {bad}")


def temperature_scale_proba(p: np.ndarray, temperature: float) -> np.ndarray:
    """p^{1/T} renormalize. Accepts (n, C) or (C,)."""
    t = float(temperature)
    if t <= 0:
        raise ValueError(f"temperature must be > 0, got {t}")
    p = np.asarray(p, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p.reshape(1, -1)
    p = np.clip(p, 1e-12, 1.0)
    # p^{1/T}
    z = np.exp(np.log(p) / t)
    z = z / z.sum(axis=1, keepdims=True)
    return z.ravel() if single else z


def hard_class_weights(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """sklearn-style balanced weights: n / (K * n_k), sum not forced to K."""
    y = np.asarray(y, dtype=np.int64).ravel()
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    n = float(len(y))
    w = np.zeros(n_classes, dtype=np.float64)
    for k in range(n_classes):
        if counts[k] > 0:
            w[k] = n / (n_classes * counts[k])
        else:
            w[k] = 0.0
    return w


def expand_soft_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    pid: np.ndarray,
    *,
    soft_by_pid: dict[int, np.ndarray],
    aux_pids: set[int],
    alpha: float,
    temperature: float,
    class_weights: np.ndarray | None,
    eps: float = 1e-6,
    n_classes: int = 4,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Soft-label row expansion (PLAN_B3 §2.5).

    Returns expanded X, y_label, sample_weight, diagnostics.
    At α=0: one row per person, weight = class_weight[y] (or 1), sum of
    pre-class-weight expansion mass per person = 1.
    """
    a = float(alpha)
    if a < 0 or a > 1:
        raise ValueError(alpha)
    y = np.asarray(y, dtype=np.int64).ravel()
    pid = np.asarray(pid)
    if len(X) != len(y) or len(X) != len(pid):
        raise AssertionError("X/y/pid length mismatch")

    rows_X: list[pd.Series] = []
    rows_y: list[int] = []
    rows_w: list[float] = []
    soft_mass_non_aux = 0.0
    n_soft_persons = 0
    per_person_mass: list[float] = []
    per_person_n_rows: list[int] = []

    for i in range(len(y)):
        p_i = int(pid[i])
        y_i = int(y[i])
        is_aux = p_i in aux_pids
        if is_aux and p_i in soft_by_pid and a > 0:
            p_soft = temperature_scale_proba(soft_by_pid[p_i], temperature)
            w = (1.0 - a) * np.eye(n_classes)[y_i] + a * p_soft
            n_soft_persons += 1
        elif is_aux and p_i in soft_by_pid and a == 0:
            w = np.eye(n_classes)[y_i]
        else:
            # non-aux or missing soft → hard only
            if not is_aux and a > 0:
                # ensure zero soft mass from teacher
                pass
            w = np.eye(n_classes)[y_i]
            if not is_aux:
                soft_mass_non_aux += 0.0

        mass = float(w.sum())
        if abs(mass - 1.0) > 1e-6:
            w = w / mass
            mass = 1.0
        per_person_mass.append(mass)

        cw = 1.0
        if class_weights is not None:
            cw = float(class_weights[y_i])

        n_emit = 0
        for k in range(n_classes):
            wk = float(w[k])
            if wk > eps:
                rows_X.append(X.iloc[i])
                rows_y.append(k)
                rows_w.append(wk * cw)
                n_emit += 1
        if n_emit == 0:
            # fallback hard
            rows_X.append(X.iloc[i])
            rows_y.append(y_i)
            rows_w.append(cw)
            n_emit = 1
        per_person_n_rows.append(n_emit)

    X_exp = pd.DataFrame(rows_X).reset_index(drop=True)
    y_exp = np.asarray(rows_y, dtype=np.int64)
    w_exp = np.asarray(rows_w, dtype=np.float64)

    diag = {
        "alpha": a,
        "temperature": float(temperature),
        "n_persons": int(len(y)),
        "n_expanded_rows": int(len(y_exp)),
        "n_soft_persons": int(n_soft_persons),
        "soft_mass_non_aux": float(soft_mass_non_aux),
        "mean_rows_per_person": float(np.mean(per_person_n_rows)),
        "max_abs_mass_minus_1": float(np.max(np.abs(np.asarray(per_person_mass) - 1.0))),
        "alpha0_single_row": bool(a == 0 and all(n == 1 for n in per_person_n_rows)),
        "alpha0_n_rows_eq_n_persons": bool(a == 0 and len(y_exp) == len(y)),
    }
    return X_exp, y_exp, w_exp, diag


def soft_label_diagnostics(
    soft_by_pid: dict[int, np.ndarray],
    *,
    temperature: float,
) -> dict[str, Any]:
    if not soft_by_pid:
        return {"n": 0}
    P = np.stack([soft_by_pid[k] for k in soft_by_pid], axis=0)
    Pt = temperature_scale_proba(P, temperature)
    ent = -(Pt * np.log(np.clip(Pt, 1e-12, 1.0))).sum(axis=1)
    mx = Pt.max(axis=1)
    return {
        "n": int(len(P)),
        "temperature": float(temperature),
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "maxprob_mean": float(mx.mean()),
        "maxprob_std": float(mx.std()),
        "raw_maxprob_mean": float(P.max(axis=1).mean()),
    }
