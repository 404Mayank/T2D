"""Multiclass probability calibration (val-fit, test-apply)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Method = Literal["sigmoid", "isotonic", "none"]


@dataclass
class MulticlassCalibrator:
    method: Method
    n_classes: int
    models: list[Any]
    fitted: bool = False

    @classmethod
    def create(cls, method: Method, n_classes: int = 4) -> "MulticlassCalibrator":
        return cls(method=method, n_classes=n_classes, models=[])

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "MulticlassCalibrator":
        p = np.asarray(proba, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        if p.ndim != 2 or p.shape[1] != self.n_classes:
            raise ValueError(f"proba shape {p.shape}")
        self.models = []
        if self.method == "none":
            self.fitted = True
            return self
        for k in range(self.n_classes):
            y_bin = (y == k).astype(int)
            x = p[:, k]
            if self.method == "sigmoid":
                # Platt: logistic regression on scalar score
                # Need both classes present
                if y_bin.min() == y_bin.max():
                    self.models.append(None)
                    continue
                lr = LogisticRegression(solver="lbfgs", max_iter=1000)
                lr.fit(x.reshape(-1, 1), y_bin)
                self.models.append(lr)
            elif self.method == "isotonic":
                if y_bin.min() == y_bin.max():
                    self.models.append(None)
                    continue
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x, y_bin)
                self.models.append(iso)
            else:
                raise ValueError(self.method)
        self.fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("calibrator not fit")
        p = np.asarray(proba, dtype=float)
        if self.method == "none":
            return _renorm(p)
        out = np.zeros_like(p, dtype=float)
        for k, model in enumerate(self.models):
            x = p[:, k]
            if model is None:
                out[:, k] = x
            elif self.method == "sigmoid":
                out[:, k] = model.predict_proba(x.reshape(-1, 1))[:, 1]
            else:
                out[:, k] = model.predict(x)
        return _renorm(out)

    def class_support(self, y: np.ndarray) -> dict[int, int]:
        y = np.asarray(y).astype(int).ravel()
        return {k: int((y == k).sum()) for k in range(self.n_classes)}


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return p / s


def fit_calibrators(
    proba_val: np.ndarray,
    y_val: np.ndarray,
    *,
    default_method: Method = "sigmoid",
    secondary_method: Method = "isotonic",
    min_pos_for_isotonic: int = 30,
) -> dict[str, Any]:
    y = np.asarray(y_val).astype(int).ravel()
    support = {k: int((y == k).sum()) for k in range(proba_val.shape[1])}
    primary = MulticlassCalibrator.create(default_method).fit(proba_val, y)
    secondary = MulticlassCalibrator.create(secondary_method).fit(proba_val, y)
    iso_ok = all(support[k] >= min_pos_for_isotonic for k in support)
    return {
        "primary": primary,
        "secondary": secondary,
        "support": support,
        "isotonic_reliable": iso_ok,
        "min_pos_for_isotonic": min_pos_for_isotonic,
        "default_method": default_method,
        "secondary_method": secondary_method,
    }
