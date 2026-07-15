"""Val permutation importance for MLP (macro-OVR drop)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from training.path_a_watch.metrics import macro_ovr_auc


def permutation_importance_mlp(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 5,
    seed: int = 42,
    max_features: int | None = None,
) -> dict[str, Any]:
    """Permute each column on val; report mean AUC drop."""
    rng = np.random.default_rng(seed)
    base = float(macro_ovr_auc(y, predict_fn(X)))
    cols = list(range(X.shape[1]))
    if max_features is not None and max_features < len(cols):
        # still permute all for c3; max_features reserved
        pass
    drops: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(float(macro_ovr_auc(y, predict_fn(Xp))))
        drops[name] = float(base - float(np.mean(scores)))

    mean_drop = float(np.mean(list(drops.values()))) if drops else 0.0
    n_pos = int(sum(1 for v in drops.values() if v > 0))
    # stable: mean drop > 0 and at least a few positive features
    stable = bool(mean_drop > 0 and n_pos >= max(3, len(feature_names) // 10))
    ranked = sorted(drops.items(), key=lambda kv: -kv[1])
    return {
        "baseline_val_auc": base,
        "mean_perm_auc_drop": mean_drop,
        "n_features_positive_drop": n_pos,
        "perm_stable": stable,
        "per_feature_drop": drops,
        "top15": [{k: v} for k, v in ranked[:15]],
        "n_repeats": n_repeats,
    }
