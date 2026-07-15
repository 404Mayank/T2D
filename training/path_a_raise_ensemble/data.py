"""Load C1 feature matrix + assert frozen parent (read-only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import (
    load_watch_onboarding_mood,
    make_block_splits,
    resolve_mood_cols,
)
from training.path_a_watch.data import SplitData
from training.path_a_watch.metrics import full_report, macro_ovr_auc
from training.path_a_watch.models import predict_proba


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def load_blocks_data_cfg(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    blocks_cfg_path = _resolve(repo, cfg["paths"]["blocks_config"])
    with blocks_cfg_path.open() as f:
        blocks = yaml.safe_load(f)
    return blocks["data"]


@dataclass(frozen=True)
class ParentC1:
    run_id: str
    art: Path
    feature_cols: list[str]
    feature_hash: str
    family: str
    test_macro_ovr_auc: float
    test_binary_auc: float
    test_macro_auprc: float
    lgbm_params: dict[str, Any]
    cat_params: dict[str, Any]
    lgbm_best_iteration: int
    cat_best_iteration: int
    lgbm_device: str
    model_path: Path
    proba_test: np.ndarray
    metrics_test: dict[str, Any]
    assert_record: dict[str, Any]


def assert_and_load_parent(
    repo: Path,
    cfg: dict[str, Any],
    splits: SplitData,
    feature_cols: list[str],
) -> ParentC1:
    """Bit-match frozen C1 CatBoost on test; pin feature contract + LGBM device."""
    pin = cfg["parent_c1"]
    run_id = pin["run_id"]
    art = _resolve(repo, cfg["paths"]["parent_artifacts_root"]) / run_id
    if not art.is_dir():
        raise FileNotFoundError(f"parent C1 artifacts missing: {art}")

    freeze = json.loads((art / "selected_model.json").read_text())
    feats_json = json.loads((art / "features.json").read_text())
    metrics = json.loads((art / "metrics_test.json").read_text())
    lgbm_pack = json.loads((art / "best_params_lgbm.json").read_text())
    cat_pack = json.loads((art / "best_params_catboost.json").read_text())

    freeze_cols = list(freeze["feature_cols"])
    checks: dict[str, Any] = {"run_id": run_id, "art": str(art)}

    if freeze.get("feature_set") != pin["feature_set"]:
        raise AssertionError(
            f"feature_set {freeze.get('feature_set')!r} != pin {pin['feature_set']!r}"
        )
    checks["feature_set"] = freeze["feature_set"]

    if len(freeze_cols) != int(pin["n_features"]):
        raise AssertionError(
            f"n_features {len(freeze_cols)} != pin {pin['n_features']}"
        )
    if freeze_cols != list(feature_cols):
        raise AssertionError("loaded feature_cols order/content != freeze feature_cols")
    checks["feature_cols_match"] = True

    fh = freeze.get("feature_hash") or feats_json.get("feature_hash")
    if fh != pin["feature_hash"]:
        raise AssertionError(f"feature_hash {fh} != pin {pin['feature_hash']}")
    checks["feature_hash"] = fh

    raw = metrics["selected_raw"]
    for k, pin_k in [
        ("macro_ovr_auc", "test_macro_ovr_auc"),
        ("binary_auc", "test_binary_auc"),
        ("macro_auprc", "test_macro_auprc"),
    ]:
        v = float(raw[k])
        if abs(v - float(pin[pin_k])) > 1e-9:
            raise AssertionError(f"parent metrics {k}: artifact {v} != pin {pin[pin_k]}")
        checks[pin_k] = v

    fam = metrics.get("selected_family") or freeze.get("family")
    if fam != pin["family_selected"]:
        raise AssertionError(f"family {fam} != pin {pin['family_selected']}")
    checks["family"] = fam

    lgbm_dev = lgbm_pack.get("device")
    if lgbm_dev != pin["lgbm_device"]:
        raise AssertionError(
            f"lgbm freeze device {lgbm_dev!r} != pin {pin['lgbm_device']!r}"
        )
    if cfg["run"].get("lgbm_device") != pin["lgbm_device"]:
        raise AssertionError(
            f"config run.lgbm_device {cfg['run'].get('lgbm_device')!r} "
            f"!= pin {pin['lgbm_device']!r}"
        )
    checks["lgbm_device"] = lgbm_dev

    model_path = art / "models" / "selected.joblib"
    model = joblib.load(model_path)
    proba = predict_proba(model, splits.X_test[feature_cols])
    re = full_report(splits.y_test, proba, tag="parent_recompute")
    if abs(re["macro_ovr_auc"] - float(pin["test_macro_ovr_auc"])) > 1e-9:
        raise AssertionError(
            f"B0 recompute auc={re['macro_ovr_auc']} != pin {pin['test_macro_ovr_auc']}"
        )
    if abs(re["binary_auc"] - float(pin["test_binary_auc"])) > 1e-9:
        raise AssertionError(
            f"B0 recompute bin={re['binary_auc']} != pin {pin['test_binary_auc']}"
        )
    checks["b0_bitmatch"] = True
    checks["b0_recompute"] = {
        "macro_ovr_auc": re["macro_ovr_auc"],
        "binary_auc": re["binary_auc"],
    }

    return ParentC1(
        run_id=run_id,
        art=art,
        feature_cols=freeze_cols,
        feature_hash=str(fh),
        family=str(fam),
        test_macro_ovr_auc=float(pin["test_macro_ovr_auc"]),
        test_binary_auc=float(pin["test_binary_auc"]),
        test_macro_auprc=float(pin["test_macro_auprc"]),
        lgbm_params=dict(lgbm_pack["params"]),
        cat_params=dict(cat_pack["params"]),
        lgbm_best_iteration=int(
            lgbm_pack.get("best_iteration", pin["lgbm_best_iteration"])
        ),
        cat_best_iteration=int(
            cat_pack.get("best_iteration", pin["catboost_best_iteration"])
        ),
        lgbm_device=str(lgbm_dev),
        model_path=model_path,
        proba_test=proba,
        metrics_test=metrics,
        assert_record=checks,
    )


def load_c1_matrix(
    repo: Path,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], SplitData, dict[str, Any]]:
    """Return df, feature_cols, splits, load_meta."""
    blocks_data = load_blocks_data_cfg(repo, cfg)
    mood_cols = resolve_mood_cols(blocks_data, "scores")
    df, watch_cols, onboard_cols, mood_feat, feature_cols = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=list(blocks_data["onboarding_keep"]),
        mood_cols=mood_cols,
        expected_n=int(cfg["data"]["expected_n"]),
        id_col=str(cfg["data"]["id_col"]),
    )
    if list(onboard_cols) != list(blocks_data["onboarding_keep"]):
        raise AssertionError("onboarding keep list drift")
    if len(watch_cols) != 30:
        raise AssertionError(f"expected 30 watch cols, got {len(watch_cols)}")
    if mood_feat != list(blocks_data["mood_scores"]):
        raise AssertionError(f"mood scores drift: {mood_feat}")

    splits = make_block_splits(df, feature_cols, feature_set="1c_scores")
    meta = {
        "n_watch": len(watch_cols),
        "n_onboarding": len(onboard_cols),
        "n_mood": len(mood_feat),
        "n_total": len(feature_cols),
        "mood_cols": list(mood_feat),
        "n_train": int(len(splits.y_train)),
        "n_val": int(len(splits.y_val)),
        "n_test": int(len(splits.y_test)),
    }
    # sanity on split sizes
    if meta["n_train"] != 1277 or meta["n_val"] != 270 or meta["n_test"] != 277:
        raise AssertionError(f"unexpected split sizes: {meta}")
    return df, list(feature_cols), splits, meta


def require_lgbm_gpu(cfg: dict[str, Any], *, quick: bool = False) -> str:
    """Return 'gpu' or raise. Quick smoke may fall back only if explicitly allowed."""
    from training.path_a_watch.models import resolve_lgbm_device

    requested = str(cfg["run"]["lgbm_device"])
    if requested != "gpu":
        raise AssertionError(
            f"run.lgbm_device must be 'gpu' (C1 freeze), got {requested!r}"
        )
    resolved = resolve_lgbm_device("gpu")
    if resolved != "gpu":
        raise RuntimeError(
            "LGBM GPU required (C1 freeze device=gpu) but probe failed. "
            "Full raise needs GPU; do not silently train LGBM-CPU under GPU params."
        )
    return "gpu"
