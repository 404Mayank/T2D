"""Phase A wrap experiments — minimal / PAID / severity / binary.

See PLAN_A_WRAP.md and PLAN_A_WRAP_IMPL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from training.path_a_blocks.data_blocks import (
    block_tags_wrap,
    load_c1_plus_comorb,
    load_watch_onboarding,
    load_watch_onboarding_mood,
    make_block_splits,
    null_report,
)
from training.path_a_blocks.diagnostics import (
    bootstrap_ci,
    bootstrap_ci_binary,
    paired_delta_bootstrap,
    paired_delta_bootstrap_binary,
)
from training.path_a_watch.calibrate import fit_calibrators
from training.path_a_watch.evaluate import write_json
from training.path_a_watch.explain import (
    permutation_on_val,
    permutation_on_val_binary,
    shap_summary,
)
from training.path_a_watch.hpo import (
    pick_family,
    pick_family_binary,
    tune_catboost,
    tune_catboost_binary,
    tune_lightgbm,
    tune_lightgbm_binary,
)
from training.path_a_watch.metrics import binary_report, full_report, macro_ovr_auc
from training.path_a_watch.models import (
    predict_proba,
    predict_proba_positive,
    resolve_lgbm_device,
)

EXPERIMENTS = (
    "paid_only",
    "ces_only",
    "minimal_s",
    "minimal_m",
    "watch_mood",
    "severity",
    "clinical_upper",
    "bin_watch",
    "bin_c1",
    "bin_min_s",
    "bin_severity",
)

BINARY_EXPS = {"bin_watch", "bin_c1", "bin_min_s", "bin_severity"}

# Fixed order for --all
ALL_ORDER = list(EXPERIMENTS)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _git_hash(repo: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _feature_hash(cols: list[str]) -> str:
    return hashlib.sha256(",".join(cols).encode()).hexdigest()[:16]


def _strip(pack: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pack.items() if k not in ("model", "val_report")}


def assert_parent_1a(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["parent_1a_run_id"]
    path = repo / cfg["paths"]["artifacts_root"] / run_id / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["parent_1a_reference"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"parent_1a {k}: config {ref[k]} != artifact {v}")
    freeze = json.loads(
        (repo / cfg["paths"]["artifacts_root"] / run_id / "selected_model.json").read_text()
    )
    model_path = repo / cfg["paths"]["artifacts_root"] / run_id / "models" / "selected.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        **checks,
        "family": m.get("selected_family"),
        "source_path": str(path),
        "feature_cols": list(freeze["feature_cols"]),
        "model_path": str(model_path),
        "run_id": run_id,
    }


def assert_parent_c1(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["parent_c1_run_id"]
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    path = art / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["parent_c1_reference"]
    checks = {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
    }
    for k, v in checks.items():
        if abs(float(ref[k]) - v) > 1e-9:
            raise AssertionError(f"parent_c1 {k}: config {ref[k]} != artifact {v}")
    freeze = json.loads((art / "selected_model.json").read_text())
    model_path = art / "models" / "selected.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        **checks,
        "family": m.get("selected_family"),
        "source_path": str(path),
        "feature_cols": list(freeze["feature_cols"]),
        "model_path": str(model_path),
        "run_id": run_id,
        "freeze": freeze,
    }


def assert_floor(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    run_id = cfg["paths"]["floor_run_id"]
    art = repo / cfg["paths"]["floor_artifacts"] / run_id
    path = art / "metrics_test.json"
    if not path.exists():
        raise FileNotFoundError(path)
    m = json.loads(path.read_text())
    raw = m["selected_raw"]
    ref = cfg["floor_reference"]
    for k_art, k_ref in (
        ("macro_ovr_auc", "test_macro_ovr_auc"),
        ("binary_auc", "test_binary_auc"),
        ("macro_auprc", "test_macro_auprc"),
    ):
        if abs(float(raw[k_art]) - float(ref[k_ref])) > 1e-9:
            raise AssertionError(
                f"floor {k_art}: config {ref[k_ref]} != artifact {raw[k_art]}"
            )
    # model filename may be selected_full_green.joblib
    model_name = cfg["paths"].get("floor_model", "selected.joblib")
    model_path = art / "models" / model_name
    if not model_path.exists():
        alt = art / "models" / "selected.joblib"
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(model_path)
    freeze_path = art / "selected_model.json"
    feature_cols = None
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        feature_cols = list(freeze.get("feature_cols") or [])
    return {
        "test_macro_ovr_auc": float(raw["macro_ovr_auc"]),
        "test_binary_auc": float(raw["binary_auc"]),
        "test_macro_auprc": float(raw["macro_auprc"]),
        "model_path": str(model_path),
        "feature_cols": feature_cols,
        "run_id": run_id,
    }


def load_wrap_ranks(repo: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = repo / cfg["paths"]["wrap_ranks"]
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run: python -m training.path_a_blocks.build_minimal_ranks"
        )
    return json.loads(path.read_text())


def load_parent_c1_proba(
    repo: Path,
    cfg: dict[str, Any],
    c1: dict[str, Any],
    test_pids: np.ndarray,
    *,
    recompute_tol: float = 1e-6,
) -> np.ndarray:
    """Independently load full C1 matrix and predict proba aligned to test_pids.

    Required when experiment feature matrix is not a superset of C1 cols (O1).
    CatBoost is positional — reindex to freeze feature_cols (O3).
    """
    mood_cols = list(cfg["data"]["mood_scores"])
    df, watch_cols, onboard_cols, mood_feat, feature_cols_full = load_watch_onboarding_mood(
        repo,
        watch_green=cfg["paths"]["watch_green"],
        onboarding=cfg["paths"]["onboarding"],
        mood=cfg["paths"]["mood"],
        pool_masks=cfg["paths"]["pool_masks"],
        onboarding_keep=cfg["data"]["onboarding_keep"],
        mood_cols=mood_cols,
        expected_n=int(cfg["data"]["expected_n"]),
    )
    c1_feat = list(c1["feature_cols"])
    # sanity: freeze list should match loader
    if set(c1_feat) != set(feature_cols_full):
        raise AssertionError(
            f"C1 freeze features != loader C1 set "
            f"(only_freeze={set(c1_feat)-set(feature_cols_full)} "
            f"only_loader={set(feature_cols_full)-set(c1_feat)})"
        )
    X_all = df.set_index("person_id")[c1_feat]
    # align to test pid order
    missing = [p for p in test_pids if p not in X_all.index]
    if missing:
        raise AssertionError(f"{len(missing)} test pids missing from C1 matrix")
    X_test = X_all.loc[list(test_pids)].reindex(columns=c1_feat)
    if list(X_test.columns) != c1_feat:
        raise AssertionError("C1 feature column order mismatch after reindex")

    model = joblib.load(c1["model_path"])
    proba = predict_proba(model, X_test)
    # bit-match against stored test metrics using labels from df
    y_test = df.set_index("person_id").loc[list(test_pids), "label"].to_numpy(dtype=np.int64)
    recomp = macro_ovr_auc(y_test, proba)
    if abs(recomp - c1["test_macro_ovr_auc"]) > recompute_tol:
        raise AssertionError(
            f"C1 recompute auc={recomp} != artifact {c1['test_macro_ovr_auc']} "
            f"(tol={recompute_tol})"
        )
    return proba


def _build_matrix(
    exp: str,
    repo: Path,
    cfg: dict[str, Any],
    ranks: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return df, feature_cols, tags, block meta."""
    paths = cfg["paths"]
    data = cfg["data"]
    onboard_keep = list(data["onboarding_keep"])
    mood_scores = list(data["mood_scores"])

    if exp in ("paid_only", "bin_c1") or exp in ("minimal_s", "minimal_m", "bin_min_s"):
        pass  # handled below

    if exp == "paid_only":
        df, w, o, m, _ = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=["paidscore"],
            expected_n=int(data["expected_n"]),
        )
        feat = list(w) + list(o) + ["paidscore"]
        tags = block_tags_wrap(w, o, ["paidscore"])
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": o,
            "mood_cols": ["paidscore"],
            "comorb_cols": [],
        }

    if exp == "ces_only":
        df, w, o, m, _ = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=["cestl"],
            expected_n=int(data["expected_n"]),
        )
        feat = list(w) + list(o) + ["cestl"]
        tags = block_tags_wrap(w, o, ["cestl"])
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": o,
            "mood_cols": ["cestl"],
            "comorb_cols": [],
        }

    if exp in ("minimal_s", "minimal_m", "bin_min_s"):
        if ranks is None:
            raise RuntimeError("ranks required")
        key = "minimal_S" if exp in ("minimal_s", "bin_min_s") else "minimal_M"
        want = list(ranks[key])
        df, w, o, m, full = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=mood_scores,
            expected_n=int(data["expected_n"]),
        )
        missing = [c for c in want if c not in full and c not in df.columns]
        if missing:
            raise AssertionError(f"minimal features missing from C1 matrix: {missing}")
        # ensure all present
        for c in want:
            if c not in df.columns:
                raise AssertionError(f"missing {c}")
        tags = {}
        for c in want:
            if c in w:
                tags[c] = "watch_green"
            elif c in o:
                tags[c] = "onboarding"
            elif c in m:
                tags[c] = "mood"
            else:
                tags[c] = "other"
        return {
            "df": df,
            "feature_cols": want,
            "tags": tags,
            "watch_cols": [c for c in want if c in w],
            "onboard_cols": [c for c in want if c in o],
            "mood_cols": [c for c in want if c in m],
            "comorb_cols": [],
        }

    if exp == "watch_mood":
        df, w, o, m, _ = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=["paidscore"],
            expected_n=int(data["expected_n"]),
        )
        feat = list(w) + ["paidscore"]
        tags = block_tags_wrap(w, None, ["paidscore"])
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": [],
            "mood_cols": ["paidscore"],
            "comorb_cols": [],
        }

    if exp in ("severity", "bin_severity"):
        bins = list(data["comorbidity_complications"])  # rnl + circ
        df, w, o, m, comb, feat = load_c1_plus_comorb(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            comorbidity=paths["comorbidity"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=mood_scores,
            comorbidity_binaries=bins,
            expected_n=int(data["expected_n"]),
        )
        if any(c.startswith("comorb_count") for c in feat):
            raise AssertionError("count feature leaked into severity matrix")
        tags = block_tags_wrap(w, o, m, comb)
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": o,
            "mood_cols": m,
            "comorb_cols": comb,
        }

    if exp == "clinical_upper":
        bins = ["mhoccur_hbp", "mhoccur_clsh"] + list(data["comorbidity_complications"])
        # dedupe preserve order
        seen: list[str] = []
        for c in bins:
            if c not in seen:
                seen.append(c)
        df, w, o, m, comb, feat = load_c1_plus_comorb(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            comorbidity=paths["comorbidity"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=mood_scores,
            comorbidity_binaries=seen,
            expected_n=int(data["expected_n"]),
        )
        if any(c.startswith("comorb_count") for c in feat):
            raise AssertionError("count feature leaked into clinical_upper matrix")
        tags = block_tags_wrap(w, o, m, comb)
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": o,
            "mood_cols": m,
            "comorb_cols": comb,
        }

    if exp == "bin_watch":
        df, w, o, feat = load_watch_onboarding(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            expected_n=int(data["expected_n"]),
        )
        # watch only
        feat = list(w)
        tags = block_tags_wrap(w)
        return {
            "df": df,
            "feature_cols": feat,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": [],
            "mood_cols": [],
            "comorb_cols": [],
        }

    if exp == "bin_c1":
        df, w, o, m, full = load_watch_onboarding_mood(
            repo,
            watch_green=paths["watch_green"],
            onboarding=paths["onboarding"],
            mood=paths["mood"],
            pool_masks=paths["pool_masks"],
            onboarding_keep=onboard_keep,
            mood_cols=mood_scores,
            expected_n=int(data["expected_n"]),
        )
        tags = block_tags_wrap(w, o, m)
        return {
            "df": df,
            "feature_cols": full,
            "tags": tags,
            "watch_cols": w,
            "onboard_cols": o,
            "mood_cols": m,
            "comorb_cols": [],
        }

    raise ValueError(f"unknown exp {exp}")


def _retention(
    auc_new: float,
    bin_new: float,
    auc_c1: float,
    bin_c1: float,
    boot_d: dict[str, Any],
) -> dict[str, Any]:
    d_auc = auc_new - auc_c1
    d_bin = bin_new - bin_c1
    ci = boot_d.get("delta_macro_ovr_auc", {})
    hi = ci.get("hi")
    fail_ci_entirely_below = hi is not None and float(hi) < 0
    retain = (d_auc >= -0.01) and (d_bin >= -0.015) and (not fail_ci_entirely_below)
    return {
        "delta_macro_ovr_auc": d_auc,
        "delta_binary_auc": d_bin,
        "point_auc_ok": d_auc >= -0.01,
        "point_bin_ok": d_bin >= -0.015,
        "bootstrap_auc_ci_entirely_below_0": fail_ci_entirely_below,
        "bootstrap_auc_ci_includes_0": (
            ci.get("lo") is not None
            and ci.get("hi") is not None
            and float(ci["lo"]) <= 0 <= float(ci["hi"])
        ),
        "retain": bool(retain),
        "rule": "ΔAUC>=-0.01 AND Δbin>=-0.015 AND not (ΔAUC CI entirely <0)",
    }


def run_one(
    exp: str,
    *,
    cfg: dict[str, Any],
    repo: Path,
    run_id: str,
    quick: bool,
    n_trials: int | None,
    skip_shap: bool,
) -> int:
    art = repo / cfg["paths"]["artifacts_root"] / run_id
    art.mkdir(parents=True, exist_ok=True)

    # O: refuse overwrite BEFORE HPO
    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError(f"refuse overwrite existing freeze/metrics under {art}")

    if quick:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = 2
        skip_shap = True
    if n_trials is not None:
        cfg = dict(cfg)
        cfg["run"] = dict(cfg["run"])
        cfg["run"]["n_trials"] = int(n_trials)

    seed = int(cfg["run"]["seed"])
    np.random.seed(seed)
    t0 = time.time()
    logs: list[str] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logs.append(line)

    log(f"run_id={run_id} exp={exp}")
    log(f"DRI_PRIME={os.environ.get('DRI_PRIME', 'unset')}")

    floor = assert_floor(repo, cfg)
    parent_1a = assert_parent_1a(repo, cfg)
    parent_c1 = assert_parent_c1(repo, cfg)
    log(
        f"parents ok floor_auc={floor['test_macro_ovr_auc']:.4f} "
        f"1a={parent_1a['test_macro_ovr_auc']:.4f} c1={parent_c1['test_macro_ovr_auc']:.4f}"
    )

    ranks = None
    if exp in ("minimal_s", "minimal_m", "bin_min_s"):
        ranks = load_wrap_ranks(repo, cfg)
        log(f"ranks loaded S={ranks['minimal_S'][:3]}... M n={len(ranks['minimal_M'])}")

    mat = _build_matrix(exp, repo, cfg, ranks)
    df = mat["df"]
    feature_cols = list(mat["feature_cols"])
    tags = mat["tags"]
    if any(c.startswith("comorb_count") for c in feature_cols):
        raise AssertionError("comorb_count must not appear in wrap features")

    write_json(
        art / "features.json",
        {
            "exp": exp,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "block_tags": tags,
            "n_watch": len(mat["watch_cols"]),
            "n_onboarding": len(mat["onboard_cols"]),
            "n_mood": len(mat["mood_cols"]),
            "n_comorb": len(mat["comorb_cols"]),
            "mood_cols": mat["mood_cols"],
            "comorb_cols": mat["comorb_cols"],
            "nulls": null_report(df, feature_cols),
            "feature_hash": _feature_hash(feature_cols),
            "task": "binary" if exp in BINARY_EXPS else "multiclass",
        },
    )
    log(f"features n={len(feature_cols)} cols_head={feature_cols[:5]}")

    splits = make_block_splits(df, feature_cols, feature_set=f"wrap_{exp}")
    lgbm_device = resolve_lgbm_device(cfg["run"].get("lgbm_device", "auto"))
    log(f"lgbm device={lgbm_device} n_trials={cfg['run']['n_trials']}")

    is_binary = exp in BINARY_EXPS

    if is_binary:
        return _run_binary(
            exp=exp,
            cfg=cfg,
            repo=repo,
            art=art,
            run_id=run_id,
            splits=splits,
            feature_cols=feature_cols,
            tags=tags,
            mat=mat,
            floor=floor,
            parent_1a=parent_1a,
            parent_c1=parent_c1,
            lgbm_device=lgbm_device,
            seed=seed,
            skip_shap=skip_shap,
            log=log,
            logs=logs,
            t0=t0,
        )

    return _run_multiclass(
        exp=exp,
        cfg=cfg,
        repo=repo,
        art=art,
        run_id=run_id,
        splits=splits,
        feature_cols=feature_cols,
        tags=tags,
        mat=mat,
        floor=floor,
        parent_1a=parent_1a,
        parent_c1=parent_c1,
        lgbm_device=lgbm_device,
        seed=seed,
        skip_shap=skip_shap,
        log=log,
        logs=logs,
        t0=t0,
    )


def _run_multiclass(
    *,
    exp: str,
    cfg: dict[str, Any],
    repo: Path,
    art: Path,
    run_id: str,
    splits: Any,
    feature_cols: list[str],
    tags: dict[str, str],
    mat: dict[str, Any],
    floor: dict[str, Any],
    parent_1a: dict[str, Any],
    parent_c1: dict[str, Any],
    lgbm_device: str,
    seed: int,
    skip_shap: bool,
    log: Any,
    logs: list[str],
    t0: float,
) -> int:
    log("HPO LGBM multiclass")
    lgbm_pack = tune_lightgbm(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        cfg,
        device=lgbm_device,
    )
    log(f"LGBM val_auc={lgbm_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log("HPO CatBoost multiclass")
    cat_pack = tune_catboost(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val, cfg
    )
    log(f"Cat val_auc={cat_pack['val_macro_ovr_auc']:.4f}")
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family([lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"]))
    log(f"VAL-SELECT {selected['family']} val_auc={selected['val_macro_ovr_auc']:.4f}")

    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError("refuse overwrite")

    freeze = {
        "run_id": run_id,
        "phase": "A_wrap",
        "exp": exp,
        "task": "multiclass",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_macro_ovr_auc": selected["val_macro_ovr_auc"],
        "val_macro_auprc": selected["val_macro_auprc"],
        "feature_cols": feature_cols,
        "feature_hash": _feature_hash(feature_cols),
        "parent_c1_run_id": cfg["paths"]["parent_c1_run_id"],
        "n_trials": int(cfg["run"]["n_trials"]),
        "selected_device": selected.get("device"),
        "selected_boosting_type": selected.get("boosting_type"),
    }
    write_json(art / "selected_model.json", freeze)
    log(f"FREEZE {art / 'selected_model.json'}")

    models_dir = art / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(selected["model"], models_dir / "selected.joblib")
    joblib.dump(lgbm_pack["model"], models_dir / "lgbm.joblib")
    joblib.dump(cat_pack["model"], models_dir / "catboost.joblib")

    proba_val = predict_proba(selected["model"], splits.X_val)
    cal = fit_calibrators(
        proba_val,
        splits.y_val,
        default_method=cfg["calibration"]["default_method"],
        secondary_method=cfg["calibration"]["secondary_method"],
        min_pos_for_isotonic=int(cfg["calibration"]["min_pos_for_isotonic"]),
    )
    proba_test = predict_proba(selected["model"], splits.X_test)
    tr = full_report(splits.y_test, proba_test, tag=f"wrap_{exp}_test")

    # Parent C1 proba for paired bootstrap (independent load when needed)
    log("recompute parent C1 proba (aligned)")
    proba_c1 = load_parent_c1_proba(repo, cfg, parent_c1, splits.person_id_test)

    d_auc_c1 = tr["macro_ovr_auc"] - parent_c1["test_macro_ovr_auc"]
    d_bin_c1 = tr["binary_auc"] - parent_c1["test_binary_auc"]
    d_auc_1a = tr["macro_ovr_auc"] - parent_1a["test_macro_ovr_auc"]
    d_bin_1a = tr["binary_auc"] - parent_1a["test_binary_auc"]
    d_floor = tr["macro_ovr_auc"] - floor["test_macro_ovr_auc"]

    boot = bootstrap_ci(
        splits.y_test,
        proba_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed,
    )
    boot_d_c1 = paired_delta_bootstrap(
        splits.y_test,
        proba_test,
        proba_c1,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed + 11,
    )

    retention = None
    if exp in ("minimal_s", "minimal_m"):
        retention = _retention(
            tr["macro_ovr_auc"],
            tr["binary_auc"],
            parent_c1["test_macro_ovr_auc"],
            parent_c1["test_binary_auc"],
            boot_d_c1,
        )
        log(f"retention={retention}")

    metrics_test: dict[str, Any] = {
        "phase": "A_wrap",
        "exp": exp,
        "task": "multiclass",
        "selected_family": selected["family"],
        "selected_raw": tr,
        "selected_cal_sigmoid": full_report(
            splits.y_test, cal["primary"].transform(proba_test), tag=f"wrap_{exp}_cal"
        ),
        "delta_vs_c1": {
            "parent_run": cfg["paths"]["parent_c1_run_id"],
            "delta_macro_ovr_auc": d_auc_c1,
            "delta_binary_auc": d_bin_c1,
            "delta_macro_auprc": tr["macro_auprc"] - parent_c1["test_macro_auprc"],
            "bootstrap_paired_delta": boot_d_c1,
        },
        "delta_vs_1a": {
            "parent_run": cfg["paths"]["parent_1a_run_id"],
            "delta_macro_ovr_auc": d_auc_1a,
            "delta_binary_auc": d_bin_1a,
        },
        "delta_vs_watch_floor": {
            "delta_macro_ovr_auc": d_floor,
            "delta_binary_auc": tr["binary_auc"] - floor["test_binary_auc"],
        },
        "bootstrap_test": boot,
        "retention": retention,
    }
    write_json(
        art / "metrics_val.json",
        {
            "selected_raw": full_report(splits.y_val, proba_val, tag=f"wrap_{exp}_val"),
            "lightgbm": lgbm_pack.get("val_report"),
            "catboost": cat_pack.get("val_report"),
        },
    )
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {"test_scored": True, "test_scored_at": datetime.now(timezone.utc).isoformat()},
    )
    log(
        f"TEST auc={tr['macro_ovr_auc']:.4f} bin={tr['binary_auc']:.4f} "
        f"Δc1={d_auc_c1:+.4f} Δbin={d_bin_c1:+.4f}"
    )

    explain: dict[str, Any] = {}
    if not skip_shap:
        log("SHAP + permutation")
        try:
            explain["shap"] = shap_summary(
                selected["model"],
                splits.X_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_wrap_{exp}",
            )
            top = list(explain["shap"]["top10"].keys())
            explain["shap"]["top10_block_tags"] = {c: tags.get(c, "?") for c in top}
        except Exception as e:
            log(f"SHAP failed: {e}")
            explain["shap_error"] = str(e)
        try:
            explain["permutation"] = permutation_on_val(
                selected["model"],
                splits.X_val,
                splits.y_val,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_wrap_{exp}",
            )
        except Exception as e:
            log(f"perm failed: {e}")
            explain["perm_error"] = str(e)
        write_json(art / "explain.json", explain)

    elapsed = time.time() - t0
    report = [
        f"# Path A wrap — {exp} — {run_id}",
        "",
        f"**exp:** `{exp}` | multiclass",
        f"- Selected: **{selected['family']}**",
        f"- Test 4-AUC: **{tr['macro_ovr_auc']:.4f}** binary: **{tr['binary_auc']:.4f}**",
        f"- Δ vs C1 AUC: **{d_auc_c1:+.4f}** binary: **{d_bin_c1:+.4f}**",
        f"- Δ vs 1A AUC: **{d_auc_1a:+.4f}**",
        f"- Δ vs floor AUC: **{d_floor:+.4f}**",
        f"- n_features: {len(feature_cols)}",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    if retention is not None:
        report += [f"- **retention:** {retention}", ""]
    if explain.get("shap"):
        report += [f"- SHAP top10: {list(explain['shap']['top10'].keys())}", ""]
    (art / "REPORT.md").write_text("\n".join(report))
    write_json(
        art / "run_manifest.json",
        {
            "run_id": run_id,
            "exp": exp,
            "task": "multiclass",
            "elapsed_sec": elapsed,
            "git_hash": _git_hash(repo),
            "python": sys.version,
            "platform": platform.platform(),
            "delta_vs_c1_auc": d_auc_c1,
            "delta_vs_c1_binary": d_bin_c1,
            "retention": retention,
        },
    )
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done → {art}")
    return 0


def _load_sibling_metrics(repo: Path, cfg: dict[str, Any], exp_prefix: str) -> dict[str, Any] | None:
    """Find latest wrap_<exp>_* metrics_test for derived binary compare."""
    root = repo / cfg["paths"]["artifacts_root"]
    cands = sorted(
        [p for p in root.glob(f"wrap_{exp_prefix}_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        mt = p / "metrics_test.json"
        if mt.exists():
            return json.loads(mt.read_text())
    return None


def _run_binary(
    *,
    exp: str,
    cfg: dict[str, Any],
    repo: Path,
    art: Path,
    run_id: str,
    splits: Any,
    feature_cols: list[str],
    tags: dict[str, str],
    mat: dict[str, Any],
    floor: dict[str, Any],
    parent_1a: dict[str, Any],
    parent_c1: dict[str, Any],
    lgbm_device: str,
    seed: int,
    skip_shap: bool,
    log: Any,
    logs: list[str],
    t0: float,
) -> int:
    y_tr = (splits.y_train > 0).astype(int)
    y_va = (splits.y_val > 0).astype(int)
    y_te = (splits.y_test > 0).astype(int)
    base_rates = {
        "train": float(y_tr.mean()),
        "val": float(y_va.mean()),
        "test": float(y_te.mean()),
        "n_train_pos": int(y_tr.sum()),
        "n_val_pos": int(y_va.sum()),
        "n_test_pos": int(y_te.sum()),
    }
    log(f"binary base_rates={base_rates}")

    log("HPO LGBM binary")
    lgbm_pack = tune_lightgbm_binary(
        splits.X_train,
        y_tr,
        splits.X_val,
        y_va,
        cfg,
        device=lgbm_device,
    )
    log(f"LGBM val_bin_auc={lgbm_pack['val_binary_auc']:.4f}")
    write_json(art / "best_params_lgbm.json", _strip(lgbm_pack))

    log("HPO CatBoost binary")
    cat_pack = tune_catboost_binary(
        splits.X_train, y_tr, splits.X_val, y_va, cfg
    )
    log(f"Cat val_bin_auc={cat_pack['val_binary_auc']:.4f}")
    write_json(art / "best_params_catboost.json", _strip(cat_pack))

    selected = pick_family_binary(
        [lgbm_pack, cat_pack], eps=float(cfg["run"]["auc_tie_eps"])
    )
    log(
        f"VAL-SELECT {selected['family']} val_bin_auc={selected['val_binary_auc']:.4f}"
    )

    if (art / "selected_model.json").exists() or (art / "metrics_test.json").exists():
        raise RuntimeError("refuse overwrite")

    freeze = {
        "run_id": run_id,
        "phase": "A_wrap",
        "exp": exp,
        "task": "binary",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "family": selected["family"],
        "source": selected.get("source"),
        "trial_number": selected.get("trial_number"),
        "params": selected["params"],
        "best_iteration": selected.get("best_iteration"),
        "val_binary_auc": selected["val_binary_auc"],
        "val_binary_auprc": selected["val_binary_auprc"],
        "feature_cols": feature_cols,
        "feature_hash": _feature_hash(feature_cols),
        "n_trials": int(cfg["run"]["n_trials"]),
        "selected_device": selected.get("device"),
        "selected_boosting_type": selected.get("boosting_type"),
        "class_weight": "balanced",
    }
    write_json(art / "selected_model.json", freeze)
    log(f"FREEZE {art / 'selected_model.json'}")

    models_dir = art / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(selected["model"], models_dir / "selected.joblib")
    joblib.dump(lgbm_pack["model"], models_dir / "lgbm.joblib")
    joblib.dump(cat_pack["model"], models_dir / "catboost.joblib")

    score_val = predict_proba_positive(selected["model"], splits.X_val)
    score_test = predict_proba_positive(selected["model"], splits.X_test)
    tr = binary_report(y_te, score_test, tag=f"wrap_{exp}_test")
    vr = binary_report(y_va, score_val, tag=f"wrap_{exp}_val")

    # Derived parent score for comparison
    derived_ref_auc: float | None = None
    derived_label = ""
    score_parent: np.ndarray | None = None

    if exp == "bin_watch":
        derived_ref_auc = floor["test_binary_auc"]
        derived_label = "W0_multiclass_derived"
        # Prefer recompute from floor model on watch GREEN (CatBoost positional)
        try:
            from training.path_a_watch.data import feature_columns, load_merged

            wdf = load_merged(
                repo,
                cfg["paths"]["watch_green"],
                cfg["paths"]["pool_masks"],
                expected_n=int(cfg["data"]["expected_n"]),
            )
            wcols = feature_columns(wdf, feature_set="full_green")
            floor_feat = list(floor.get("feature_cols") or wcols)
            if floor.get("feature_cols") and list(floor["feature_cols"]) != list(wcols):
                # prefer freeze order; require set equality
                if set(floor_feat) != set(wcols):
                    raise AssertionError(
                        f"floor feature set drift: freeze vs loader "
                        f"only_freeze={set(floor_feat)-set(wcols)} "
                        f"only_loader={set(wcols)-set(floor_feat)}"
                    )
            Xw = wdf.set_index("person_id")[floor_feat].loc[list(splits.person_id_test)]
            Xw = Xw.reindex(columns=floor_feat)
            if list(Xw.columns) != floor_feat:
                raise AssertionError("floor feature column order mismatch after reindex")
            floor_model = joblib.load(floor["model_path"])
            proba_f = predict_proba(floor_model, Xw)
            score_parent = 1.0 - proba_f[:, 0]
            recomp = binary_report(y_te, score_parent, tag="floor_derived")
            if abs(recomp["binary_auc"] - float(derived_ref_auc)) > 1e-6:
                raise AssertionError(
                    f"floor derived recomp {recomp['binary_auc']} "
                    f"!= artifact {derived_ref_auc} (tol=1e-6)"
                )
            derived_ref_auc = recomp["binary_auc"]
        except Exception as e:
            log(f"floor recompute failed ({e}); using stored binary only")
            score_parent = None

    elif exp == "bin_c1":
        derived_ref_auc = parent_c1["test_binary_auc"]
        derived_label = "C1_multiclass_derived"
        proba_c1 = load_parent_c1_proba(repo, cfg, parent_c1, splits.person_id_test)
        score_parent = 1.0 - proba_c1[:, 0]

    elif exp == "bin_min_s":
        sibling = _load_sibling_metrics(repo, cfg, "minimal_s")
        if sibling is None:
            raise FileNotFoundError(
                "bin_min_s requires wrap_minimal_s_* metrics_test (run E1a first)"
            )
        derived_ref_auc = float(sibling["selected_raw"]["binary_auc"])
        derived_label = f"E1a_{sibling.get('exp', 'minimal_s')}_multiclass_derived"
        # recompute from sibling model for paired boot
        try:
            # find dir
            root = repo / cfg["paths"]["artifacts_root"]
            cands = sorted(
                [p for p in root.glob("wrap_minimal_s_*") if (p / "metrics_test.json").exists()],
                key=lambda p: p.name,
            )
            sdir = cands[-1]
            sfreeze = json.loads((sdir / "selected_model.json").read_text())
            smodel = joblib.load(sdir / "models" / "selected.joblib")
            sfeat = list(sfreeze["feature_cols"])
            Xs = mat["df"].set_index("person_id")[sfeat].loc[list(splits.person_id_test)]
            Xs = Xs.reindex(columns=sfeat)
            proba_s = predict_proba(smodel, Xs)
            score_parent = 1.0 - proba_s[:, 0]
        except Exception as e:
            log(f"E1a recompute for paired boot failed: {e}")

    elif exp == "bin_severity":
        sibling = _load_sibling_metrics(repo, cfg, "severity")
        if sibling is None:
            raise FileNotFoundError(
                "bin_severity requires wrap_severity_* metrics_test (run E3a first)"
            )
        derived_ref_auc = float(sibling["selected_raw"]["binary_auc"])
        derived_label = f"E3a_{sibling.get('exp', 'severity')}_multiclass_derived"
        try:
            root = repo / cfg["paths"]["artifacts_root"]
            cands = sorted(
                [p for p in root.glob("wrap_severity_*") if (p / "metrics_test.json").exists()],
                key=lambda p: p.name,
            )
            sdir = cands[-1]
            sfreeze = json.loads((sdir / "selected_model.json").read_text())
            smodel = joblib.load(sdir / "models" / "selected.joblib")
            sfeat = list(sfreeze["feature_cols"])
            Xs = mat["df"].set_index("person_id")[sfeat].loc[list(splits.person_id_test)]
            Xs = Xs.reindex(columns=sfeat)
            proba_s = predict_proba(smodel, Xs)
            score_parent = 1.0 - proba_s[:, 0]
        except Exception as e:
            log(f"E3a recompute for paired boot failed: {e}")

    d_vs_derived = (
        tr["binary_auc"] - derived_ref_auc if derived_ref_auc is not None else float("nan")
    )
    prefer_binary_primary = (
        derived_ref_auc is not None and (tr["binary_auc"] - derived_ref_auc) >= 0.01
    )

    boot = bootstrap_ci_binary(
        y_te,
        score_test,
        n_boot=int(cfg["run"]["bootstrap_n"]),
        alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
        seed=seed,
    )
    boot_d = None
    if score_parent is not None:
        boot_d = paired_delta_bootstrap_binary(
            y_te,
            score_test,
            score_parent,
            n_boot=int(cfg["run"]["bootstrap_n"]),
            alpha=1.0 - float(cfg["run"]["bootstrap_ci"]),
            seed=seed + 17,
        )

    metrics_test: dict[str, Any] = {
        "phase": "A_wrap",
        "exp": exp,
        "task": "binary",
        "selected_family": selected["family"],
        "selected_raw": tr,
        "base_rates": base_rates,
        "vs_multiclass_derived": {
            "derived_label": derived_label,
            "derived_binary_auc": derived_ref_auc,
            "delta_binary_auc": d_vs_derived,
            "prefer_binary_primary": prefer_binary_primary,
            "gate": "prefer binary-primary if ΔAUC >= +0.01",
            "bootstrap_paired_delta": boot_d,
        },
        "bootstrap_test": boot,
    }
    write_json(art / "metrics_val.json", {"selected_raw": vr, "base_rates": base_rates})
    write_json(art / "metrics_test.json", metrics_test)
    write_json(
        art / "selected_model_post.json",
        {"test_scored": True, "test_scored_at": datetime.now(timezone.utc).isoformat()},
    )
    log(
        f"TEST bin_auc={tr['binary_auc']:.4f} auprc={tr['binary_auprc']:.4f} "
        f"vs_derived={d_vs_derived:+.4f} prefer_bin_primary={prefer_binary_primary}"
    )

    explain: dict[str, Any] = {}
    if not skip_shap:
        try:
            explain["permutation"] = permutation_on_val_binary(
                selected["model"],
                splits.X_val,
                y_va,
                art / "shap",
                seed=seed,
                prefix=f"{selected['family']}_wrap_{exp}",
            )
        except Exception as e:
            log(f"binary perm failed: {e}")
            explain["perm_error"] = str(e)
        write_json(art / "explain.json", explain)

    elapsed = time.time() - t0
    report = [
        f"# Path A wrap — {exp} — {run_id}",
        "",
        f"**exp:** `{exp}` | binary y=(label>0)",
        f"- Selected: **{selected['family']}**",
        f"- Test binary AUC: **{tr['binary_auc']:.4f}** AUPRC: **{tr['binary_auprc']:.4f}**",
        f"- Brier: {tr['binary_brier']:.4f}",
        f"- Base rates: {base_rates}",
        f"- vs derived ({derived_label}): **{d_vs_derived:+.4f}** "
        f"prefer_binary_primary={prefer_binary_primary}",
        f"- n_features: {len(feature_cols)}",
        f"- Elapsed: {elapsed/60:.1f} min",
        "",
    ]
    (art / "REPORT.md").write_text("\n".join(report))
    write_json(
        art / "run_manifest.json",
        {
            "run_id": run_id,
            "exp": exp,
            "task": "binary",
            "elapsed_sec": elapsed,
            "git_hash": _git_hash(repo),
            "python": sys.version,
            "platform": platform.platform(),
            "test_binary_auc": tr["binary_auc"],
            "delta_vs_derived": d_vs_derived,
            "prefer_binary_primary": prefer_binary_primary,
        },
    )
    (art / "run.log").write_text("\n".join(logs) + "\n")
    log(f"done → {art}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Path A wrap experiments")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--exp", type=str, choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--all", action="store_true", help="Run all experiments in plan order")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if not args.all and not args.exp:
        ap.error("need --exp or --all")

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()

    if args.all:
        exps = list(ALL_ORDER)
    else:
        exps = [args.exp]

    rc = 0
    for exp in exps:
        run_id = args.run_id
        if run_id is None or (args.all and len(exps) > 1):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"wrap_{exp}_{ts}"
        try:
            r = run_one(
                exp,
                cfg=cfg,
                repo=repo,
                run_id=run_id,
                quick=args.quick,
                n_trials=args.n_trials,
                skip_shap=args.skip_shap,
            )
            if r != 0:
                rc = r
                if args.all:
                    print(f"STOP: {exp} failed rc={r}", flush=True)
                    break
        except Exception as e:
            print(f"FAIL exp={exp}: {e}", flush=True)
            if args.all:
                raise
            raise
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
