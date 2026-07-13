"""Load and validate pipeline/config.yaml."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

SERIES_MODALITIES = (
    "heart_rate",
    "stress",
    "respiratory_rate",
    "oxygen_saturation",
    "sleep",
    "physical_activity",
    "physical_activity_calorie",
    "cgm",
)

# raw relative paths under raw_root
MODALITY_RELPATH = {
    "heart_rate": "garmin/heart_rate.parquet",
    "stress": "garmin/stress.parquet",
    "respiratory_rate": "garmin/respiratory_rate.parquet",
    "oxygen_saturation": "garmin/oxygen_saturation.parquet",
    "sleep": "garmin/sleep.parquet",
    "physical_activity": "garmin/physical_activity.parquet",
    "physical_activity_calorie": "garmin/physical_activity_calorie.parquet",
    "cgm": "dexcom/cgm.parquet",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return resolve_paths(cfg)


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Attach absolute Path objects under cfg['_paths']."""
    p = cfg["paths"]
    raw = Path(p["raw_root"])
    out = Path(p["out_root"])
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    if not out.is_absolute():
        out = REPO_ROOT / out

    def sub(key: str, default_name: str) -> Path:
        v = p.get(key)
        if v:
            path = Path(v)
            return path if path.is_absolute() else REPO_ROOT / path
        return out / default_name

    cfg["_paths"] = {
        "repo_root": REPO_ROOT,
        "raw_root": raw.resolve(),
        "out_root": out.resolve(),
        "clean_dir": sub("clean_dir", "clean").resolve(),
        "meta_dir": sub("meta_dir", "meta").resolve(),
        "features_dir": sub("features_dir", "features").resolve(),
        "reports_dir": sub("reports_dir", "reports").resolve(),
        "config_path": DEFAULT_CONFIG_PATH,
    }
    return cfg


def ensure_out_dirs(cfg: dict[str, Any]) -> None:
    for key in ("out_root", "clean_dir", "meta_dir", "features_dir", "reports_dir"):
        cfg["_paths"][key].mkdir(parents=True, exist_ok=True)


def modality_list(cfg: dict[str, Any]) -> list[str]:
    only = cfg.get("runtime", {}).get("only_modalities")
    if only:
        return list(only)
    return list(SERIES_MODALITIES)
