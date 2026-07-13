"""Build dual-rank minimal feature lists from frozen C1 SHAP + perm CSVs.

Writes training/path_a_blocks/artifacts/wrap_feature_ranks.json before any wrap HPO.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def dual_rank_minimal(
    shap_csv: Path,
    perm_csv: Path,
    *,
    exclude: list[str] | None = None,
    n_s: int = 12,
    n_m: int = 18,
) -> dict[str, Any]:
    exclude = list(exclude or ["cestl"])
    shap = pd.read_csv(shap_csv, index_col=0).iloc[:, 0].astype(float)
    perm = pd.read_csv(perm_csv, index_col=0).iloc[:, 0].astype(float)
    shap.name = "shap"
    perm.name = "perm"
    if set(shap.index) != set(perm.index):
        only_s = set(shap.index) - set(perm.index)
        only_p = set(perm.index) - set(shap.index)
        raise AssertionError(f"shap/perm feature mismatch only_shap={only_s} only_perm={only_p}")

    df = pd.DataFrame({"shap": shap, "perm": perm})
    dropped = [c for c in exclude if c in df.index]
    df = df.drop(index=dropped, errors="ignore")
    if len(df) < n_m:
        raise AssertionError(f"only {len(df)} candidates after exclude; need >= {n_m}")

    df["rank_shap"] = df["shap"].rank(ascending=False, method="average")
    df["rank_perm"] = df["perm"].rank(ascending=False, method="average")
    df["combined"] = df["rank_shap"] + df["rank_perm"]
    df = df.sort_values(
        ["combined", "shap", "perm"],
        ascending=[True, False, False],
    )

    ranks = []
    for feat, row in df.iterrows():
        ranks.append(
            {
                "feature": str(feat),
                "shap": float(row["shap"]),
                "perm": float(row["perm"]),
                "rank_shap": float(row["rank_shap"]),
                "rank_perm": float(row["rank_perm"]),
                "combined": float(row["combined"]),
            }
        )
    minimal_s = [r["feature"] for r in ranks[:n_s]]
    minimal_m = [r["feature"] for r in ranks[:n_m]]
    return {
        "exclude": exclude,
        "dropped": dropped,
        "n_candidates": len(ranks),
        "n_s": n_s,
        "n_m": n_m,
        "ranks": ranks,
        "minimal_S": minimal_s,
        "minimal_M": minimal_m,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build wrap dual-rank minimal feature lists")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-s", type=int, default=12)
    ap.add_argument("--n-m", type=int, default=18)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = (args.repo_root or Path.cwd()).resolve()
    c1_id = cfg["paths"]["parent_c1_run_id"]
    art_root = repo / cfg["paths"]["artifacts_root"]
    shap_csv = (
        art_root
        / c1_id
        / "shap"
        / "catboost_1c_scores_shap_importance.csv"
    )
    perm_csv = (
        art_root
        / c1_id
        / "shap"
        / "catboost_1c_scores_perm_importance.csv"
    )
    out_path = repo / cfg["paths"]["wrap_ranks"]

    if not shap_csv.exists():
        raise FileNotFoundError(shap_csv)
    if not perm_csv.exists():
        raise FileNotFoundError(perm_csv)
    if out_path.exists() and not args.force:
        raise RuntimeError(f"refuse overwrite {out_path} (pass --force)")

    pack = dual_rank_minimal(
        shap_csv,
        perm_csv,
        exclude=["cestl"],
        n_s=int(args.n_s),
        n_m=int(args.n_m),
    )
    payload = {
        "source_run_id": c1_id,
        "shap_csv": str(shap_csv.relative_to(repo)),
        "perm_csv": str(perm_csv.relative_to(repo)),
        "built_at": datetime.now(timezone.utc).isoformat(),
        **pack,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {out_path}")
    print(f"n_candidates={pack['n_candidates']} dropped={pack['dropped']}")
    print(f"minimal_S ({len(pack['minimal_S'])}): {pack['minimal_S']}")
    print(f"minimal_M ({len(pack['minimal_M'])}): {pack['minimal_M']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
