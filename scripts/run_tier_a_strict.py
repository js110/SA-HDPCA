#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import methods
from src.utils import ensure_dir


METHOD_CFG_KEYS = {
    "feature_mode",
    "use_pca",
    "init_mode",
    "budget_mode",
    "schedule_kind",
    "feedback_overrides",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair Tier A strict-DP tuning and frozen evaluation.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="outputs/tier_a_strict")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--best-configs", type=str, default=None, help="Optional path to a saved best_configs.json.")
    return parser.parse_args()


def _cartesian(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[k]) for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def _tier_a_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    tier_a = dict(config.get("tier_a", {}))
    tier_a.setdefault("datasets", ["har", "gas"])
    tier_a.setdefault(
        "methods",
        [
            "dp_kmeans",
            "kmeanspp_dp",
            "pca_dp",
            "sahdpca_wo_feedback",
            "sahdpca",
            "sahdpca_proxy_kpp",
            "sahdpca_proxy_rr",
        ],
    )
    tier_a.setdefault("tune_eps", [0.8, 1.0])
    tier_a.setdefault("eval_eps", [0.5, 0.8, 1.0, 1.5])
    tier_a.setdefault("tune_seeds", [0, 1])
    tier_a.setdefault("eval_seeds", config.get("seeds", [0, 1, 2, 3, 4]))
    tier_a.setdefault("search_space", {})
    search = tier_a["search_space"]
    search.setdefault("default", {})
    search.setdefault("datasets", {})
    return tier_a


def _candidate_grid(tier_a: Dict[str, Any], dataset: str, method: str) -> List[Dict[str, Any]]:
    search = tier_a["search_space"]
    base = dict(search.get("default", {}).get(method, {}))
    ds = dict(search.get("datasets", {}).get(dataset, {}).get(method, {}))
    base.update(ds)
    combos = _cartesian(base)
    return combos or [{}]


def _split_overrides(method: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    dataset_overrides: Dict[str, Any] = {}
    method_overrides: Dict[str, Any] = {}
    for key, value in candidate.items():
        if key in METHOD_CFG_KEYS:
            method_overrides[key] = value
        else:
            dataset_overrides[key] = value
    if method_overrides:
        dataset_overrides["method_overrides"] = {method: method_overrides}
    return dataset_overrides


def _summarize_candidate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    return {
        "f1_mean": float(df["f1"].mean()),
        "f1_std": float(df["f1"].std(ddof=0)),
        "sse_mean": float(df["sse_x"].mean()),
        "runtime_mean_ms": float(df["runtime_ms_total"].mean()),
    }


def _candidate_sort_key(summary: Dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["f1_mean"]),
        -float(summary["f1_std"]),
        -float(summary["sse_mean"]),
    )


def _run_grid(
    config: Dict[str, Any],
    out_dir: str,
    dataset: str,
    method: str,
    eps_list: List[float],
    seeds: List[int],
    candidate: Dict[str, Any],
    tag: str,
) -> List[Dict[str, Any]]:
    dataset_overrides = _split_overrides(method, candidate)
    rows: List[Dict[str, Any]] = []
    for eps in eps_list:
        for seed in seeds:
            res = methods.run_method(
                dataset=dataset,
                method=method,
                eps_tot=float(eps),
                seed=int(seed),
                config=config,
                out_dir=out_dir,
                dataset_overrides=dataset_overrides if dataset_overrides else None,
            )
            row = {
                "stage": tag,
                "dataset": dataset,
                "method": method,
                "eps_tot": float(eps),
                "seed": int(seed),
                "candidate_json": json.dumps(candidate, sort_keys=True),
            }
            row.update(res)
            rows.append(row)
    return rows


def tune_tier_a(config: Dict[str, Any], out_dir: str) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    tier_a = _tier_a_cfg(config)
    tuning_rows: List[Dict[str, Any]] = []
    best_configs: Dict[str, Dict[str, Any]] = {}
    best_rows: List[Dict[str, Any]] = []

    for dataset in tier_a["datasets"]:
        best_configs[dataset] = {}
        for method in tier_a["methods"]:
            candidates = _candidate_grid(tier_a, dataset, method)
            scored: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            for candidate in candidates:
                rows = _run_grid(
                    config=config,
                    out_dir=out_dir,
                    dataset=dataset,
                    method=method,
                    eps_list=[float(e) for e in tier_a["tune_eps"]],
                    seeds=[int(s) for s in tier_a["tune_seeds"]],
                    candidate=candidate,
                    tag="tune",
                )
                tuning_rows.extend(rows)
                summary = _summarize_candidate_rows(rows)
                summary["candidate_json"] = json.dumps(candidate, sort_keys=True)
                scored.append((candidate, summary))
            best_candidate, best_summary = max(scored, key=lambda item: _candidate_sort_key(item[1]))
            best_configs[dataset][method] = {
                "candidate": best_candidate,
                "summary": best_summary,
            }
            best_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "candidate_json": json.dumps(best_candidate, sort_keys=True),
                    **best_summary,
                }
            )

    return pd.DataFrame(tuning_rows), best_configs, pd.DataFrame(best_rows)


def evaluate_tier_a(config: Dict[str, Any], out_dir: str, best_configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    tier_a = _tier_a_cfg(config)
    rows: List[Dict[str, Any]] = []
    for dataset in tier_a["datasets"]:
        for method in tier_a["methods"]:
            candidate = best_configs.get(dataset, {}).get(method, {}).get("candidate", {})
            rows.extend(
                _run_grid(
                    config=config,
                    out_dir=out_dir,
                    dataset=dataset,
                    method=method,
                    eps_list=[float(e) for e in tier_a["eval_eps"]],
                    seeds=[int(s) for s in tier_a["eval_seeds"]],
                    candidate=candidate,
                    tag="eval",
                )
            )
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby(["stage", "dataset", "method", "eps_tot"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            sse_mean=("sse_x", "mean"),
            sse_std=("sse_x", "std"),
            runtime_mean_ms=("runtime_ms_total", "mean"),
        )
        .sort_values(["stage", "dataset", "eps_tot", "f1_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    return agg


def lead_check(df_eval: pd.DataFrame) -> pd.DataFrame:
    if df_eval.empty:
        return df_eval
    core = df_eval[df_eval["eps_tot"].isin([0.8, 1.0])]
    agg = core.groupby(["dataset", "method", "eps_tot"], as_index=False)["f1"].mean()
    leaders = agg.groupby(["dataset", "eps_tot"])["f1"].transform("max")
    agg["is_best_f1"] = agg["f1"] >= leaders - 1e-12
    return agg.sort_values(["dataset", "eps_tot", "f1"], ascending=[True, True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out)
    ensure_dir(str(out_dir))

    best_configs_path = Path(args.best_configs) if args.best_configs else out_dir / "best_configs.json"
    tuning_df = pd.DataFrame()

    if args.skip_tune:
        if not best_configs_path.exists():
            raise FileNotFoundError(f"Missing best config file: {best_configs_path}")
        best_configs = json.loads(best_configs_path.read_text())
    else:
        tuning_df, best_configs, best_df = tune_tier_a(config, str(out_dir))
        tuning_df.to_csv(out_dir / "tuning_results.csv", index=False)
        best_df.to_csv(out_dir / "best_configs_summary.csv", index=False)
        best_configs_path.write_text(json.dumps(best_configs, indent=2, sort_keys=True))

    if args.tune_only:
        aggregate_results(tuning_df).to_csv(out_dir / "tuning_summary.csv", index=False)
        if not args.skip_tune:
            print(f"[OK] wrote {out_dir / 'best_configs_summary.csv'}")
        print(f"[OK] wrote {out_dir / 'tuning_results.csv'}")
        print(f"[OK] wrote {best_configs_path}")
        return

    eval_df = evaluate_tier_a(config, str(out_dir), best_configs)
    eval_df.to_csv(out_dir / "results.csv", index=False)
    aggregate_results(eval_df).to_csv(out_dir / "summary.csv", index=False)
    lead_check(eval_df).to_csv(out_dir / "lead_check.csv", index=False)

    if not tuning_df.empty:
        aggregate_results(tuning_df).to_csv(out_dir / "tuning_summary.csv", index=False)

    print(f"[OK] wrote {best_configs_path}")
    print(f"[OK] wrote {out_dir / 'results.csv'}")
    print(f"[OK] wrote {out_dir / 'summary.csv'}")
    print(f"[OK] wrote {out_dir / 'lead_check.csv'}")


if __name__ == "__main__":
    main()
