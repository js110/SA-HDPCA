#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml


def load_collapse_thresholds(config: Dict) -> Tuple[float, Dict[str, float]]:
    budget_cfg = config.get("budget", {})
    default_thr = float(budget_cfg.get("collapse_threshold", 0.55))
    per_ds_cfg = budget_cfg.get("datasets", {})
    thresholds = {}
    for ds in config.get("datasets", {}):
        ds_cfg = per_ds_cfg.get(ds, {})
        thresholds[ds] = float(ds_cfg.get("collapse_threshold", default_thr))
    return default_thr, thresholds


def compute_collapse(df: pd.DataFrame, config: Dict) -> pd.Series:
    if "collapse_final" in df.columns:
        return df["collapse_final"].astype(float)
    default_thr, thresholds = load_collapse_thresholds(config)
    collapse = []
    for _, row in df.iterrows():
        ds = str(row.get("dataset", "")).lower()
        thr = thresholds.get(ds, default_thr)
        non_empty = float(row.get("non_empty_k_final", 0))
        k_val = float(row.get("k", 0))
        max_ratio = float(row.get("max_cluster_ratio_final", 0.0))
        collapse.append(1.0 if (non_empty < k_val or max_ratio > thr) else 0.0)
    return pd.Series(collapse, index=df.index, dtype=float)


def summarize(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    df["collapse_rate"] = compute_collapse(df, config)
    group_cols = ["dataset", "method", "eps_tot", "eps_proxy_ratio"]
    agg = df.groupby(group_cols).agg(
        n=("seed", "count"),
        eps_proxy_mean=("eps_proxy", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        sse_x_mean=("sse_x", "mean"),
        sse_x_std=("sse_x", "std"),
        collapse_rate=("collapse_rate", "mean"),
        runtime_ms_mean=("runtime_ms_total", "mean"),
        runtime_ms_std=("runtime_ms_total", "std"),
    )
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize budget-split results by proxy ratio.")
    parser.add_argument("--results", type=str, default="outputs/budget_split/results.csv", help="Path to results.csv")
    parser.add_argument("--config", type=str, default="configs/budget_split.yaml", help="Config for thresholds")
    parser.add_argument("--out", type=str, default="outputs/budget_split/summary_budget_split.csv", help="Output CSV")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(results_path)
    summary = summarize(df, config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
