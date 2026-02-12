                      

"""
遍历 runs/ 目录收集实验结果，输出 tidy 汇总 CSV。

预期目录结构：
  runs/{dataset}/{method}/eps{eps_tot}_seed{seed}.csv

CSV 至少包含：
  - 迭代行：t 或 iter，eps_t，noise_scale，drift
  - 汇总行：macro_f1, sse, non_empty_k, max_cluster_ratio, runtime_sec
    可以通过 t=-1 标记；若无标记则取最后一行含上述指标。
兼容旧字段：runtime_ms_total / runtime_ms（自动转秒）、non_empty_k_final、max_cluster_ratio_final。
"""

from __future__ import annotations



import argparse

import re

from pathlib import Path

from typing import Dict, List, Optional



import numpy as np

import pandas as pd





CANON_METHOD = {

    "sahdpca": "sahdpca_fb",

    "sahdpca_fb": "sahdpca_fb",

    "sahdpca_wo_feedback": "sahdpca_no_fb",

    "sahdpca_no_fb": "sahdpca_no_fb",

    "sahdpca_strong": "sahdpca_strong",

    "dp_kmeans": "dp_kmeans",

    "kmeanspp_dp": "kmeanspp_dp",

    "pca_dp": "pca_dp",

}





def _infer_meta(path: Path) -> Optional[Dict[str, str]]:

    m = re.search(r"eps([0-9.]+)_seed([0-9]+)\.csv$", path.name)

    if not m:

        return None

    eps_tot = float(m.group(1))

    seed = int(m.group(2))

    try:

        method = path.parent.name

        dataset = path.parent.parent.name

    except Exception:

        return None

    method_std = CANON_METHOD.get(method)

    if method_std is None:

        return None

    return {"dataset": dataset.upper(), "method": method_std, "eps_tot": eps_tot, "seed": seed}





def _runtime_seconds(row: pd.Series) -> float | None:

    if "runtime_sec" in row:

        return row["runtime_sec"]

    if "runtime_ms_total" in row:

        return row["runtime_ms_total"] / 1000.0

    if "runtime_ms" in row:

        return row["runtime_ms"] / 1000.0

    if "runtime" in row:

        return row["runtime"]

    return None





def _extract_final(df: pd.DataFrame) -> pd.Series:

                      

    if "t" in df.columns:

        candidates = df[df["t"] < 0]

        if not candidates.empty:

            return candidates.iloc[-1]

    if "iter" in df.columns:

        candidates = df[df["iter"] < 0]

        if not candidates.empty:

            return candidates.iloc[-1]

                              

    metric_cols = [c for c in ["macro_f1", "sse", "non_empty_k", "max_cluster_ratio"] if c in df.columns]

    if metric_cols:

        non_null = df.dropna(subset=metric_cols, how="any")

        if not non_null.empty:

            return non_null.iloc[-1]

             

    return df.iloc[-1]





def collect_runs(runs_dir: Path) -> pd.DataFrame:

    records: List[Dict[str, float]] = []

    paths = sorted(runs_dir.glob("*/*/eps*_seed*.csv"))

    for path in paths:

        meta = _infer_meta(path)

        if meta is None:

            continue

        df = pd.read_csv(path)

        final_row = _extract_final(df)

        rec = {

            "dataset": meta["dataset"],

            "method": meta["method"],

            "eps_tot": meta["eps_tot"],

            "seed": meta["seed"],

            "macro_f1": final_row.get("macro_f1"),

            "sse": final_row.get("sse") if "sse" in final_row else final_row.get("sse_x"),

            "non_empty_k": final_row.get("non_empty_k") if "non_empty_k" in final_row else final_row.get("non_empty_k_final"),

            "max_cluster_ratio": final_row.get("max_cluster_ratio") if "max_cluster_ratio" in final_row else final_row.get("max_cluster_ratio_final"),

            "runtime_sec": _runtime_seconds(final_row),

        }

        records.append(rec)

    return pd.DataFrame(records)





def main() -> None:

    parser = argparse.ArgumentParser(description="Collect run-level results into a tidy CSV.")

    parser.add_argument("--runs_dir", type=str, default="runs", help="根目录：runs/{dataset}/{method}/epsX_seedY.csv")

    parser.add_argument("--out", type=str, default="results_summary.csv", help="输出 CSV 路径")

    args = parser.parse_args()



    runs_dir = Path(args.runs_dir)

    if not runs_dir.exists():

        raise FileNotFoundError(f"runs 目录不存在: {runs_dir}")



    df = collect_runs(runs_dir)

    df.to_csv(args.out, index=False)

    print(f"[OK] 写入 {args.out}，共 {len(df)} 行。")





if __name__ == "__main__":

    main()

