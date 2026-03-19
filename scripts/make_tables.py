#!/usr/bin/env python3
"""
从 results_summary.csv 生成 LaTeX 表：
  - tables/table_main.tex
  - tables/table_sig.tex
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


METHOD_ORDER = ["sahdpca_fb", "sahdpca_no_fb", "sahdpca_strong", "dp_kmeans", "kmeanspp_dp", "pca_dp"]
METHOD_LABEL = {
    "sahdpca_fb": "SA-HDPCA (fb)",
    "sahdpca_no_fb": "SA-HDPCA (no-fb)",
    "sahdpca_strong": "SA-HDPCA (strong)",
    "dp_kmeans": "DP-k-means",
    "kmeanspp_dp": "kmeans++-DP",
    "pca_dp": "PCA-DP",
}


def _select_eps(df: pd.DataFrame, dataset: str, desired: List[float]) -> List[float]:
    avail = sorted(df[df["dataset"] == dataset]["eps_tot"].unique())
    if not avail:
        return []
    chosen = []
    for e in desired:
        nearest = min(avail, key=lambda x: abs(x - e))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def _fmt(mean: float, std: float, kind: str) -> str:
    if pd.isna(mean):
        return "--"
    if kind == "f1":
        f = lambda v: f"{v:.3f}"
    elif kind == "sse":
        f = lambda v: f"{v:.3g}"
    else:
        f = lambda v: f"{v:.3f}"
    return f"{f(mean)} ± {f(std)}"


def bold_best(values: List[str], best_idx: int) -> List[str]:
    out = []
    for i, v in enumerate(values):
        if i == best_idx and v != "--":
            out.append(f"\\textbf{{{v}}}")
        else:
            out.append(v)
    return out


def build_table_main(df: pd.DataFrame, out_path: Path) -> None:
    datasets = ["HAR", "GAS"]
    eps_list = [0.5, 1.0]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & $\\epsilon$ & Method & Macro-F1 & SSE & Non-empty K & Max ratio \\\\",
        "\\midrule",
    ]
    for ds in datasets:
        eps_sel = _select_eps(df, ds, eps_list)
        if not eps_sel:
            continue
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{ds}}}}} \\\\")
        for eps in eps_sel:
            block = []
            for m in METHOD_ORDER:
                sub = df[(df["dataset"] == ds) & (df["method"] == m) & (np.isclose(df["eps_tot"], eps))]
                if sub.empty:
                    block.append(("--", "--", "--", "--"))
                else:
                    macro = _fmt(sub["macro_f1"].mean(), sub["macro_f1"].std(ddof=1), "f1")
                    sse = _fmt(sub["sse"].mean(), sub["sse"].std(ddof=1), "sse")
                    nek = _fmt(sub["non_empty_k"].mean(), sub["non_empty_k"].std(ddof=1), "f1")
                    mcr = _fmt(sub["max_cluster_ratio"].mean(), sub["max_cluster_ratio"].std(ddof=1), "f1")
                    block.append((macro, sse, nek, mcr))
            # 找最佳 macro_f1
            macro_vals = [b[0] for b in block]
            macro_means = []
            for s in macro_vals:
                try:
                    macro_means.append(float(s.split("±")[0].split("±")[0].split(" ")[0]))
                except Exception:
                    macro_means.append(-math.inf)
            best_idx = int(np.nanargmax(macro_means)) if macro_means else -1
            macro_vals = bold_best(macro_vals, best_idx)
            for m, (macro, sse, nek, mcr) in zip(METHOD_ORDER, block):
                macro_disp = macro_vals[METHOD_ORDER.index(m)]
                lines.append(f"{ds} & {eps:.2f} & {METHOD_LABEL[m]} & {macro_disp} & {sse} & {nek} & {mcr} \\\\")
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\caption{主结果表，均值±标准差；最佳 Macro-F1 加粗（20 seeds）。}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def build_table_sig(df: pd.DataFrame, out_path: Path) -> None:
    datasets = ["HAR", "GAS"]
    eps_list = [0.5, 0.8, 1.0]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & $\\epsilon$ & mean diff & $p_{ttest}$ & $p_{MWU}$ \\\\",
        "\\midrule",
    ]
    for ds in datasets:
        for eps in eps_list:
            fb = df[(df["dataset"] == ds) & (df["method"] == "sahdpca_fb") & (np.isclose(df["eps_tot"], eps))]
            nfb = df[(df["dataset"] == ds) & (df["method"] == "sahdpca_no_fb") & (np.isclose(df["eps_tot"], eps))]
            if fb.empty or nfb.empty:
                continue
            # 对齐 seeds
            merged = fb.merge(nfb, on="seed", suffixes=("_fb", "_nfb"))
            if merged.empty:
                continue
            vals_fb = merged["macro_f1_fb"].to_numpy()
            vals_nfb = merged["macro_f1_nfb"].to_numpy()
            mean_diff = float(np.nanmean(vals_fb - vals_nfb))
            if len(vals_fb) >= 2:
                _, p_t = stats.ttest_rel(vals_fb, vals_nfb, nan_policy="omit")
            else:
                _, p_t = stats.ttest_ind(vals_fb, vals_nfb, equal_var=False, nan_policy="omit")
            try:
                _, p_mwu = stats.mannwhitneyu(vals_fb, vals_nfb, alternative="two-sided", method="auto")
            except ValueError:
                p_mwu = np.nan
            lines.append(f"{ds} & {eps:.2f} & {mean_diff:.3f} & {p_t:.2e} & {(p_mwu if not math.isnan(p_mwu) else np.nan):.2e} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\caption{fb vs no-fb 显著性；双侧 t-test 与 Mann--Whitney U，显著性阈值 0.05。}", "\\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results_summary.csv.")
    parser.add_argument("--results", type=str, default="results_summary.csv")
    parser.add_argument("--out_dir", type=str, default="tables")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    out_dir = Path(args.out_dir)
    build_table_main(df, out_dir / "table_main.tex")
    build_table_sig(df, out_dir / "table_sig.tex")
    print(f"[OK] 表格已生成到 {out_dir}")


if __name__ == "__main__":
    main()
