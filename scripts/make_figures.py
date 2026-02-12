                      

"""
使用 results_summary.csv + per-iter 日志生成 Fig1-Fig9（PDF+PNG，600dpi）。
风格：Times/serif，线宽1.8，marker 6，capsize 3，浅灰虚线网格。
"""

from __future__ import annotations



import argparse

from pathlib import Path

from typing import Dict, Iterable, List, Tuple



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





METHOD_DISPLAY = {

    "sahdpca_fb": "SA-HDPCA (fb)",

    "sahdpca_no_fb": "SA-HDPCA (no-fb)",

    "sahdpca_strong": "SA-HDPCA (strong)",

    "dp_kmeans": "DP-k-means",

    "kmeanspp_dp": "kmeans++-DP",

    "pca_dp": "PCA-DP",

}



COLORS = {

    "sahdpca_fb": "#1f77b4",

    "sahdpca_no_fb": "#d62728",

    "sahdpca_strong": "#2ca02c",

    "dp_kmeans": "#9467bd",

    "kmeanspp_dp": "#8c564b",

    "pca_dp": "#e377c2",

}



MARKERS = {

    "sahdpca_fb": "o",

    "sahdpca_no_fb": "s",

    "sahdpca_strong": "^",

    "dp_kmeans": "D",

    "kmeanspp_dp": "v",

    "pca_dp": "P",

}





def set_style() -> None:

    plt.rcParams.update(

        {

            "font.family": "serif",

            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],

            "mathtext.fontset": "stix",

            "font.size": 9,

            "axes.labelsize": 9,

            "legend.fontsize": 9,

            "lines.linewidth": 1.8,

            "lines.markersize": 6,

            "errorbar.capsize": 3,

            "figure.dpi": 150,

        }

    )





def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")

    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=600)

    plt.close(fig)





def add_label(ax: plt.Axes, text: str) -> None:

    ax.text(-0.12, 1.05, text, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="top")





def load_schedule(runs_dir: Path, dataset: str, method: str, eps: float = 1.0, seed: int = 0) -> pd.DataFrame:

    path = runs_dir / dataset.lower() / method / f"eps{eps}_seed{seed}.csv"

    if not path.exists():

        raise FileNotFoundError(f"缺少调度日志: {path}")

    df = pd.read_csv(path)

           

    if "t" in df.columns:

        df = df[df["t"] >= 0]

        df = df.rename(columns={"t": "iter"})

    elif "iter" not in df.columns:

        df = df.reset_index().rename(columns={"index": "iter"})

    return df.sort_values("iter")





def agg_mean_std(df: pd.DataFrame, metric: str, methods: Iterable[str], datasets: Iterable[str]) -> pd.DataFrame:

    sub = df[df["method"].isin(methods) & df["dataset"].isin(datasets)].copy()

    return (

        sub.groupby(["dataset", "method", "eps_tot"], as_index=False)[metric]

        .agg(["mean", "std"])

        .reset_index()

        .rename(columns={"mean": "metric_mean", "std": "metric_std"})

    )





def plot_f1_sse(df: pd.DataFrame, metric: str, methods: List[str], out_dir: Path, fname: str, ylabel: str, ylim=None, use_sci=False):

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)

    for ax, dataset, label in zip(axes, ["HAR", "GAS"], ["(a) HAR", "(b) GAS"]):

        for m in methods:

            sub = df[(df["dataset"] == dataset) & (df["method"] == m)].sort_values("eps_tot")

            if sub.empty:

                continue

            ax.errorbar(

                sub["eps_tot"],

                sub["metric_mean"],

                yerr=sub["metric_std"],

                label=METHOD_DISPLAY.get(m, m),

                color=COLORS.get(m, "#333"),

                marker=MARKERS.get(m, "o"),

            )

        ax.set_xlabel(r"$\epsilon$")

        ax.set_ylabel(ylabel)

        if ylim:

            ax.set_ylim(*ylim)

        if use_sci:

            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        add_label(ax, label)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig.tight_layout()

    save_fig(fig, out_dir, fname)





def plot_collapse(df: pd.DataFrame, out_dir: Path):

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    panels = [

        ("HAR", "non_empty_k", "(a) HAR non_empty_k"),

        ("HAR", "max_cluster_ratio", "(b) HAR max_cluster_ratio"),

        ("GAS", "non_empty_k", "(c) GAS non_empty_k"),

        ("GAS", "max_cluster_ratio", "(d) GAS max_cluster_ratio"),

    ]

    methods = ["sahdpca_fb", "sahdpca_no_fb", "sahdpca_strong"]

    for ax, (dataset, metric, label) in zip(axes.flat, panels):

        agg = agg_mean_std(df, metric, methods, [dataset])

        for m in methods:

            sub = agg[(agg["dataset"] == dataset) & (agg["method"] == m)].sort_values("eps_tot")

            if sub.empty:

                continue

            ax.errorbar(

                sub["eps_tot"],

                sub["metric_mean"],

                yerr=sub["metric_std"],

                label=METHOD_DISPLAY[m],

                color=COLORS[m],

                marker=MARKERS[m],

            )

        ax.set_xlabel(r"$\epsilon$")

        ax.set_ylabel(metric.replace("_", " "))

        add_label(ax, label)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig.tight_layout()

    save_fig(fig, out_dir, "fig6_collapse_vs_eps")





def plot_schedule_figs(runs_dir: Path, out_dir: Path):

    datasets = ["HAR", "GAS"]

                     

    fig1, ax1 = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, ds, lab in zip(ax1, datasets, ["(a) HAR", "(b) GAS"]):

        for m in ["sahdpca_fb", "sahdpca_no_fb"]:

            try:

                df = load_schedule(runs_dir, ds, m, eps=1.0, seed=0)

            except FileNotFoundError as e:

                print(f"[WARN] {e}")

                continue

            ax.plot(df["iter"], df["eps_t"].cumsum(), label=METHOD_DISPLAY[m], color=COLORS[m], marker=MARKERS[m])

        ax.set_xlabel("Iteration")

        if ax is ax1[0]:

            ax.set_ylabel("Cumulative epsilon")

        add_label(ax, lab)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig1.tight_layout()

    save_fig(fig1, out_dir, "fig1_cumulative_budget")



                       

    fig2, ax2 = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, ds, lab in zip(ax2, datasets, ["(a) HAR", "(b) GAS"]):

        for m in ["sahdpca_fb", "sahdpca_no_fb", "sahdpca_strong"]:

            try:

                df = load_schedule(runs_dir, ds, m, eps=1.0, seed=0)

            except FileNotFoundError as e:

                print(f"[WARN] {e}")

                continue

            ax.plot(df["iter"], df["eps_t"], label=METHOD_DISPLAY[m], color=COLORS[m], marker=MARKERS[m])

        ax.set_xlabel("Iteration")

        if ax is ax2[0]:

            ax.set_ylabel(r"$\epsilon_t$")

        add_label(ax, lab)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig2.tight_layout()

    save_fig(fig2, out_dir, "fig2_eps_schedule")



                

    fig3, ax3 = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, ds, lab in zip(ax3, datasets, ["(a) HAR", "(b) GAS"]):

        for m in ["sahdpca_fb", "sahdpca_no_fb"]:

            try:

                df = load_schedule(runs_dir, ds, m, eps=1.0, seed=0)

            except FileNotFoundError as e:

                print(f"[WARN] {e}")

                continue

            noise_col = "noise_scale" if "noise_scale" in df.columns else "noise_scale_counts"

            ax.plot(df["iter"], df[noise_col], label=METHOD_DISPLAY[m], color=COLORS[m], marker=MARKERS[m])

        ax.set_xlabel("Iteration")

        if ax is ax3[0]:

            ax.set_ylabel("Laplace scale (b_t)")

        add_label(ax, lab)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig3.tight_layout()

    save_fig(fig3, out_dir, "fig3_noise_scale")





def plot_runtime(df: pd.DataFrame, out_dir: Path):

    methods = ["sahdpca_fb", "sahdpca_no_fb", "sahdpca_strong", "dp_kmeans", "kmeanspp_dp", "pca_dp"]

    agg = agg_mean_std(df, "runtime_sec", methods, ["HAR", "GAS"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, dataset, label in zip(axes, ["HAR", "GAS"], ["(a) HAR", "(b) GAS"]):

        for m in methods:

            sub = agg[(agg["dataset"] == dataset) & (agg["method"] == m)].sort_values("eps_tot")

            if sub.empty:

                continue

            ax.errorbar(

                sub["eps_tot"],

                sub["metric_mean"],

                yerr=sub["metric_std"],

                label=METHOD_DISPLAY.get(m, m),

                color=COLORS.get(m, "#333"),

                marker=MARKERS.get(m, "o"),

            )

        ax.set_xlabel(r"$\epsilon$")

        ax.set_ylabel("Runtime (sec)")

        add_label(ax, label)

        ax.grid(True, linestyle="--", color="#ccc", alpha=0.7)

        ax.legend(frameon=False)

    fig.tight_layout()

    save_fig(fig, out_dir, "fig9_runtime_vs_eps")





def main() -> None:

    parser = argparse.ArgumentParser(description="Generate publication-ready figures.")

    parser.add_argument("--results", type=str, default="results_summary.csv", help="由 collect_results 生成的 tidy CSV")

    parser.add_argument("--runs_dir", type=str, default="runs", help="per-iter 日志所在目录")

    parser.add_argument("--out_dir", type=str, default="figures", help="输出目录")

    args = parser.parse_args()



    set_style()

    results_path = Path(args.results)

    runs_dir = Path(args.runs_dir)

    out_dir = Path(args.out_dir)



    df = pd.read_csv(results_path)

            

    plot_schedule_figs(runs_dir, out_dir)



    trio = ["sahdpca_fb", "sahdpca_no_fb", "sahdpca_strong"]

    f1_trio = agg_mean_std(df, "macro_f1", trio, ["HAR", "GAS"])

    plot_f1_sse(f1_trio, "macro_f1", trio, out_dir, "fig4_f1_vs_eps", "Macro-F1", ylim=(0, 0.6))



    sse_trio = agg_mean_std(df, "sse", trio, ["HAR", "GAS"])

    plot_f1_sse(sse_trio, "sse", trio, out_dir, "fig5_sse_vs_eps", "SSE", use_sci=True)



    plot_collapse(df, out_dir)



    baselines = ["sahdpca_fb", "dp_kmeans", "kmeanspp_dp", "pca_dp"]

    f1_base = agg_mean_std(df, "macro_f1", baselines, ["HAR", "GAS"])

    plot_f1_sse(f1_base, "macro_f1", baselines, out_dir, "fig7_baseline_f1_vs_eps", "Macro-F1", ylim=(0, 0.6))



    sse_base = agg_mean_std(df, "sse", baselines, ["HAR", "GAS"])

    plot_f1_sse(sse_base, "sse", baselines, out_dir, "fig8_baseline_sse_vs_eps", "SSE", use_sci=True)



    plot_runtime(df, out_dir)

    print("[OK] 所有图已生成到", out_dir)





if __name__ == "__main__":

    main()

