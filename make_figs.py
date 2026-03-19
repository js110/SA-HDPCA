#!/usr/bin/env python3
"""
生成论文主文的图表（Matplotlib-only，符合期刊排版）。

数据来源：
  - 结果与历史：outputs/c_check_tuned_ext2/
      * results.csv                    （种子级最终指标）
      * history/*eps1.0_seed0.csv      （迭代级 eps_t / 噪声）
      * summary_tables/table_main_*.tex（基线汇总，含 DP 等）

输出：
  - 图：figs/*.pdf 和 figs/*.png
  - 表：tables/table_main.tex 与 tables/table_sig.tex
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 路径配置
ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs" / "c_check_tuned_ext2"
HISTORY_DIR = SOURCE_DIR / "history"
SUMMARY_DIR = SOURCE_DIR / "summary_tables"
FIG_DIR = ROOT / "figs"
TABLE_DIR = ROOT / "tables"

# 方法映射与样式
RAW_TO_STD_METHOD = {
    "sahdpca": "fb",
    "sahdpca_wo_feedback": "no-fb",
    "sahdpca_strong": "strong",
    "dp_kmeans": "DP-k-means",
    "kmeanspp_dp": "kmeans++-DP",
    "pca_dp": "PCA-DP",
}
RAW_FROM_STD_METHOD = {
    "fb": "sahdpca",
    "no-fb": "sahdpca_wo_feedback",
    "strong": "sahdpca_strong",
    "DP-k-means": "dp_kmeans",
    "kmeans++-DP": "kmeanspp_dp",
    "PCA-DP": "pca_dp",
}

METHOD_LABELS = {
    "fb": "fb",
    "no-fb": "no-fb",
    "strong": "strong",
    "DP-k-means": "DP-k-means",
    "kmeans++-DP": "kmeans++-DP",
    "PCA-DP": "PCA-DP",
    "SA-HDPCA (fb)": "SA-HDPCA (fb)",
}

METHOD_COLORS = {
    "fb": "#1f77b4",
    "no-fb": "#d62728",
    "strong": "#2ca02c",
    "DP-k-means": "#9467bd",
    "kmeans++-DP": "#8c564b",
    "PCA-DP": "#e377c2",
    "SA-HDPCA (fb)": "#1f77b4",
}

METHOD_MARKERS = {
    "fb": "o",
    "no-fb": "s",
    "strong": "^",
    "DP-k-means": "D",
    "kmeans++-DP": "v",
    "PCA-DP": "P",
    "SA-HDPCA (fb)": "o",
}

METHOD_LINESTYLES = {
    "fb": "-",
    "no-fb": "--",
    "strong": "-.",
    "DP-k-means": "-",
    "kmeans++-DP": "--",
    "PCA-DP": "-.",
    "SA-HDPCA (fb)": "-",
}


# 全局绘图风格
def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "errorbar.capsize": 3,
            "figure.dpi": 150,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="center",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    return pd.read_csv(path)


def load_final_metrics(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    df = df.rename(columns={"eps_tot": "epsilon", "runtime_ms_total": "runtime_ms"})
    df["method"] = df["method"].map(RAW_TO_STD_METHOD)
    df = df[df["method"].notna()].copy()
    df["dataset"] = df["dataset"].str.upper()
    return df


def load_schedule_data(history_dir: Path, eps_target: float = 1.0, seed: int = 0) -> Dict[str, Dict[str, pd.DataFrame]]:
    datasets = ["HAR", "GAS"]
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds in datasets:
        ds_lower = ds.lower()
        data[ds] = {}
        for method, raw_name in RAW_FROM_STD_METHOD.items():
            fname = f"{ds_lower}_{raw_name}_eps{eps_target}_seed{seed}.csv"
            fpath = history_dir / fname
            if not fpath.exists():
                print(f"[WARN] 缺少调度文件: {fpath}")
                continue
            df = pd.read_csv(fpath)
            df = df.sort_values("iter")
            data[ds][method] = df
    return data


def _extract_noise_scale(df: pd.DataFrame) -> pd.Series:
    if "noise_scale" in df.columns:
        return df["noise_scale"]
    if "noise_scale_counts" in df.columns:
        return df["noise_scale_counts"]
    if "noise_scale_sums" in df.columns:
        return df["noise_scale_sums"]
    raise KeyError("噪声尺度列未找到")


def load_and_aggregate_metric(
    df: pd.DataFrame,
    metric: str,
    methods: Iterable[str],
    datasets: Iterable[str] | None = None,
) -> pd.DataFrame:
    data = df[df["method"].isin(methods)].copy()
    if datasets is not None:
        data = data[data["dataset"].isin(datasets)]
    grouped = (
        data.groupby(["dataset", "method", "epsilon"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "metric_mean", "std": "metric_std"})
    )
    return grouped


def _select_epsilons(df: pd.DataFrame, dataset: str, desired: List[float]) -> List[float]:
    available = sorted(df.loc[df["dataset"] == dataset, "epsilon"].unique())
    if not available:
        print(f"[WARN] {dataset} 无可用 epsilon")
        return []
    chosen: List[float] = []
    for target in desired:
        nearest = min(available, key=lambda x: abs(x - target))
        if not math.isclose(nearest, target, rel_tol=1e-6, abs_tol=1e-6):
            print(f"[INFO] {dataset}: epsilon {target} 缺失，使用最接近的 {nearest}")
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def _format_mean_std(mean: float, std: float, kind: str) -> str:
    if pd.isna(mean):
        return "--"
    if kind == "f1":
        fmt = lambda v: f"{v:.3f}"
    elif kind == "sse":
        fmt = lambda v: f"{v:.3g}"
    elif kind == "runtime":
        fmt = lambda v: f"{v:.1f}"
    else:
        fmt = lambda v: f"{v:.3f}"
    std_str = fmt(std) if not pd.isna(std) else "0"
    return f"{fmt(mean)}±{std_str}"


# 解析基线 Tex 汇总（含 DP/kmeans++/PCA）
def parse_baseline_table(tex_path: Path, dataset: str) -> pd.DataFrame:
    if not tex_path.exists():
        print(f"[WARN] 基线表缺失: {tex_path}")
        return pd.DataFrame(
            columns=[
                "dataset",
                "method",
                "epsilon",
                "f1_mean",
                "f1_std",
                "sse_x_mean",
                "sse_x_std",
                "runtime_ms_mean",
                "runtime_ms_std",
            ]
        )
    rows = []
    with tex_path.open("r") as f:
        for line in f:
            if "&" not in line or line.lstrip().startswith("\\"):
                continue
            line = line.strip().rstrip("\\").replace("\\\\", "")
            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 12:
                continue
            (
                method,
                eps,
                f1_mean,
                f1_std,
                sse_mean,
                sse_std,
                _nek_mean,
                _nek_std,
                _mcr_mean,
                _mcr_std,
                runtime_mean,
                runtime_std,
            ) = parts[:12]
            method_map = {
                "dp_kmeans": "DP-k-means",
                "kmeanspp_dp": "kmeans++-DP",
                "pca_dp": "PCA-DP",
                "sahdpca": "fb",
                "sahdpca_wo_feedback": "no-fb",
                "sahdpca_strong": "strong",
            }
            method_std = method_map.get(method, method)
            try:
                eps_val = float(eps)
            except ValueError:
                continue
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "method": method_std,
                    "epsilon": eps_val,
                    "f1_mean": float(f1_mean),
                    "f1_std": float(f1_std),
                    "sse_x_mean": float(sse_mean),
                    "sse_x_std": float(sse_std),
                    "runtime_ms_mean": float(runtime_mean),
                    "runtime_ms_std": float(runtime_std),
                }
            )
    return pd.DataFrame(rows)


def load_baseline_summary(summary_dir: Path) -> pd.DataFrame:
    parts = []
    har = parse_baseline_table(summary_dir / "table_main_har.tex", "HAR")
    gas = parse_baseline_table(summary_dir / "table_main_gas.tex", "GAS")
    parts.extend([har, gas])
    fallback_dir = ROOT / "outputs" / "summary_tables"
    if fallback_dir != summary_dir:
        parts.append(parse_baseline_table(fallback_dir / "table_main_har.tex", "HAR"))
        parts.append(parse_baseline_table(fallback_dir / "table_main_gas.tex", "GAS"))
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(
        subset=["dataset", "method", "epsilon"], keep="first"
    )
    return combined


def _baseline_metric_df(baseline_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    mean_col = f"{metric}_mean" if metric != "runtime_ms" else "runtime_ms_mean"
    std_col = f"{metric}_std" if metric != "runtime_ms" else "runtime_ms_std"
    df = baseline_df.rename(columns={mean_col: "metric_mean", std_col: "metric_std"})[
        ["dataset", "method", "epsilon", "metric_mean", "metric_std"]
    ].copy()
    return df


# 绘图
def plot_cumulative_budget(schedule_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, dataset, label in zip(axes, ["HAR", "GAS"], ["(a) HAR", "(b) GAS"]):
        ds_data = schedule_data.get(dataset, {})
        for method in ["fb", "no-fb"]:
            sub = ds_data.get(method)
            if sub is None or sub.empty:
                print(f"[WARN] {dataset} 缺少 {method} 调度数据")
                continue
            cumulative = sub["eps_t"].cumsum()
            ax.plot(
                sub["iter"],
                cumulative,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
            )
        ax.set_xlabel("Iteration")
        if ax is axes[0]:
            ax.set_ylabel("Cumulative epsilon")
        add_panel_label(ax, label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig1_cumulative_budget")


def plot_eps_schedule(schedule_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, dataset, label in zip(axes, ["HAR", "GAS"], ["(a) HAR", "(b) GAS"]):
        ds_data = schedule_data.get(dataset, {})
        for method in ["fb", "no-fb", "strong"]:
            sub = ds_data.get(method)
            if sub is None or sub.empty:
                print(f"[WARN] {dataset} 缺少 {method} 调度数据")
                continue
            ax.plot(
                sub["iter"],
                sub["eps_t"],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
            )
        ax.set_xlabel("Iteration")
        if ax is axes[0]:
            ax.set_ylabel(r"$\epsilon_t$")
        add_panel_label(ax, label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig2_eps_schedule")


def plot_noise_scale(schedule_data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, dataset, label in zip(axes, ["HAR", "GAS"], ["(a) HAR", "(b) GAS"]):
        ds_data = schedule_data.get(dataset, {})
        for method in ["fb", "no-fb"]:
            sub = ds_data.get(method)
            if sub is None or sub.empty:
                print(f"[WARN] {dataset} 缺少 {method} 噪声数据")
                continue
            noise = _extract_noise_scale(sub)
            ax.plot(
                sub["iter"],
                noise,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
            )
        ax.set_xlabel("Iteration")
        if ax is axes[0]:
            ax.set_ylabel("Laplace scale (counts)")
        add_panel_label(ax, label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig3_noise_scale")


def _set_f1_ylim(ax: plt.Axes, max_val: float) -> None:
    upper = 0.6
    if max_val > upper:
        upper = min(1.0, max_val + 0.05)
    ax.set_ylim(0, upper)


def _plot_metric_vs_eps(
    df: pd.DataFrame,
    metric: str,
    methods: List[str],
    ylabel: str,
    filename: str,
    panel_labels: Tuple[str, str],
    ylim: Tuple[float, float] | None = None,
    label_override: Dict[str, str] | None = None,
    use_sci: bool = False,
    scale_sse: bool = False,
    show_errorbars: bool = True,
    use_panel_label: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=False)
    y_label = ylabel
    df_plot = df.copy()
    if scale_sse and metric in {"sse_x", "sse"}:
        max_val = df_plot["metric_mean"].abs().max()
        if not pd.isna(max_val) and max_val > 0:
            if max_val >= 1e6:
                exp = 6
            elif max_val >= 1e5:
                exp = 5
            else:
                exp = 0
            if exp > 0:
                factor = 10 ** exp
                df_plot["metric_mean"] = df_plot["metric_mean"] / factor
                df_plot["metric_std"] = df_plot["metric_std"] / factor
                y_label = f"{ylabel} (x10^{exp})"
    datasets = ["HAR", "GAS"]
    for ax, dataset, label in zip(axes, datasets, panel_labels):
        for method in methods:
            df_method = df_plot[
                (df_plot["dataset"] == dataset) & (df_plot["method"] == method)
            ].sort_values("epsilon")
            if df_method.empty:
                print(f"[WARN] {dataset} 缺少 {method} 的 {metric}")
                continue
            lbl = label_override.get(method, METHOD_LABELS.get(method, method)) if label_override else METHOD_LABELS.get(method, method)
            if show_errorbars:
                ax.errorbar(
                    df_method["epsilon"],
                    df_method["metric_mean"],
                    yerr=df_method["metric_std"],
                    label=lbl,
                    color=METHOD_COLORS.get(method, "#333333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linestyle=METHOD_LINESTYLES.get(method, "-"),
                    linewidth=1.8,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    capsize=3,
                )
            else:
                ax.plot(
                    df_method["epsilon"],
                    df_method["metric_mean"],
                    label=lbl,
                    color=METHOD_COLORS.get(method, "#333333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linestyle=METHOD_LINESTYLES.get(method, "-"),
                    linewidth=1.8,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                )
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(y_label)
        if use_panel_label:
            add_panel_label(ax, label)
        else:
            ax.set_title(label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if metric == "f1":
            max_val = df_plot[df_plot["dataset"] == dataset]["metric_mean"].max()
            _set_f1_ylim(ax, max_val if not pd.isna(max_val) else 0.6)
        if use_sci:
            y_max = df_plot[df_plot["dataset"] == dataset]["metric_mean"].abs().max()
            if not pd.isna(y_max) and (y_max > 1e3 or y_max < 1e-2):
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(methods),
            frameon=True,
            bbox_to_anchor=(0.5, 1.05),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    save_fig(fig, filename)


def plot_collapse_metrics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharey=False)
    panels = [
        ("HAR", "non_empty_k_final", "(a) HAR non-empty K"),
        ("HAR", "max_cluster_ratio_final", "(b) HAR max cluster ratio"),
        ("GAS", "non_empty_k_final", "(c) GAS non-empty K"),
        ("GAS", "max_cluster_ratio_final", "(d) GAS max cluster ratio"),
    ]
    methods = ["fb", "no-fb", "strong"]
    for ax, (dataset, metric, label) in zip(axes.flat, panels):
        agg = load_and_aggregate_metric(df, metric, methods, datasets=[dataset])
        for method in methods:
            sub = agg[(agg["dataset"] == dataset) & (agg["method"] == method)].sort_values(
                "epsilon"
            )
            if sub.empty:
                print(f"[WARN] {dataset} 缺少 {method} 的 {metric}")
                continue
            ax.plot(
                sub["epsilon"],
                sub["metric_mean"],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                linewidth=1.8,
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
        ax.set_xlabel(r"$\epsilon$")
        if "non_empty" in metric:
            ax.set_ylabel("Non-empty clusters")
        else:
            ax.set_ylabel("Max cluster ratio")
        add_panel_label(ax, label)
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(methods),
            frameon=True,
            bbox_to_anchor=(0.5, 1.05),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    save_fig(fig, "fig6_collapse_vs_eps")


# 表格
def generate_table_main(final_df: pd.DataFrame, baseline_summary: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    desired_eps = [0.5, 1.0, 1.5]
    datasets = ["HAR", "GAS"]
    baseline_methods = ["DP-k-means", "kmeans++-DP", "PCA-DP"]
    sdp_methods = ["fb", "no-fb"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & $\\epsilon$ & Method & Macro-F1 & SSE$_x$ & Runtime (ms) \\\\",
        "\\midrule",
    ]
    eps_parts = [final_df[["dataset", "epsilon"]]]
    if not baseline_summary.empty:
        eps_parts.append(baseline_summary[["dataset", "epsilon"]])
    combined_eps_df = pd.concat(eps_parts, ignore_index=True).dropna()
    for dataset in datasets:
        eps_list = _select_epsilons(combined_eps_df, dataset, desired_eps)
        if not eps_list:
            continue
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{dataset}}}}} \\\\")
        for eps in eps_list:
            for method in baseline_methods + sdp_methods:
                sub = final_df[
                    (final_df["dataset"] == dataset)
                    & (final_df["method"] == method)
                    & (np.isclose(final_df["epsilon"], eps))
                ]
                if sub.empty and method in baseline_methods and not baseline_summary.empty:
                    # fallback旧表
                    sub_base = baseline_summary[
                        (baseline_summary["dataset"] == dataset)
                        & (np.isclose(baseline_summary["epsilon"], eps))
                        & (baseline_summary["method"] == method)
                    ]
                    if not sub_base.empty:
                        row = sub_base.iloc[0]
                        f1_str = _format_mean_std(row["f1_mean"], row["f1_std"], "f1")
                        sse_str = _format_mean_std(row["sse_x_mean"], row["sse_x_std"], "sse")
                        rt_str = _format_mean_std(row["runtime_ms_mean"], row["runtime_ms_std"], "runtime")
                    else:
                        print(f"[WARN] 缺少 {dataset} {method} epsilon={eps}")
                        f1_str = sse_str = rt_str = "--"
                elif sub.empty:
                    print(f"[WARN] 缺少 {dataset} {method} epsilon={eps}")
                    f1_str = sse_str = rt_str = "--"
                else:
                    f1 = sub["f1"]
                    sse = sub["sse_x"]
                    runtime = sub["runtime_ms"]
                    f1_str = _format_mean_std(f1.mean(), f1.std(ddof=1), "f1")
                    sse_str = _format_mean_std(sse.mean(), sse.std(ddof=1), "sse")
                    rt_str = _format_mean_std(runtime.mean(), runtime.std(ddof=1), "runtime")
                lines.append(
                    f"{dataset} & {eps:.2f} & {METHOD_LABELS.get(method, method)} & "
                    f"{f1_str} & {sse_str} & {rt_str} \\\\"
                )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    table_path = TABLE_DIR / "table_main.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def generate_table_sig(df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    datasets = ["HAR", "GAS"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & $\\epsilon$ & mean diff & $p_\\text{ttest}$ & $p_\\text{MWU}$ \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        epsilons = sorted(df.loc[df["dataset"] == dataset, "epsilon"].unique())
        for eps in epsilons:
            fb_rows = df[
                (df["dataset"] == dataset) & (df["method"] == "fb") & (np.isclose(df["epsilon"], eps))
            ].copy()
            nofb_rows = df[
                (df["dataset"] == dataset) & (df["method"] == "no-fb") & (np.isclose(df["epsilon"], eps))
            ].copy()
            if fb_rows.empty or nofb_rows.empty:
                print(f"[WARN] 统计检验缺数据: {dataset}, epsilon={eps}")
                continue
            fb_rows = fb_rows.sort_values("seed")
            nofb_rows = nofb_rows.sort_values("seed")
            fb_seeds = fb_rows["seed"].to_numpy()
            nofb_seeds = nofb_rows["seed"].to_numpy()
            common_seeds = np.intersect1d(fb_seeds, nofb_seeds)
            if common_seeds.size >= 2:
                fb_vals = fb_rows[fb_rows["seed"].isin(common_seeds)]["f1"].to_numpy()
                nofb_vals = nofb_rows[nofb_rows["seed"].isin(common_seeds)]["f1"].to_numpy()
                _, p_ttest = stats.ttest_rel(fb_vals, nofb_vals, nan_policy="omit")
            else:
                fb_vals = fb_rows["f1"].to_numpy()
                nofb_vals = nofb_rows["f1"].to_numpy()
                _, p_ttest = stats.ttest_ind(
                    fb_vals, nofb_vals, equal_var=False, nan_policy="omit"
                )
            try:
                _, p_mwu = stats.mannwhitneyu(
                    fb_vals, nofb_vals, alternative="two-sided", method="auto"
                )
            except ValueError:
                p_mwu = np.nan
            mean_diff = float(np.nanmean(fb_vals) - np.nanmean(nofb_vals))
            lines.append(
                f"{dataset} & {eps:.2f} & {mean_diff:.3f} & {p_ttest:.2e} & "
                f"{(p_mwu if not math.isnan(p_mwu) else np.nan):.2e} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    table_path = TABLE_DIR / "table_sig.tex"
    table_path.write_text("\n".join(lines))
    return table_path


def main() -> None:
    set_pub_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    final_metrics_path = SOURCE_DIR / "results.csv"
    final_metrics = load_final_metrics(final_metrics_path)

    # 调度与噪声（epsilon=1.0, seed=0）
    schedule_data = load_schedule_data(HISTORY_DIR, eps_target=1.0, seed=0)
    plot_cumulative_budget(schedule_data)
    plot_eps_schedule(schedule_data)
    plot_noise_scale(schedule_data)

    # Macro-F1 / SSE：fb, no-fb, strong
    trio_methods = ["fb", "no-fb", "strong"]
    f1_trio = load_and_aggregate_metric(final_metrics, "f1", trio_methods)
    _plot_metric_vs_eps(
        f1_trio,
        metric="f1",
        methods=trio_methods,
        ylabel="Macro-F1",
        filename="fig4_f1_vs_eps",
        panel_labels=("(a) HAR", "(b) GAS"),
        ylim=(0, 0.6),
        show_errorbars=False,
        use_panel_label=True,
    )
    sse_trio = load_and_aggregate_metric(final_metrics, "sse_x", trio_methods)
    _plot_metric_vs_eps(
        sse_trio,
        metric="sse_x",
        methods=trio_methods,
        ylabel="SSE_x",
        filename="fig5_sse_vs_eps",
        panel_labels=("(a) HAR", "(b) GAS"),
        use_sci=False,
        scale_sse=True,
        show_errorbars=False,
        use_panel_label=True,
    )

    # 崩塌鲁棒性
    plot_collapse_metrics(final_metrics)

    # 基线：DP/kmeans++/PCA + fb
    baseline_summary = load_baseline_summary(SUMMARY_DIR)
    baseline_methods = ["DP-k-means", "kmeans++-DP", "PCA-DP"]
    f1_baseline = load_and_aggregate_metric(final_metrics, "f1", ["fb"] + baseline_methods)
    label_override = {"fb": "SA-HDPCA (fb)"}
    _plot_metric_vs_eps(
        f1_baseline,
        metric="f1",
        methods=["fb"] + baseline_methods,
        ylabel="Macro-F1",
        filename="fig7_baseline_f1_vs_eps",
        panel_labels=("(a) HAR", "(b) GAS"),
        ylim=(0, 0.6),
        label_override=label_override,
    )

    sse_base = load_and_aggregate_metric(final_metrics, "sse_x", ["fb"] + baseline_methods)
    _plot_metric_vs_eps(
        sse_base,
        metric="sse_x",
        methods=["fb"] + baseline_methods,
        ylabel="SSE_x",
        filename="fig8_baseline_sse_vs_eps",
        panel_labels=("(a) HAR", "(b) GAS"),
        use_sci=False,
        label_override={"fb": "SA-HDPCA (fb)"},
        scale_sse=True,
    )

    # 表格
    table_main_path = generate_table_main(final_metrics, baseline_summary)
    table_sig_path = generate_table_sig(final_metrics)

    generated = [
        "figs/fig1_cumulative_budget.(pdf|png)",
        "figs/fig2_eps_schedule.(pdf|png)",
        "figs/fig3_noise_scale.(pdf|png)",
        "figs/fig4_f1_vs_eps.(pdf|png)",
        "figs/fig5_sse_vs_eps.(pdf|png)",
        "figs/fig6_collapse_vs_eps.(pdf|png)",
        "figs/fig7_baseline_f1_vs_eps.(pdf|png)",
        "figs/fig8_baseline_sse_vs_eps.(pdf|png)",
        str(table_main_path.relative_to(ROOT)),
        str(table_sig_path.relative_to(ROOT)),
    ]
    print("生成的文件：")
    for item in generated:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
