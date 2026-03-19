#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make comparison plots for 2024/2025 baselines.")
    parser.add_argument("--input", type=str, default="outputs/compare_2024_2025/raw_results.csv", help="Path to raw_results.csv")
    parser.add_argument("--schedules_dir", type=str, default="outputs/compare_2024_2025/schedules", help="Dir containing schedule json files")
    parser.add_argument("--out_dir", type=str, default="outputs/compare_2024_2025/figures", help="Output dir for figures")
    return parser.parse_args()


def set_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def agg(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["method", "eps_tot"])[metric].agg(["mean", "std"]).reset_index()
    return g


def line_with_error(ax, data: pd.DataFrame, metric: str, methods: list[str], title: str):
    for m in methods:
        dm = data[data["method"] == m].sort_values("eps_tot")
        if dm.empty:
            continue
        ax.errorbar(dm["eps_tot"], dm["mean"], yerr=dm["std"], marker="o", capsize=3, label=m)
    ax.set_xlabel("eps_tot")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def save_fig(fig, out_base: Path):
    fig.tight_layout()
    png_path = Path(f"{out_base}.png")
    pdf_path = Path(f"{out_base}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_metric_single(df: pd.DataFrame, dataset: str, metric: str, fname: str, methods: list[str], out_dir: Path):
    data = agg(df, dataset, metric)
    if data.empty:
        return
    fig, ax = plt.subplots()
    line_with_error(ax, data, metric, methods, f"{dataset} {metric} vs eps")
    save_fig(fig, out_dir / fname)


def plot_metric_dual(df: pd.DataFrame, datasets: list[str], metric: str, fname: str, methods: list[str], out_dir: Path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        data = agg(df, ds, metric)
        if data.empty:
            continue
        line_with_error(ax, data, metric, methods, f"{ds} {metric} vs eps")
    save_fig(fig, out_dir / fname)


def load_schedule(schedules_dir: Path, dataset: str, method: str, eps: float) -> np.ndarray | None:
    path = schedules_dir / f"{dataset}_{method}_eps{eps}.json"
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    return np.array(obj.get("schedule", []), dtype=float)


def plot_schedules(schedules_dir: Path, dataset: str, eps: float, out_dir: Path, fname: str):
    sched_sa = load_schedule(schedules_dir, dataset, "sahdpca", eps)
    sched_uniform = load_schedule(schedules_dir, dataset, "uniform", eps)
    sched_gapbas = load_schedule(schedules_dir, dataset, "gapbas", eps)
    t_len = max(
        len(sched_sa) if sched_sa is not None else 0,
        len(sched_uniform) if sched_uniform is not None else 0,
        len(sched_gapbas) if sched_gapbas is not None else 0,
    )
    if t_len == 0:
        return
    t_axis = np.arange(1, t_len + 1)
    fig, ax = plt.subplots()
    if sched_sa is not None:
        ax.plot(t_axis, sched_sa, marker="o", label="SA-HDPCA")
    if sched_uniform is not None:
        ax.plot(t_axis, sched_uniform, marker="o", label="Uniform")
    if sched_gapbas is not None:
        ax.plot(t_axis, sched_gapbas, marker="o", label="GAPBAS")
    ax.set_xlabel("t")
    ax.set_ylabel("eps_t")
    ax.set_title(f"{dataset} schedules (eps={eps})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_dir / fname)


def plot_ablation(df: pd.DataFrame, dataset: str, eps: float, out_dir: Path, fname: str):
    methods = ["SA-HDPCA", "SAHDPCA-wo-M1", "SAHDPCA-wo-M2", "SAHDPCA-wo-M3"]
    sub = df[(df["dataset"] == dataset) & (df["eps_tot"] == eps) & (df["method"].isin(methods))]
    if sub.empty:
        return
    agg = sub.groupby("method")["macro_f1"].agg(["mean", "std"]).reindex(methods)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(methods)), agg["mean"].values, yerr=agg["std"].values, capsize=4)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"{dataset} ablation (eps={eps})")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_dir / fname)


def make_table(df: pd.DataFrame, out_path: Path, eps_points: list[float], methods: list[str]):
    rows = []
    metrics = ["macro_f1", "nmi", "ari", "sse", "non_empty_clusters", "max_cluster_ratio", "runtime_seconds"]
    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        for method in methods:
            for eps in eps_points:
                filt = sub[(sub["method"] == method) & (sub["eps_tot"] == eps)]
                if filt.empty:
                    continue
                row = {"dataset": dataset, "method": method, "eps_tot": eps}
                for m in metrics:
                    mean = filt[m].mean()
                    std = filt[m].std()
                    row[m] = f"{mean:.3f}±{std:.3f}"
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    set_style()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    schedules_dir = Path(args.schedules_dir)

    df = pd.read_csv(args.input)
    core_methods = ["SA-HDPCA", "Uniform", "GAPBAS", "DBDP", "DPDP", "DP-KMEANS", "KMEANSPP-DP", "PCA-DP"]

    # Fig1/2 Macro-F1 vs eps
    plot_metric_single(df, "HAR", "macro_f1", "Fig1_macroF1_vs_eps_HAR", core_methods, out_dir)
    plot_metric_single(df, "GAS", "macro_f1", "Fig2_macroF1_vs_eps_GAS", core_methods, out_dir)
    # Fig3/4 NMI vs eps
    plot_metric_single(df, "HAR", "nmi", "Fig3_NMI_vs_eps_HAR", core_methods, out_dir)
    plot_metric_single(df, "GAS", "nmi", "Fig4_NMI_vs_eps_GAS", core_methods, out_dir)
    # Fig5 schedule eps=0.8 (HAR)
    plot_schedules(schedules_dir, "HAR", 0.8, out_dir, "Fig5_schedule_eps_t_vs_t_eps0.8")
    # Fig6 non-empty clusters (HAR+GAS, one figure with two panels)
    plot_metric_dual(df, ["HAR", "GAS"], "non_empty_clusters", "Fig6_non_empty_clusters_vs_eps", core_methods, out_dir)
    # Fig7 runtime (HAR+GAS)
    plot_metric_dual(df, ["HAR", "GAS"], "runtime_seconds", "Fig7_runtime_vs_eps", core_methods, out_dir)
    # Fig8 ablation (HAR eps=0.8)
    plot_ablation(df, "HAR", 0.8, out_dir, "Fig8_ablation_macroF1_eps0.8")

    # Table
    table_path = out_dir.parent / "Table_key_eps_points.csv"
    make_table(df, table_path, eps_points=[0.5, 0.8, 1.0], methods=core_methods + ["SAHDPCA-wo-M1", "SAHDPCA-wo-M2", "SAHDPCA-wo-M3"])

    print(f"[OK] plots saved to {out_dir}")
    print(f"[OK] table saved to {table_path}")


if __name__ == "__main__":
    main()
