import os

from typing import Dict, Iterable, List, Tuple



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





RESULTS_PATH = "outputs/c_check_tuned_ext2/results.csv"

                                 

OUT_DIR = "outputs/paper_figs_color"

BASELINE_OUT_DIR = "outputs/paper_baselines"

SUMMARY_FILES = {

    "har": "outputs/summary_tables/har_eps05_1.csv",

    "gas": "outputs/summary_tables/gas_eps05_1.csv",

}



                               

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(

    {

        "font.size": 11,

        "axes.labelsize": 12,

        "axes.titlesize": 12,

        "legend.fontsize": 10,

        "lines.linewidth": 2.0,

        "lines.markersize": 6,

        "figure.dpi": 120,

    }

)



METHOD_LABELS = {

    "sahdpca": "fb",

    "sahdpca_wo_feedback": "no-fb",

    "sahdpca_strong": "strong",

}



METHOD_MARKERS = {

    "sahdpca": "o",                 

    "sahdpca_wo_feedback": "s",        

    "sahdpca_strong": "^",              

}



                              

PALETTE = {

    "sahdpca": "#1b9e77",                      

    "sahdpca_wo_feedback": "#d95f02",          

    "sahdpca_strong": "#7570b3",               

}



LINESTYLES = {

    "sahdpca": "-",

    "sahdpca_wo_feedback": "--",

    "sahdpca_strong": ":",

}





def ensure_dir(path: str) -> None:

    os.makedirs(path, exist_ok=True)





def load_results(results_path: str) -> pd.DataFrame:

    df = pd.read_csv(results_path)

    if "history_path" in df.columns:

        df["history_path"] = df["history_path"].apply(os.path.normpath)

    return df





def load_summary(dataset: str) -> pd.DataFrame:

    path = SUMMARY_FILES.get(dataset)

    if path is None or not os.path.exists(path):

        raise FileNotFoundError(f"Summary file not found for dataset {dataset}: {path}")

    return pd.read_csv(path)





def format_val(mean: float, std: float, kind: str) -> str:

    if kind == "f1":

        return f"{mean:.3f}±{std:.3f}"

    if kind == "sse":

                                    

        return f"{mean:.2e}±{std:.1e}"

    if kind == "runtime":

        return f"{mean:.0f}±{std:.0f}"

    return f"{mean:.3f}±{std:.3f}"





def make_main_tables() -> None:

    ensure_dir(BASELINE_OUT_DIR)

    methods_order = ["dp_kmeans", "kmeanspp_dp", "pca_dp", "sahdpca_wo_feedback", "sahdpca"]

    method_labels = {

        "dp_kmeans": "DP-k-means",

        "kmeanspp_dp": "kmeans++-DP",

        "pca_dp": "PCA-DP",

        "sahdpca_wo_feedback": "SA-HDPCA (no-fb)",

        "sahdpca": "SA-HDPCA (fb)",

    }

    eps_vals = [0.5, 1.0]

    metrics = [("f1_mean", "f1_std", "f1"), ("sse_x_mean", "sse_x_std", "sse"), ("runtime_ms_total_mean", "runtime_ms_total_std", "runtime")]



    for dataset in ["har", "gas"]:

        df = load_summary(dataset)

        rows = []

        for m in methods_order:

            row = {"Method": method_labels[m]}

            for eps in eps_vals:

                sub = df[(df["method"] == m) & (df["eps_tot"] == eps)]

                if sub.empty:

                                  

                    row[f"F1@{eps}"] = "-"

                    row[f"SSE@{eps}"] = "-"

                    row[f"Runtime@{eps}(ms)"] = "-"

                    continue

                for (mean_col, std_col, kind) in metrics:

                    mean = float(sub.iloc[0][mean_col])

                    std = float(sub.iloc[0][std_col])

                    key = {"f1": f"F1@{eps}", "sse": f"SSE@{eps}", "runtime": f"Runtime@{eps}(ms)"}[kind]

                    row[key] = format_val(mean, std, kind)

            rows.append(row)

        out_df = pd.DataFrame(rows)

        csv_path = os.path.join(BASELINE_OUT_DIR, f"table_main_{dataset}.csv")

        out_df.to_csv(csv_path, index=False)



                             

        cols = ["Method"]

        for eps in eps_vals:

            cols.extend([f"F1@{eps}", f"SSE@{eps}", f"Runtime@{eps}(ms)"])

        latex_lines = ["\\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}"]

        latex_lines.append(" & ".join(cols) + " \\\\")

        latex_lines.append("\\hline")

        for _, r in out_df.iterrows():

            latex_lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")

        latex_lines.append("\\end{tabular}")

        tex_path = os.path.join(BASELINE_OUT_DIR, f"table_main_{dataset}.tex")

        with open(tex_path, "w") as f:

            f.write("\n".join(latex_lines))





def plot_baseline_curve(dataset: str, out_path: str) -> None:

    ensure_dir(os.path.dirname(out_path))

    df = load_summary(dataset)

    methods = ["dp_kmeans", "kmeanspp_dp", "pca_dp", "sahdpca"]

    labels = {

        "dp_kmeans": "DP-k-means",

        "kmeanspp_dp": "kmeans++-DP",

        "pca_dp": "PCA-DP",

        "sahdpca": "SA-HDPCA (fb)",

    }

    colors = {

        "dp_kmeans": "#1f77b4",

        "kmeanspp_dp": "#ff7f0e",

        "pca_dp": "#7f7f7f",

        "sahdpca": "#1b9e77",

    }

    linestyles = {

        "dp_kmeans": "-",

        "kmeanspp_dp": "--",

        "pca_dp": ":",

        "sahdpca": "-.",

    }



    fig, ax = plt.subplots(figsize=(6, 3.8))

    for m in methods:

        sub = df[df["method"] == m].sort_values("eps_tot")

        ax.plot(

            sub["eps_tot"],

            sub["f1_mean"],

            marker=METHOD_MARKERS.get(m, "o"),

            color=colors[m],

            linestyle=linestyles[m],

            label=labels[m],

        )

    ax.text(0.02, 0.95, dataset.upper(), transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    ax.set_xlabel("epsilon")

    ax.set_ylabel("Macro-F1")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def get_history(df: pd.DataFrame, dataset: str, method: str, eps: float, seed: int) -> pd.DataFrame:

    row = df[

        (df["dataset"] == dataset)

        & (df["method"] == method)

        & (df["eps_tot"] == eps)

        & (df["seed"] == seed)

    ]

    if row.empty:

        raise ValueError(f"history not found for {dataset}/{method}/eps={eps}/seed={seed}")

    path = row.iloc[0]["history_path"]

    if not os.path.exists(path):

        raise FileNotFoundError(f"missing history file: {path}")

    return pd.read_csv(path)





def plot_cumulative_budget(df: pd.DataFrame, datasets: Iterable[str], eps: float, seed: int, out_path: str) -> None:

    fig, axes = plt.subplots(1, len(list(datasets)), figsize=(10, 4), sharey=True)

    if not isinstance(axes, np.ndarray):

        axes = np.array([axes])

    for ax, dataset in zip(axes, datasets):

        hist_fb = get_history(df, dataset, "sahdpca", eps, seed)

        hist_nofb = get_history(df, dataset, "sahdpca_wo_feedback", eps, seed)

        totals = {}

        for hist, method in [(hist_fb, "sahdpca"), (hist_nofb, "sahdpca_wo_feedback")]:

            cum = hist["eps_used"].cumsum()

            totals[method] = cum.iloc[-1]

            ax.plot(

                hist["iter"],

                cum,

                marker=METHOD_MARKERS[method],

                color=PALETTE[method],

                linestyle=LINESTYLES[method],

                label=METHOD_LABELS[method],

            )

        ax.set_xlabel("iter")

        ax.grid(True, alpha=0.3)

        max_total = max(totals.values())

        upper = max(1.0, max_total * 1.05)

        ax.set_ylim(0, upper)

        ax.set_yticks(np.linspace(0, upper, 6))

        ax.set_title(dataset.upper())

    axes[0].set_ylabel("cumulative epsilon")

    axes[0].legend(title="budget mode")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def plot_eps_schedule(df: pd.DataFrame, datasets: Iterable[str], eps: float, seed: int, out_path: str) -> None:

    fig, axes = plt.subplots(1, len(list(datasets)), figsize=(10, 4), sharey=True)

    if not isinstance(axes, np.ndarray):

        axes = np.array([axes])

    target_methods = ["sahdpca_wo_feedback", "sahdpca", "sahdpca_strong"]

    for ax, dataset in zip(axes, datasets):

        for method in target_methods:

            hist = get_history(df, dataset, method, eps, seed)

            ax.plot(

                hist["iter"],

                hist["eps_t"],

                marker=METHOD_MARKERS.get(method, "o"),

                color=PALETTE.get(method, "#444444"),

                linestyle=LINESTYLES.get(method, "-"),

                label=METHOD_LABELS.get(method, method),

            )

        ax.set_xlabel("iter")

        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\epsilon_t$")

    axes[0].legend(title="method")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def plot_noise_scale(df: pd.DataFrame, datasets: Iterable[str], eps: float, seed: int, out_path: str) -> None:

    fig, axes = plt.subplots(1, len(list(datasets)), figsize=(10, 4), sharey=True)

    if not isinstance(axes, np.ndarray):

        axes = np.array([axes])

    for ax, dataset in zip(axes, datasets):

        for method in ["sahdpca", "sahdpca_wo_feedback"]:

            hist = get_history(df, dataset, method, eps, seed)

            ax.plot(

                hist["iter"],

                hist["noise_scale_counts"],

                marker=METHOD_MARKERS[method],

                color=PALETTE[method],

                linestyle=LINESTYLES[method],

                label=METHOD_LABELS[method],

            )

        ax.set_xlabel("iter")

        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Laplace scale (counts)")

    axes[0].legend(title="budget mode")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def _aggregate_metric(

    df: pd.DataFrame, dataset: str, metric: str, methods: List[str]

) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    subset = df[df["dataset"] == dataset]

    for method in methods:

        data = subset[subset["method"] == method].groupby("eps_tot")[metric].agg(["mean", "std"]).reset_index()

        out[method] = (

            data["eps_tot"].values,

            data["mean"].values,

            data["std"].values,

        )

    return out





def plot_metric_vs_eps(df: pd.DataFrame, metric: str, out_path: str, title: str) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    datasets = ["har", "gas"]

    methods = ["sahdpca", "sahdpca_wo_feedback", "sahdpca_strong"]

    for ax, dataset in zip(axes, datasets):

        agg = _aggregate_metric(df, dataset, metric, methods)

        for method in methods:

            eps, mean, std = agg[method]

            ax.plot(

                eps,

                mean,

                marker=METHOD_MARKERS[method],

                color=PALETTE[method],

                linestyle=LINESTYLES[method],

                label=METHOD_LABELS[method],

            )

        ax.set_xlabel("epsilon")

        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric.replace("_", " "))

    axes[0].legend(title="method")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def plot_collapse_metrics(df: pd.DataFrame, out_path: str) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey="row")

    datasets = ["har", "gas"]

    methods = ["sahdpca", "sahdpca_wo_feedback", "sahdpca_strong"]

    metrics = ["non_empty_k_final", "max_cluster_ratio_final"]

    for row_idx, dataset in enumerate(datasets):

        agg_non_empty = _aggregate_metric(df, dataset, metrics[0], methods)

        agg_ratio = _aggregate_metric(df, dataset, metrics[1], methods)

        for method in methods:

            eps, mean, std = agg_non_empty[method]

            axes[row_idx, 0].plot(

                eps,

                mean,

                marker=METHOD_MARKERS[method],

                color=PALETTE[method],

                linestyle=LINESTYLES[method],

                label=METHOD_LABELS[method],

            )

            eps, mean, std = agg_ratio[method]

            axes[row_idx, 1].plot(

                eps,

                mean,

                marker=METHOD_MARKERS[method],

                color=PALETTE[method],

                linestyle=LINESTYLES[method],

                label=METHOD_LABELS[method],

            )

        axes[row_idx, 0].set_xlabel("epsilon")

        axes[row_idx, 1].set_xlabel("epsilon")

        axes[row_idx, 0].set_ylabel("non-empty k")

        axes[row_idx, 1].set_ylabel("max cluster ratio")

        axes[row_idx, 0].grid(True, alpha=0.3)

        axes[row_idx, 1].grid(True, alpha=0.3)

    axes[0, 0].legend(title="method")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def main() -> None:

    ensure_dir(OUT_DIR)

    df = load_results(RESULTS_PATH)



                                               

    plot_cumulative_budget(

        df,

        datasets=["har", "gas"],

        eps=1.0,

        seed=0,

        out_path=os.path.join(OUT_DIR, "fig1_cumulative_budget.png"),

    )



                                                        

    plot_eps_schedule(

        df,

        datasets=["har", "gas"],

        eps=1.0,

        seed=0,

        out_path=os.path.join(OUT_DIR, "fig2_eps_schedule.png"),

    )



                                                         

    plot_noise_scale(

        df,

        datasets=["har", "gas"],

        eps=1.0,

        seed=0,

        out_path=os.path.join(OUT_DIR, "fig3_noise_scale.png"),

    )



                                             

    plot_metric_vs_eps(

        df,

        metric="f1",

        out_path=os.path.join(OUT_DIR, "fig4_f1_vs_eps.png"),

        title="F1 vs epsilon",

    )



                                                     

    plot_metric_vs_eps(

        df,

        metric="sse_x",

        out_path=os.path.join(OUT_DIR, "fig5_sse_vs_eps.png"),

        title="Reconstruction SSE vs epsilon",

    )



                                         

    plot_collapse_metrics(

        df,

        out_path=os.path.join(OUT_DIR, "fig6_collapse_vs_eps.png"),

    )



                                           

    ensure_dir(BASELINE_OUT_DIR)

    make_main_tables()

    plot_baseline_curve(

        dataset="har",

        out_path=os.path.join(BASELINE_OUT_DIR, "har_baseline_f1_vs_eps.png"),

    )





if __name__ == "__main__":

    main()

