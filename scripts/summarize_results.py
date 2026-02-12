import argparse

import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





def ensure_dir(path: str):

    os.makedirs(path, exist_ok=True)





def summarize(df: pd.DataFrame, dataset: str, eps_list, out_dir: str):

    metrics = ["f1", "sse_x", "non_empty_k_final", "max_cluster_ratio_final", "runtime_ms_total"]

    subset = df[(df["dataset"] == dataset)]

    if eps_list:

        subset = subset[subset["eps_tot"].isin(eps_list)]

    if subset.empty:

        return None

    agg = subset.groupby(["method", "eps_tot"])[metrics].agg(["mean", "std"])

    agg.columns = ["_".join(col).strip() for col in agg.columns.to_flat_index()]

    csv_path = os.path.join(out_dir, f"{dataset}_summary.csv")

    agg.to_csv(csv_path)

    return agg





def latex_table(agg: pd.DataFrame, dataset: str, out_dir: str):

    if agg is None or agg.empty:

        return

    df_flat = agg.reset_index()

    cols = [str(c) for c in df_flat.columns]

    header = " & ".join(cols) + r" \\"

    body = []

    for _, row in df_flat.iterrows():

        vals = []

        for c in cols:

            val = row[c]

            if isinstance(val, float):

                vals.append(f"{val:.4g}")

            else:

                vals.append(str(val))

        body.append(" & ".join(vals) + r" \\")

    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", header]

    lines.extend(body)

    lines.append("\\end{tabular}")

    path = os.path.join(out_dir, f"table_main_{dataset}.tex")

    with open(path, "w") as f:

        f.write("\n".join(lines))





def plot_metric(df, dataset, metric, out_path):

    subset = df[df["dataset"] == dataset]

    if subset.empty:

        return

    fig, ax = plt.subplots()

    for method in subset["method"].unique():

        data = subset[subset["method"] == method].groupby("eps_tot")[metric].agg(["mean", "std"]).reset_index()

        ax.errorbar(data["eps_tot"], data["mean"], yerr=data["std"], marker="o", capsize=3, label=method)

    ax.set_xlabel("epsilon")

    ax.set_ylabel(metric)

    ax.set_title(f"{dataset.upper()} {metric} vs eps")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def plot_collapse(df, dataset, out_path):

    subset = df[df["dataset"] == dataset]

    if subset.empty:

        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for metric, ax in zip(["non_empty_k_final", "max_cluster_ratio_final"], axes):

        for method in subset["method"].unique():

            data = subset[subset["method"] == method].groupby("eps_tot")[metric].agg(["mean", "std"]).reset_index()

            ax.errorbar(data["eps_tot"], data["mean"], yerr=data["std"], marker="o", capsize=3, label=method)

        ax.set_xlabel("epsilon")

        ax.set_ylabel(metric)

        ax.grid(True, alpha=0.3)

    axes[0].legend()

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)





def plot_schedule(df, dataset, figs_dir):

    target = df[(df["dataset"] == dataset) & (df["method"] == "sahdpca")]

    if target.empty:

        return

    hist_path = target.iloc[0]["history_path"]

    if not os.path.exists(hist_path):

        return

    hist = pd.read_csv(hist_path)

    fig, ax = plt.subplots()

    ax.plot(hist["iter"], hist["eps_t"], marker="o")

    ax.set_xlabel("iter")

    ax.set_ylabel("eps_t")

    ax.set_title(f"{dataset.upper()} eps schedule")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_eps_schedule.png"))

    plt.close(fig)





def effect_size(df: pd.DataFrame, dataset: str, out_dir: str):

    subset = df[(df["dataset"] == dataset) & (df["method"].isin(["sahdpca", "sahdpca_wo_feedback"]))]

    if subset.empty:

        return

    metrics = ["f1", "sse_x", "non_empty_k_final", "max_cluster_ratio_final"]

    records = []

    for eps in sorted(subset["eps_tot"].unique()):

        a = subset[(subset["eps_tot"] == eps) & (subset["method"] == "sahdpca")]

        b = subset[(subset["eps_tot"] == eps) & (subset["method"] == "sahdpca_wo_feedback")]

        if a.empty or b.empty:

            continue

        for m in metrics:

            va = a[m].values

            vb = b[m].values

            mean_diff = va.mean() - vb.mean()

            pooled = np.sqrt(((va.size - 1) * va.var(ddof=1) + (vb.size - 1) * vb.var(ddof=1)) / (va.size + vb.size - 2 + 1e-12))

            d = mean_diff / (pooled + 1e-12)

            records.append({"eps_tot": eps, "metric": m, "cohen_d": d, "mean_diff": mean_diff, "n_fb": va.size, "n_nofb": vb.size})

    if records:

        pd.DataFrame(records).to_csv(os.path.join(out_dir, f"{dataset}_effect_size.csv"), index=False)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results", type=str, default="outputs/feedback_v2_check/results.csv")

    parser.add_argument("--out_dir", type=str, default="outputs/summary_tables")

    parser.add_argument("--eps", type=float, nargs="*", default=None)

    args = parser.parse_args()



    ensure_dir(args.out_dir)

    df = pd.read_csv(args.results)

    eps_list = args.eps if args.eps else None



    for dataset in ["har", "gas"]:

        agg = summarize(df, dataset, eps_list, args.out_dir)

        latex_table(agg, dataset, args.out_dir)

        plot_metric(df[df["method"].isin(["sahdpca", "sahdpca_wo_feedback"])], dataset, "f1", os.path.join(args.out_dir, f"{dataset}_f1_vs_eps.png"))

        plot_metric(df[df["method"].isin(["sahdpca", "sahdpca_wo_feedback"])], dataset, "sse_x", os.path.join(args.out_dir, f"{dataset}_sse_vs_eps.png"))

        plot_collapse(df[df["method"].isin(["sahdpca", "sahdpca_wo_feedback"])], dataset, os.path.join(args.out_dir, f"{dataset}_collapse_vs_eps.png"))

        plot_schedule(df, dataset, args.out_dir)

        effect_size(df, dataset, args.out_dir)





if __name__ == "__main__":

    main()

