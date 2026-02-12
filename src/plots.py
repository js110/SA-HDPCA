import os

from typing import Dict



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





def _available_methods(df: pd.DataFrame) -> set[str]:

    if df.empty or "method" not in df.columns:

        return set()

    return set(df["method"].unique())





def _pick_method(available: set[str], candidates: list[str]) -> str | None:

    for m in candidates:

        if m in available:

            return m

    return None





def _lineplot(ax, x, y, yerr, label):

    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)





def _prepare(df: pd.DataFrame, dataset: str, metric: str):

    subset = df[df["dataset"] == dataset]

    if subset.empty:

        return None

    agg = subset.groupby(["method", "eps_tot"])[metric].agg(["mean", "std"]).reset_index()

    return agg





def _plot_eps_curves(df: pd.DataFrame, dataset: str, metric: str, path: str):

    agg = _prepare(df, dataset, metric)

    if agg is None or agg.empty:

        return

    fig, ax = plt.subplots()

    for method in agg["method"].unique():

        data = agg[agg["method"] == method].sort_values("eps_tot")

        _lineplot(ax, data["eps_tot"], data["mean"], data["std"], label=method)

    ax.set_xlabel("epsilon")

    ax.set_ylabel(metric.upper())

    ax.set_title(f"{dataset.upper()} {metric} vs eps")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(path)

    plt.close(fig)





def _plot_dim_curve(df: pd.DataFrame, config: Dict, path: str):

    exp_cfg = config["experiments"]["e2"]

    subset = df[

        (df["dataset"] == exp_cfg["dataset"])

        & (df["eps_tot"] == exp_cfg["eps"])

        & (df["method"].isin(exp_cfg["methods"]))

    ]

    if subset.empty:

        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for metric, ax in zip(["sse_x", "f1"], axes):

        agg = subset.groupby(["method", "d"])[metric].agg(["mean", "std"]).reset_index()

        for method in agg["method"].unique():

            data = agg[agg["method"] == method].sort_values("d")

            _lineplot(ax, data["d"], data["mean"], data["std"], label=method)

        ax.set_xlabel("dimension (d)")

        ax.set_ylabel(metric.upper())

        ax.grid(True, alpha=0.3)

    axes[0].legend()

    fig.tight_layout()

    fig.savefig(path)

    plt.close(fig)





def _plot_ablation(df: pd.DataFrame, config: Dict, path: str):

    e3 = config["experiments"]["e3"]

    subset = df[(df["dataset"].isin(e3["datasets"])) & (df["eps_tot"] == e3["eps"]) & (df["method"].isin(e3["methods"]))]

    if subset.empty:

        return

    fig, ax = plt.subplots(figsize=(8, 4))

    bar_width = 0.2

    datasets = e3["datasets"]

    methods = e3["methods"]

    x = np.arange(len(methods))

    for i, dataset in enumerate(datasets):

        data = subset[subset["dataset"] == dataset].groupby("method")["f1"].mean().reindex(methods)

        ax.bar(x + i * bar_width, data.values, width=bar_width, label=dataset)

    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)

    ax.set_xticklabels(methods, rotation=20)

    ax.set_ylabel("F1")

    ax.set_title("Ablation (F1, eps=1)")

    ax.legend()

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(path)

    plt.close(fig)





def _plot_budget_for(df: pd.DataFrame, dataset: str, figs_dir: str):

    available = _available_methods(df)

    method = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    if method is None:

        return

    target = df[

        (df["dataset"] == dataset) & (df["method"] == method) & (df["eps_tot"] == 1) & (df["seed"] == 0)

    ]

    if target.empty:

        return

    history_path = target.iloc[0]["history_path"]

    if not os.path.exists(history_path):

        return

    hist = pd.read_csv(history_path)

    fig, ax = plt.subplots()

    col = "eps_t" if "eps_t" in hist.columns else "eps"

    ax.plot(hist["iter"], hist[col], marker="o")

    ax.set_xlabel("iter")

    ax.set_ylabel("eps_t")

    ax.set_title(f"{dataset.upper()} eps schedule (feedback)")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_budget_curve.png"))

    plt.close(fig)



    fig, ax = plt.subplots()

    ax.plot(hist["iter"], hist["drift"], marker="o", color="tab:red")

    ax.set_xlabel("iter")

    ax.set_ylabel("drift")

    ax.set_title(f"{dataset.upper()} drift over iters")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_drift_curve.png"))

    plt.close(fig)





def _plot_budget_noise_compare(df: pd.DataFrame, dataset: str, figs_dir: str, eps: float = 1.0):

    available = _available_methods(df)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    no_fb = _pick_method(available, ["sahdpca_wo_feedback"])

    if fb is None or no_fb is None:

        return

    methods = [fb, no_fb]

    labels = ["fb", "no-fb"]

    rows = df[(df["dataset"] == dataset) & (df["eps_tot"] == eps) & (df["seed"] == 0) & (df["method"].isin(methods))]

    if rows.empty or rows["method"].nunique() < 2:

        return

    fig, ax = plt.subplots()

    for method, label in zip(methods, labels):

        row = rows[rows["method"] == method]

        if row.empty:

            continue

        hist_path = row.iloc[0]["history_path"]

        if not os.path.exists(hist_path):

            continue

        hist = pd.read_csv(hist_path)

        if "eps_t" not in hist.columns:

            continue

        ax.plot(hist["iter"], hist["eps_t"], marker="o", label=f"{label} eps_t")

    ax.set_xlabel("iter")

    ax.set_ylabel("eps_t")

    ax.set_title(f"{dataset.upper()} per-iter budget (eps={eps}, seed=0)")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_eps_budget_compare_eps{eps}.png"))

    plt.close(fig)



    fig, ax = plt.subplots()

    for method, label in zip(methods, ["fb", "no-fb"]):

        row = rows[rows["method"] == method]

        if row.empty:

            continue

        hist_path = row.iloc[0]["history_path"]

        if not os.path.exists(hist_path):

            continue

        hist = pd.read_csv(hist_path)

        ax.plot(hist["iter"], hist["noise_scale_counts"], marker="o", label=f"{label} noise_counts")

    ax.set_xlabel("iter")

    ax.set_ylabel("Laplace scale (counts)")

    ax.set_title(f"{dataset.upper()} noise scale (eps={eps}, seed=0)")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_noise_scale_compare_eps{eps}.png"))

    plt.close(fig)





def _plot_budget_cumsum_compare(df: pd.DataFrame, dataset: str, figs_dir: str, eps: float = 1.0):

    available = _available_methods(df)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    no_fb = _pick_method(available, ["sahdpca_wo_feedback"])

    if fb is None or no_fb is None:

        return

    methods = [fb, no_fb]

    labels = ["fb", "no-fb"]

    rows = df[(df["dataset"] == dataset) & (df["eps_tot"] == eps) & (df["seed"] == 0) & (df["method"].isin(methods))]

    if rows.empty or rows["method"].nunique() < 2:

        return

    fig, ax = plt.subplots()

    for method, label in zip(methods, labels):

        row = rows[rows["method"] == method]

        if row.empty:

            continue

        hist_path = row.iloc[0]["history_path"]

        if not os.path.exists(hist_path):

            continue

        hist = pd.read_csv(hist_path)

        if "eps_t" not in hist.columns:

            continue

        cumsum = hist["eps_t"].cumsum()

        ax.plot(hist["iter"], cumsum, marker="o", label=label)

    ax.set_xlabel("iter")

    ax.set_ylabel("cumulative eps")

    ax.set_title(f"{dataset.upper()} cumulative budget (eps={eps}, seed=0)")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_eps_cumsum_compare_eps{eps}.png"))

    plt.close(fig)





def _plot_synth_runtime(df: pd.DataFrame, figs_dir: str):

    subset = df[(df["dataset"] == "synthetic") & (df["eps_tot"] == 1)]

    if subset.empty:

        return

    for group_key, fname in [

        ("n", "synth_runtime_vs_n.png"),

        ("d", "synth_runtime_vs_d.png"),

    ]:

        fig, ax = plt.subplots()

        agg = subset.groupby(["method", group_key])["runtime_ms_total"].mean().reset_index()

        for method in agg["method"].unique():

            data = agg[agg["method"] == method].sort_values(group_key)

            ax.plot(data[group_key], data["runtime_ms_total"], marker="o", label=method)

        ax.set_xlabel(group_key)

        ax.set_ylabel("runtime (ms)")

        ax.grid(True, alpha=0.3)

        ax.legend()

        fig.tight_layout()

        fig.savefig(os.path.join(figs_dir, fname))

        plt.close(fig)





def _plot_robustness(df: pd.DataFrame, config: Dict, path: str):

    e6 = config["experiments"]["e6"]

    subset = df[

        (df["dataset"] == e6["dataset"])

        & (df["eps_tot"] == e6["eps"])

        & (df["method"].isin(e6["methods"]))

        & (df["seed"].isin(e6["seeds"]))

    ]

    if subset.empty:

        return

    fig, ax = plt.subplots()

    data = [subset[subset["method"] == m]["f1"] for m in e6["methods"]]

    ax.boxplot(data, labels=e6["methods"])

    ax.set_ylabel("F1")

    ax.set_title("HAR robustness (eps=0.5)")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(path)

    plt.close(fig)





def _plot_collapse(df: pd.DataFrame, dataset: str, path: str):

    subset = df[df["dataset"] == dataset]

    required_cols = {"non_empty_k_final", "max_cluster_ratio_final"}

    if subset.empty or not required_cols.issubset(subset.columns):

        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    metrics = [

        ("non_empty_k_final", "non-empty clusters"),

        ("max_cluster_ratio_final", "max cluster ratio"),

    ]

    for (metric, ylabel), ax in zip(metrics, axes):

        agg = subset.groupby(["method", "eps_tot"])[metric].agg(["mean", "std"]).reset_index()

        for method in agg["method"].unique():

            data = agg[agg["method"] == method].sort_values("eps_tot")

            _lineplot(ax, data["eps_tot"], data["mean"], data["std"], label=method)

        ax.set_xlabel("epsilon")

        ax.set_ylabel(ylabel)

        ax.set_title(f"{dataset.upper()} {metric} vs eps")

        ax.grid(True, alpha=0.3)

    axes[0].legend()

    fig.tight_layout()

    fig.savefig(path)

    plt.close(fig)





def _plot_feedback_effect(df: pd.DataFrame, dataset: str, figs_dir: str):

    available = _available_methods(df)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    no_fb = _pick_method(available, ["sahdpca_wo_feedback"])

    if fb is None or no_fb is None:

        return

    methods = [fb, no_fb]

    labels = ["fb", "no-fb"]

    subset = df[(df["dataset"] == dataset) & (df["eps_tot"] == 1) & (df["method"].isin(methods))]

    if subset.empty:

        return

    metrics = [

        ("f1", "Macro-F1"),

        ("non_empty_k_final", "Non-empty K"),

        ("max_cluster_ratio_final", "Max cluster ratio"),

    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(methods))

    bar_width = 0.6

    for ax, (metric, title) in zip(axes, metrics):

        agg = subset.groupby("method")[metric].agg(["mean", "std"]).reindex(methods)

        ax.bar(x, agg["mean"].values, yerr=agg["std"].values, width=bar_width, capsize=4, color=["tab:blue", "tab:orange"])

        ax.set_xticks(x)

        ax.set_xticklabels(labels)

        ax.set_ylabel(title)

        ax.set_title(f"{title} (eps=1)")

        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_feedback_effect_eps1.png"))

    plt.close(fig)





def _plot_per_seed_scatter(df: pd.DataFrame, dataset: str, figs_dir: str, eps_target: float = 0.5):

    subset = df[(df["dataset"] == dataset) & (df["eps_tot"] == eps_target)]

    if subset.empty:

        return

    available = _available_methods(subset)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    fb_label = "fb_v3" if fb == "sahdpca_fb_v3" else "fb"

    fb_v2 = "sahdpca_fb_v2" if "sahdpca_fb_v2" in available else None

    no_fb = "sahdpca_wo_feedback" if "sahdpca_wo_feedback" in available else None

    strong = "sahdpca_strong" if "sahdpca_strong" in available else None

    methods = []

    if fb is not None:

        methods.append((fb, fb_label))

    if fb_v2 is not None:

        methods.append((fb_v2, "fb_v2"))

    if no_fb is not None:

        methods.append((no_fb, "no-fb"))

    if strong is not None:

        methods.append((strong, "strong"))

    if not methods:

        return

    metrics = [

        ("f1", "Macro-F1"),

        ("non_empty_k_final", "Non-empty K"),

        ("max_cluster_ratio_final", "Max cluster ratio"),

    ]

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(methods))

    jitter_scale = 0.08

    for ax, (metric, title) in zip(axes, metrics):

        for i, (m, label) in enumerate(methods):

            vals = subset[subset["method"] == m][metric].values

            if vals.size == 0:

                continue

            jitter = (rng.random(len(vals)) - 0.5) * jitter_scale

            ax.scatter(np.full_like(vals, x[i]) + jitter, vals, alpha=0.6, s=25, label=label if metric == "f1" else None)

            ax.hlines(np.mean(vals), x[i] - 0.15, x[i] + 0.15, colors="k", linestyles="--", linewidth=1)

        ax.set_xticks(x)

        ax.set_xticklabels([label for _, label in methods])

        ax.set_title(f"{title} (eps={eps_target})")

        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend()

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_per_seed_eps{eps_target}.png"))

    plt.close(fig)





def _plot_schedule_contrast(df: pd.DataFrame, dataset: str, figs_dir: str):

    subset = df[(df["dataset"] == dataset) & (df["eps_tot"] == 1)]

    if subset.empty:

        return

    available = _available_methods(subset)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    fb_label = "fb_v3" if fb == "sahdpca_fb_v3" else "fb"

    fb_v2 = "sahdpca_fb_v2" if "sahdpca_fb_v2" in available else None

    no_fb = "sahdpca_wo_feedback" if "sahdpca_wo_feedback" in available else None

    strong = "sahdpca_strong" if "sahdpca_strong" in available else None

    methods = []

    if no_fb is not None:

        methods.append((no_fb, "no-fb"))

    if fb is not None:

        methods.append((fb, fb_label))

    if fb_v2 is not None:

        methods.append((fb_v2, "fb_v2"))

    if strong is not None:

        methods.append((strong, "strong"))

    if len(methods) < 2:

        return

    metrics = [

        ("f1", "Macro-F1"),

        ("non_empty_k_final", "Non-empty K"),

        ("max_cluster_ratio_final", "Max cluster ratio"),

    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(methods))

    bar_width = 0.6

    for ax, (metric, title) in zip(axes, metrics):

        method_names = [m for m, _ in methods]

        agg = subset.groupby("method")[metric].agg(["mean", "std"]).reindex(method_names)

        ax.bar(

            x,

            agg["mean"].values,

            yerr=agg["std"].values,

            width=bar_width,

            capsize=4,

            color=["tab:orange", "tab:blue", "tab:purple", "tab:green"][: len(methods)],

        )

        ax.set_xticks(x)

        ax.set_xticklabels([label for _, label in methods])

        ax.set_ylabel(title)

        ax.set_title(f"{title} (eps=1)")

        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_schedule_contrast_eps1.png"))

    plt.close(fig)





def _plot_schedule_compare(df: pd.DataFrame, dataset: str, figs_dir: str):

    target = df[(df["dataset"] == dataset) & (df["eps_tot"] == 1) & (df["seed"] == 0)]

    if target.empty:

        return

    available = _available_methods(target)

    fb = _pick_method(available, ["sahdpca_fb_v3", "sahdpca"])

    fb_label = "fb_v3" if fb == "sahdpca_fb_v3" else "fb"

    fb_v2 = "sahdpca_fb_v2" if "sahdpca_fb_v2" in available else None

    no_fb = "sahdpca_wo_feedback" if "sahdpca_wo_feedback" in available else None

    strong = "sahdpca_strong" if "sahdpca_strong" in available else None

    methods = []

    if no_fb is not None:

        methods.append((no_fb, "no-fb"))

    if fb is not None:

        methods.append((fb, fb_label))

    if fb_v2 is not None:

        methods.append((fb_v2, "fb_v2"))

    if strong is not None:

        methods.append((strong, "strong"))

    if len(methods) < 2:

        return

    fig, ax = plt.subplots()

    for method, label in methods:

        row = target[target["method"] == method]

        if row.empty:

            continue

        history_path = row.iloc[0]["history_path"]

        if not os.path.exists(history_path):

            continue

        hist = pd.read_csv(history_path)

        if "eps_t" not in hist.columns:

            continue

        ax.plot(hist["iter"], hist["eps_t"], marker="o", label=label)

    ax.set_xlabel("iter")

    ax.set_ylabel("eps_t")

    ax.set_title(f"{dataset.upper()} eps schedule compare (eps=1, seed=0)")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(os.path.join(figs_dir, f"{dataset}_eps_schedule_compare_eps1.png"))

    plt.close(fig)





def generate_figures(df: pd.DataFrame, figs_dir: str, config: Dict):

    if df.empty:

        return

    _plot_eps_curves(df, "har", "sse_x", os.path.join(figs_dir, "har_sse_vs_eps.png"))

    _plot_eps_curves(df, "har", "f1", os.path.join(figs_dir, "har_f1_vs_eps.png"))

    _plot_eps_curves(df, "gas", "sse_x", os.path.join(figs_dir, "gas_sse_vs_eps.png"))

    _plot_eps_curves(df, "gas", "f1", os.path.join(figs_dir, "gas_f1_vs_eps.png"))

    exp_cfg = config.get("experiments", {})

    if "e2" in exp_cfg:

        _plot_dim_curve(df, config, os.path.join(figs_dir, "har_metric_vs_dim.png"))

    if "e3" in exp_cfg:

        _plot_ablation(df, config, os.path.join(figs_dir, "ablation_f1.png"))

    _plot_budget_for(df, "har", figs_dir)

    _plot_budget_for(df, "gas", figs_dir)

    _plot_budget_noise_compare(df, "har", figs_dir, eps=1.0)

    _plot_budget_noise_compare(df, "gas", figs_dir, eps=1.0)

    _plot_synth_runtime(df, figs_dir)

    if "e6" in exp_cfg:

        _plot_robustness(df, config, os.path.join(figs_dir, "har_robustness_boxplot.png"))

    _plot_collapse(df, "har", os.path.join(figs_dir, "har_collapse_vs_eps.png"))

    _plot_collapse(df, "gas", os.path.join(figs_dir, "gas_collapse_vs_eps.png"))

    _plot_feedback_effect(df, "har", figs_dir)

    _plot_feedback_effect(df, "gas", figs_dir)

    _plot_per_seed_scatter(df, "har", figs_dir, eps_target=0.5)

    _plot_per_seed_scatter(df, "gas", figs_dir, eps_target=0.5)

    _plot_schedule_contrast(df, "har", figs_dir)

    _plot_schedule_contrast(df, "gas", figs_dir)

    _plot_schedule_compare(df, "har", figs_dir)

    _plot_schedule_compare(df, "gas", figs_dir)

