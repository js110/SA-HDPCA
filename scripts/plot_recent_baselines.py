                      

import argparse

import sys

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:

    sys.path.append(str(ROOT))





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Plot recent baseline comparison figures.")

    parser.add_argument(

        "--input",

        type=str,

        default="outputs/compare_recent_2024_2025/raw_results_new.csv",

        help="Path to raw_results_new.csv",

    )

    parser.add_argument(

        "--outdir",

        type=str,

        default="outputs/compare_recent_2024_2025/figures",

        help="Output dir for figures",

    )

    return parser.parse_args()





def set_style() -> None:

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["axes.titlesize"] = 12

    plt.rcParams["axes.labelsize"] = 11

    plt.rcParams["legend.fontsize"] = 9





def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)





def agg(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:

    sub = df[df["dataset"] == dataset]

    if sub.empty:

        return pd.DataFrame()

    return sub.groupby(["method", "eps"])[metric].agg(["mean", "std"]).reset_index()





def line_with_markers(

    ax,

    data: pd.DataFrame,

    metric: str,

    methods: list[str],

    title: str,

    label_map: dict[str, str],

) -> None:

    eps_vals = sorted(data["eps"].unique().tolist())

    if not eps_vals:

        return

    markers = ["o", "s", "^", "D", "X", "P", "v"]

    linestyles = ["-", "--", "-.", ":", (0, (5, 2))]



    for idx, method in enumerate(methods):

        dm = data[data["method"] == method].sort_values("eps")

        if dm.empty:

            continue

        marker = markers[idx % len(markers)]

        linestyle = linestyles[idx % len(linestyles)]

        label = label_map.get(method, method)

        ax.plot(

            dm["eps"],

            dm["mean"],

            marker=marker,

            linestyle=linestyle,

            linewidth=1.8,

            markersize=5.5,

            markerfacecolor="white",

            markeredgewidth=1.0,

            label=label,

        )



    ax.set_xlabel("eps")

    ax.set_ylabel(metric)

    ax.set_title(title)

    ax.set_xticks(eps_vals)

    ax.grid(True, alpha=0.3)





def apply_tight_ylim(ax, data: pd.DataFrame, metric: str) -> None:

    if data.empty:

        return

    lower = (data["mean"] - data["std"]).min()

    upper = (data["mean"] + data["std"]).max()

    if not np.isfinite(lower) or not np.isfinite(upper):

        return

    span = max(upper - lower, 1e-12)

    pad = 0.08 * span

    y_min = lower - pad

    y_max = upper + pad

    if metric in {"macro_f1", "sse"}:

        y_min = max(0.0, y_min)

    if metric == "macro_f1":

        y_max = min(1.0, y_max)

    ax.set_ylim(y_min, y_max)





def save_fig(fig, out_base: Path, rect: tuple[float, float, float, float] | None = None) -> None:

    if rect is None:

        fig.tight_layout()

    else:

        fig.tight_layout(rect=rect)

    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")

    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")

    plt.close(fig)





def plot_metric_dual(

    df: pd.DataFrame,

    datasets: list[str],

    metric: str,

    fname: str,

    methods: list[str],

    out_dir: Path,

    label_map: dict[str, str],

) -> None:

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.2), sharey=False)

    if len(datasets) == 1:

        axes = [axes]

    for idx, (ax, ds) in enumerate(zip(axes, datasets)):

        data = agg(df, ds, metric)

        if data.empty:

            continue

        panel = chr(ord("a") + idx)

        line_with_markers(ax, data, metric, methods, f"({panel}) {ds}", label_map)

        apply_tight_ylim(ax, data, metric)

        if metric == "sse":

            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:

        fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=True, bbox_to_anchor=(0.5, 1.05))

        save_fig(fig, out_dir / fname, rect=(0, 0, 1, 0.93))

    else:

        save_fig(fig, out_dir / fname)





def mean_std_fmt(values: pd.Series) -> str:

    return f"{values.mean():.3f}+/-{values.std():.3f}"





def make_table(df: pd.DataFrame, out_path: Path, methods: list[str]) -> None:

    rows = []

    eps_points = sorted(df["eps"].unique().tolist())

    for dataset in sorted(df["dataset"].unique()):

        sub = df[df["dataset"] == dataset]

        for method in methods:

            for eps in eps_points:

                filt = sub[(sub["method"] == method) & (sub["eps"] == eps)]

                if filt.empty:

                    continue

                rows.append(

                    {

                        "dataset": dataset,

                        "method": method,

                        "eps": eps,

                        "macro_f1": mean_std_fmt(filt["macro_f1"]),

                        "sse": mean_std_fmt(filt["sse"]),

                        "runtime": mean_std_fmt(filt["runtime"]),

                    }

                )

    pd.DataFrame(rows).to_csv(out_path, index=False)





def main() -> None:

    args = parse_args()

    set_style()



    out_dir = Path(args.outdir)

    ensure_dir(out_dir)



    df = pd.read_csv(args.input)

    methods = ["SA-HDPCA(fb)", "PCA-DP", "GAPBAS", "DBDP", "DPDP"]

    label_map = {

        "SA-HDPCA(fb)": "Ours",

        "GAPBAS": "GAPBAS [28]",

        "DBDP": "DBDP [30]",

        "DPDP": "Dynamical processing [29]",

    }



    plot_metric_dual(

        df,

        ["HAR", "GAS"],

        "macro_f1",

        "Fig10_recent_macroF1_bar_eps0.8_1.0",

        methods,

        out_dir,

        label_map,

    )

    plot_metric_dual(

        df,

        ["HAR", "GAS"],

        "sse",

        "Fig11_recent_SSE_bar_eps0.8_1.0",

        methods,

        out_dir,

        label_map,

    )



    table_path = out_dir.parent / "Table5_recent_keypoints.csv"

    make_table(df, table_path, methods)



    print(f"[OK] plots saved to {out_dir}")

    print(f"[OK] table saved to {table_path}")





if __name__ == "__main__":

    main()

