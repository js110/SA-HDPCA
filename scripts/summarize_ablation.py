                      

import argparse

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)





def load_history(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path)





def collapse_rate_from_history(hist: pd.DataFrame, k: int, min_ratio: float) -> float:

    if hist.empty or "non_empty_k" not in hist.columns:

        return float("nan")

    if "min_cluster_ratio" in hist.columns:

        flags = (hist["non_empty_k"] < k) | (hist["min_cluster_ratio"] < min_ratio)

    else:

        flags = hist["non_empty_k"] < k

    return float(flags.mean()) if len(flags) else float("nan")





def build_history_long(df: pd.DataFrame, min_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []

    collapse_rates = []

    for _, row in df.iterrows():

        hist_val = row.get("history_path", "")

        hist_path = Path(hist_val) if isinstance(hist_val, str) and hist_val else None

        if hist_path is None or not hist_path.is_file():

            hist_path = None

            if row.get("dataset") and row.get("method") is not None:

                hist_guess = Path(row["out_dir"]) / "history" / f"{row['dataset']}_{row['method']}_eps{row['eps_tot']}_seed{row['seed']}.csv"

                if hist_guess.is_file():

                    hist_path = hist_guess

        if hist_path is None:

            collapse_rates.append(float("nan"))

            continue

        hist = load_history(hist_path)

        if hist.empty:

            collapse_rates.append(float("nan"))

            continue

        k_val = int(row.get("k", 0))

        collapse_rates.append(collapse_rate_from_history(hist, k_val, min_ratio))

        hist = hist.copy()

        hist["dataset"] = row["dataset"]

        hist["method"] = row["method"]

        hist["eps_tot"] = row["eps_tot"]

        hist["seed"] = row["seed"]

        rows.append(hist)

    history_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df = df.copy()

    df["collapse_rate"] = collapse_rates

    return history_long, df





def summarize_results(df: pd.DataFrame, out_dir: Path) -> None:

    metrics = ["f1", "sse_x", "collapse_rate"]

    present = [m for m in metrics if m in df.columns]

    if not present:

        return

    agg = df.groupby(["dataset", "method", "eps_tot"])[present].agg(["mean", "std", "count"])

    agg.columns = ["_".join(col).strip() for col in agg.columns.to_flat_index()]

    agg.reset_index().to_csv(out_dir / "ablation_summary.csv", index=False)





def summarize_drift(history_long: pd.DataFrame, out_dir: Path) -> pd.DataFrame:

    if history_long.empty:

        return pd.DataFrame()

    drift_col = "drift_raw" if "drift_raw" in history_long.columns else "drift"

    if drift_col not in history_long.columns:

        return pd.DataFrame()

    agg = (

        history_long.groupby(["dataset", "method", "eps_tot", "iter"])[drift_col]

        .agg(["mean", "std", "count"])

        .reset_index()

    )

    agg = agg.rename(columns={"mean": "drift_mean", "std": "drift_std", "count": "n"})

    agg.to_csv(out_dir / "drift_curve.csv", index=False)

    return agg





def plot_drift_curves(agg: pd.DataFrame, out_dir: Path) -> None:

    if agg.empty:

        return

    for dataset in sorted(agg["dataset"].unique()):

        data_ds = agg[agg["dataset"] == dataset]

        for eps in sorted(data_ds["eps_tot"].unique()):

            data_eps = data_ds[data_ds["eps_tot"] == eps]

            fig, ax = plt.subplots()

            for method in sorted(data_eps["method"].unique()):

                data_m = data_eps[data_eps["method"] == method].sort_values("iter")

                ax.plot(data_m["iter"], data_m["drift_mean"], marker="o", label=method)

                if "drift_std" in data_m.columns:

                    ax.fill_between(

                        data_m["iter"],

                        data_m["drift_mean"] - data_m["drift_std"],

                        data_m["drift_mean"] + data_m["drift_std"],

                        alpha=0.15,

                    )

            ax.set_xlabel("iter")

            ax.set_ylabel("drift")

            ax.set_title(f"{dataset.upper()} drift (eps={eps})")

            ax.grid(True, alpha=0.3)

            ax.legend()

            fig.tight_layout()

            fig.savefig(out_dir / f"drift_{dataset}_eps{eps}.png")

            plt.close(fig)





def main() -> None:

    parser = argparse.ArgumentParser(description="Summarize ablation outputs.")

    parser.add_argument("--out", type=str, required=True, help="Output directory from src.runner.")

    parser.add_argument("--min_ratio", type=float, default=0.01, help="Near-empty threshold for collapse rate.")

    args = parser.parse_args()



    out_dir = Path(args.out)

    results_path = out_dir / "results.csv"

    if not results_path.exists():

        raise FileNotFoundError(f"Missing results.csv at {results_path}")

    df = pd.read_csv(results_path)

    if "out_dir" not in df.columns:

        df["out_dir"] = str(out_dir)



    summary_dir = out_dir / "summary"

    plots_dir = out_dir / "plots"

    ensure_dir(summary_dir)

    ensure_dir(plots_dir)



    history_long, df_with_collapse = build_history_long(df, args.min_ratio)

    if not history_long.empty:

        history_long.to_csv(summary_dir / "history.csv", index=False)



    summarize_results(df_with_collapse, summary_dir)

    drift_agg = summarize_drift(history_long, summary_dir)

    plot_drift_curves(drift_agg, plots_dir)



    print(f"[OK] summary saved to {summary_dir}")

    print(f"[OK] plots saved to {plots_dir}")





if __name__ == "__main__":

    main()

