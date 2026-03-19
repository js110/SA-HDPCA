#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_tag(summary_path: Path) -> str:
    if summary_path.parent.name == "summary":
        return summary_path.parent.parent.name
    return summary_path.stem


def format_value(mean: float, std: float, metric: str, latex: bool) -> str:
    if mean is None or std is None or not np.isfinite(mean) or not np.isfinite(std):
        return ""
    if metric == "sse_x":
        if latex:
            return f"{mean:.3e}\\\\pm{std:.3e}"
        return f"{mean:.3e}+/-{std:.3e}"
    if latex:
        return f"{mean:.3f}\\\\pm{std:.3f}"
    return f"{mean:.3f}+/-{std:.3f}"


def build_formatted(subset: pd.DataFrame, latex: bool) -> pd.DataFrame:
    rows = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "eps_tot": row["eps_tot"],
                "f1": format_value(row["f1_mean"], row["f1_std"], "f1", latex),
                "sse_x": format_value(row["sse_x_mean"], row["sse_x_std"], "sse_x", latex),
                "collapse_rate": format_value(
                    row["collapse_rate_mean"], row["collapse_rate_std"], "collapse_rate", latex
                ),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    cols = [str(c) for c in df.columns]
    header = " & ".join(cols) + r" \\"
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", header]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric(subset: pd.DataFrame, metric: str, label: str, out_path: Path) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in subset.columns or std_col not in subset.columns:
        return
    fig, ax = plt.subplots()
    for method in sorted(subset["method"].unique()):
        data = subset[subset["method"] == method].sort_values("eps_tot")
        ax.errorbar(
            data["eps_tot"],
            data[mean_col],
            yerr=data[std_col],
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set_xlabel("epsilon")
    ax.set_ylabel(label)
    ax.set_title(f"{subset.iloc[0]['dataset'].upper()} {label} vs eps")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make ablation tables and figures from summary CSV.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to ablation_summary.csv (from summarize_ablation.py).",
    )
    parser.add_argument("--out_tables", type=str, default="tables", help="Output tables directory.")
    parser.add_argument("--out_figs", type=str, default="figs", help="Output figures directory.")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename prefix.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        return

    tag = args.tag or infer_tag(summary_path)
    tables_dir = Path(args.out_tables)
    figs_dir = Path(args.out_figs)
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    raw_cols = [
        "dataset",
        "method",
        "eps_tot",
        "f1_mean",
        "f1_std",
        "sse_x_mean",
        "sse_x_std",
        "collapse_rate_mean",
        "collapse_rate_std",
    ]

    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        raw = subset[raw_cols]
        raw.to_csv(tables_dir / f"{tag}_{dataset}_raw.csv", index=False)

        formatted_csv = build_formatted(subset, latex=False)
        formatted_csv.to_csv(tables_dir / f"{tag}_{dataset}.csv", index=False)

        formatted_tex = build_formatted(subset, latex=True)
        write_latex_table(formatted_tex, tables_dir / f"{tag}_{dataset}.tex")

        plot_metric(subset, "f1", "Macro-F1", figs_dir / f"{tag}_{dataset}_f1_vs_eps.png")
        plot_metric(subset, "sse_x", "SSE_x", figs_dir / f"{tag}_{dataset}_sse_x_vs_eps.png")
        plot_metric(subset, "collapse_rate", "Collapse rate", figs_dir / f"{tag}_{dataset}_collapse_vs_eps.png")

    print(f"[OK] tables saved to {tables_dir}")
    print(f"[OK] figures saved to {figs_dir}")


if __name__ == "__main__":
    main()
