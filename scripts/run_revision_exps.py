import argparse
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import methods
from src.utils import ensure_dir


def _run_sweep(
    dataset: str,
    method: str,
    eps: float,
    seeds: List[int],
    config: Dict,
    out_dir: str,
    tag: str,
    dataset_overrides: Dict | None = None,
    budget_overrides: Dict | None = None,
    proxy_eps_ratio: float | None = None,
    n_override: int | None = None,
    d_override: int | None = None,
    k_override: int | None = None,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        res = methods.run_method(
            dataset=dataset,
            method=method,
            eps_tot=eps,
            seed=seed,
            config=config,
            out_dir=out_dir,
            n_override=n_override,
            d_override=d_override,
            k_override=k_override,
            proxy_eps_ratio=proxy_eps_ratio,
            dataset_overrides=dataset_overrides,
            budget_overrides=budget_overrides,
        )
        res["tag"] = tag
        rows.append(res)
    return pd.DataFrame(rows)


def _errorbar_plot(df: pd.DataFrame, x_col: str, metric: str, path: str, title: str, xlabel: str):
    fig, ax = plt.subplots()
    agg = df.groupby(x_col)[metric].agg(["mean", "std"]).reset_index()
    ax.errorbar(agg[x_col], agg["mean"], yerr=agg["std"], marker="o", capsize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _method_plot(df: pd.DataFrame, x_col: str, metric: str, path: str, title: str, xlabel: str):
    fig, ax = plt.subplots()
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        agg = sub.groupby(x_col)[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg[x_col], agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=method)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run revision experiments (E7-E9).")
    parser.add_argument("--config", type=str, default="configs/revision.yaml")
    parser.add_argument("--out", type=str, default="outputs/revision")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Run only a subset: sensitivity|scalability|imbalance|streaming|all",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Limit sensitivity runs to a single dataset (har|gas).",
    )
    parser.add_argument(
        "--param",
        type=str,
        default="all",
        help="Limit sensitivity param sweep (pus_top_m|proxy_eps_ratio|beta|gamma).",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        help="Limit scalability sweep (n|d|k).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = args.out
    ensure_dir(out_dir)
    figs_dir = os.path.join(out_dir, "figures")
    ensure_dir(figs_dir)

    rev = config["revision"]

    if args.only in {"all", "sensitivity"}:
        # E7: sensitivity
        sens_cfg = rev["sensitivity"]
        sens_rows = []
        datasets = sens_cfg["datasets"]
        if args.dataset in {"har", "gas"}:
            datasets = [args.dataset]
        params = ["pus_top_m", "proxy_eps_ratio", "beta", "gamma"]
        if args.param in {"pus_top_m", "proxy_eps_ratio", "beta", "gamma"}:
            params = [args.param]
        for dataset in datasets:
            for m, r in zip(sens_cfg["pus_top_m_list"], sens_cfg["pca_r_list"]):
                if "pus_top_m" in params:
                    df = _run_sweep(
                        dataset=dataset,
                        method=sens_cfg["method"],
                        eps=sens_cfg["eps"],
                        seeds=sens_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="pus_top_m",
                        dataset_overrides={"pus_top_m": int(m), "pca_r": int(r)},
                    )
                    df["param"] = "pus_top_m"
                    df["value"] = int(m)
                    sens_rows.append(df)
            for ratio in sens_cfg["proxy_eps_ratios"]:
                if "proxy_eps_ratio" in params:
                    df = _run_sweep(
                        dataset=dataset,
                        method=sens_cfg["method"],
                        eps=sens_cfg["eps"],
                        seeds=sens_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="proxy_eps_ratio",
                        proxy_eps_ratio=float(ratio),
                    )
                    df["param"] = "proxy_eps_ratio"
                    df["value"] = float(ratio)
                    sens_rows.append(df)
            for beta in sens_cfg["budget_beta_list"]:
                if "beta" in params:
                    df = _run_sweep(
                        dataset=dataset,
                        method=sens_cfg["method"],
                        eps=sens_cfg["eps"],
                        seeds=sens_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="beta",
                        budget_overrides={"beta": float(beta)},
                    )
                    df["param"] = "beta"
                    df["value"] = float(beta)
                    sens_rows.append(df)
            for gamma in sens_cfg["budget_gamma_list"]:
                if "gamma" in params:
                    df = _run_sweep(
                        dataset=dataset,
                        method=sens_cfg["method"],
                        eps=sens_cfg["eps"],
                        seeds=sens_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="gamma",
                        budget_overrides={"gamma": float(gamma)},
                    )
                    df["param"] = "gamma"
                    df["value"] = float(gamma)
                    sens_rows.append(df)

        if sens_rows:
            sens_df = pd.concat(sens_rows, ignore_index=True)
            sens_path = os.path.join(out_dir, "sensitivity.csv")
            sens_df.to_csv(sens_path, index=False)
        else:
            sens_df = pd.DataFrame()
            sens_path = os.path.join(out_dir, "sensitivity.csv")

        for dataset in datasets:
            for param in params:
                sub = sens_df[(sens_df["dataset"] == dataset) & (sens_df["param"] == param)]
                if sub.empty:
                    continue
                _errorbar_plot(
                    sub,
                    "value",
                    "f1",
                    os.path.join(figs_dir, f"sens_{dataset}_{param}_f1.png"),
                    f"{dataset.upper()} F1 vs {param}",
                    param,
                )
                _errorbar_plot(
                    sub,
                    "value",
                    "sse_x",
                    os.path.join(figs_dir, f"sens_{dataset}_{param}_sse.png"),
                    f"{dataset.upper()} SSE_x vs {param}",
                    param,
                )

    if args.only in {"all", "scalability"}:
        # E8: scalability (n/d/k sweeps)
        sc_cfg = rev["scalability"]
        sc_rows = []
        n0 = sc_cfg["n_list"][0]
        d0 = sc_cfg["d_list"][0]
        k0 = sc_cfg["k_list"][0]
        sweeps = ["n", "d", "k"]
        if args.sweep in {"n", "d", "k"}:
            sweeps = [args.sweep]
        if "n" in sweeps:
            for n in sc_cfg["n_list"]:
                for method in sc_cfg["methods"]:
                    df = _run_sweep(
                        dataset=sc_cfg["dataset"],
                        method=method,
                        eps=sc_cfg["eps"],
                        seeds=sc_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="n_sweep",
                        n_override=int(n),
                        d_override=int(d0),
                        k_override=int(k0),
                    )
                    df["param"] = "n"
                    df["value"] = int(n)
                    sc_rows.append(df)
        if "d" in sweeps:
            for d in sc_cfg["d_list"]:
                for method in sc_cfg["methods"]:
                    df = _run_sweep(
                        dataset=sc_cfg["dataset"],
                        method=method,
                        eps=sc_cfg["eps"],
                        seeds=sc_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="d_sweep",
                        n_override=int(n0),
                        d_override=int(d),
                        k_override=int(k0),
                    )
                    df["param"] = "d"
                    df["value"] = int(d)
                    sc_rows.append(df)
        if "k" in sweeps:
            for k in sc_cfg["k_list"]:
                for method in sc_cfg["methods"]:
                    df = _run_sweep(
                        dataset=sc_cfg["dataset"],
                        method=method,
                        eps=sc_cfg["eps"],
                        seeds=sc_cfg["seeds"],
                        config=config,
                        out_dir=out_dir,
                        tag="k_sweep",
                        n_override=int(n0),
                        d_override=int(d0),
                        k_override=int(k),
                    )
                    df["param"] = "k"
                    df["value"] = int(k)
                    sc_rows.append(df)
        sc_df = pd.concat(sc_rows, ignore_index=True)
        sc_path = os.path.join(out_dir, "scalability.csv")
        sc_df.to_csv(sc_path, index=False)

        for param, xlabel in [("n", "n"), ("d", "dimension d"), ("k", "k")]:
            sub = sc_df[sc_df["param"] == param]
            if sub.empty:
                continue
            _method_plot(
                sub,
                "value",
                "runtime_ms_total",
                os.path.join(figs_dir, f"scalability_runtime_{param}.png"),
                f"Runtime vs {param}",
                xlabel,
            )

    if args.only in {"all", "imbalance"}:
        # E8 (imbalanced)
        im_cfg = rev["imbalance"]
        im_rows = []
        for method in im_cfg["methods"]:
            df = _run_sweep(
                dataset=im_cfg["dataset"],
                method=method,
                eps=im_cfg["eps"],
                seeds=im_cfg["seeds"],
                config=config,
                out_dir=out_dir,
                tag="imbalance",
                n_override=int(im_cfg["n"]),
                d_override=int(im_cfg["d"]),
                k_override=int(im_cfg["k"]),
                dataset_overrides={"cluster_weights": im_cfg["weights"]},
            )
            df["param"] = "imbalance"
            df["value"] = 1
            im_rows.append(df)
        im_df = pd.concat(im_rows, ignore_index=True)
        im_path = os.path.join(out_dir, "imbalance.csv")
        im_df.to_csv(im_path, index=False)

        _method_plot(
            im_df,
            "method",
            "f1",
            os.path.join(figs_dir, "imbalance_f1.png"),
            "Imbalanced synthetic (Macro-F1)",
            "method",
        )

    if args.only in {"all", "streaming"}:
        # E9: streaming
        st_cfg = rev["streaming"]
        st_rows = []
        for method in st_cfg["methods"]:
            df = _run_sweep(
                dataset=st_cfg["dataset"],
                method=method,
                eps=st_cfg["eps"],
                seeds=st_cfg["seeds"],
                config=config,
                out_dir=out_dir,
                tag="streaming",
            )
            df["param"] = "streaming"
            df["value"] = 1
            st_rows.append(df)
        st_df = pd.concat(st_rows, ignore_index=True)
        st_path = os.path.join(out_dir, "streaming.csv")
        st_df.to_csv(st_path, index=False)

        _method_plot(
            st_df,
            "method",
            "f1",
            os.path.join(figs_dir, "streaming_f1.png"),
            "Streaming synthetic (Macro-F1)",
            "method",
        )

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Revision experiments (E7-E9)\n")
        if args.only in {"all", "sensitivity"}:
            f.write(f"- sensitivity: {os.path.join(out_dir, 'sensitivity.csv')}\n")
        if args.only in {"all", "scalability"}:
            f.write(f"- scalability: {os.path.join(out_dir, 'scalability.csv')}\n")
        if args.only in {"all", "imbalance"}:
            f.write(f"- imbalance: {os.path.join(out_dir, 'imbalance.csv')}\n")
        if args.only in {"all", "streaming"}:
            f.write(f"- streaming: {os.path.join(out_dir, 'streaming.csv')}\n")
        f.write(f"- figures: {figs_dir}\n")


if __name__ == "__main__":
    main()
