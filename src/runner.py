import argparse
import os
import subprocess
from typing import Any, Dict, List

import joblib
import pandas as pd
import yaml

from . import methods
from .plots import generate_figures
from .utils import ensure_dir, set_thread_env


def _git_rev_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        rev = out.strip()
        return rev if rev else "unknown"
    except Exception:
        return "unknown"


def _build_tasks(config: Dict[str, Any], exp: str | None) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    datasets_cfg = config["datasets"]

    def _proxy_ratios_for(exp_cfg: Dict[str, Any] | None = None) -> List[float | None]:
        ratios = exp_cfg.get("proxy_eps_ratios") if exp_cfg else None
        if ratios is None:
            ratios = config.get("proxy_eps_ratios")
        if ratios is None:
            ratio = config.get("proxy_eps_ratio")
            return [ratio] if ratio is not None else [None]
        return [float(r) for r in ratios]

    def add_task(dataset: str, method: str, eps: float, seed: int, proxy_ratios: List[float | None] | None = None, **kwargs):
        ratios = proxy_ratios if proxy_ratios is not None else [None]
        for ratio in ratios:
            tasks.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "eps": eps,
                    "seed": seed,
                    "proxy_eps_ratio": ratio,
                    **kwargs,
                }
            )

    if exp in (None, "e1", "all"):
        e1 = config["experiments"]["e1"]
        proxy_ratios = _proxy_ratios_for(e1)
        for dataset in e1["datasets"]:
            d_sub_default = datasets_cfg[dataset].get("d_sub_list", [None])[0]
            for eps in e1["eps_list"]:
                for seed in e1["seeds"]:
                    for method in e1["methods"]:
                        add_task(dataset, method, eps, seed, proxy_ratios=proxy_ratios, d_sub=d_sub_default)

    if exp in (None, "ea", "all") and "ea" in config["experiments"]:
        ea = config["experiments"]["ea"]
        proxy_ratios = _proxy_ratios_for(ea)
        for dataset in ea["datasets"]:
            for eps in ea["eps_list"]:
                for seed in ea["seeds"]:
                    for method in ea["methods"]:
                        add_task(dataset, method, eps, seed, proxy_ratios=proxy_ratios)

    if exp in (None, "eb", "all") and "eb" in config["experiments"]:
        eb = config["experiments"]["eb"]
        proxy_ratios = _proxy_ratios_for(eb)
        for dataset in eb["datasets"]:
            for eps in eb["eps_list"]:
                for seed in eb["seeds"]:
                    for method in eb["methods"]:
                        add_task(dataset, method, eps, seed, proxy_ratios=proxy_ratios)

    if exp in (None, "ec", "all") and "ec" in config["experiments"]:
        ec = config["experiments"]["ec"]
        proxy_ratios = _proxy_ratios_for(ec)
        for dataset in ec["datasets"]:
            for eps in ec["eps_list"]:
                for seed in ec["seeds"]:
                    for method in ec["methods"]:
                        add_task(dataset, method, eps, seed, proxy_ratios=proxy_ratios)

    if exp in (None, "e2", "all"):
        e2 = config["experiments"]["e2"]
        proxy_ratios = _proxy_ratios_for(e2)
        for d_sub in e2["d_sub_list"]:
            for seed in config["seeds"]:
                for method in e2["methods"]:
                    add_task(e2["dataset"], method, e2["eps"], seed, proxy_ratios=proxy_ratios, d_sub=d_sub)

    if exp in (None, "e3", "all"):
        e3 = config["experiments"]["e3"]
        proxy_ratios = _proxy_ratios_for(e3)
        for dataset in e3["datasets"]:
            for seed in config["seeds"]:
                for method in e3["methods"]:
                    add_task(dataset, method, e3["eps"], seed, proxy_ratios=proxy_ratios)

    if exp in (None, "e4", "all"):
        e4 = config["experiments"]["e4"]
        proxy_ratios = _proxy_ratios_for(e4)
        add_task(e4["dataset"], e4["method"], e4["eps"], e4["seed"], proxy_ratios=proxy_ratios)

    if exp in (None, "e5", "all"):
        e5 = config["experiments"]["e5"]
        proxy_ratios = _proxy_ratios_for(e5)
        seeds_e5 = e5.get("seeds", config["seeds"][:5])
        syn_cfg = datasets_cfg["synthetic"]
        for n in syn_cfg["n_list"]:
            for d in syn_cfg["d_list"]:
                for seed in seeds_e5:
                    for method in e5["methods"]:
                        add_task(
                            "synthetic",
                            method,
                            e5["eps"],
                            seed,
                            proxy_ratios=proxy_ratios,
                            n_override=n,
                            d_override=d,
                            k_override=syn_cfg["k"],
                        )

    if exp in (None, "e6", "all"):
        e6 = config["experiments"]["e6"]
        proxy_ratios = _proxy_ratios_for(e6)
        for seed in e6["seeds"]:
            for method in e6["methods"]:
                add_task(e6["dataset"], method, e6["eps"], seed, proxy_ratios=proxy_ratios)

    return tasks


def _run_task(task: Dict[str, Any], config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    return methods.run_method(
        dataset=task["dataset"],
        method=task["method"],
        eps_tot=task["eps"],
        seed=task["seed"],
        config=config,
        out_dir=out_dir,
        d_sub=task.get("d_sub"),
        n_override=task.get("n_override"),
        d_override=task.get("d_override"),
        k_override=task.get("k_override"),
        proxy_eps_ratio=task.get("proxy_eps_ratio"),
        proxy_eps=task.get("proxy_eps"),
    )


def run_smoke(config: Dict[str, Any], out_dir: str) -> List[Dict[str, Any]]:
    smoke = config["smoke"]
    task = {
        "dataset": smoke["dataset"],
        "method": None,
        "eps": smoke["eps"],
        "seed": smoke["seed"],
        "n_override": smoke["n"],
        "d_override": smoke["d"],
        "k_override": smoke["k"],
    }
    results = []
    for method in smoke["methods"]:
        t = task.copy()
        t["method"] = method
        res = _run_task(t, config, out_dir)
        results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SA-HDPCA experiments.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="outputs/run1")
    parser.add_argument("--exp", type=str, default="all", help="Select experiment id (e0-e6) or 'all'.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test instead of full experiments.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ensure_dir(args.out)
    set_thread_env()
    # joblib 在非 ASCII 路径下容易触发编码问题，强制使用 ASCII 临时目录。
    temp_dir = os.environ.get("JOBLIB_TEMP_FOLDER", r"C:\joblib_temp")
    ensure_dir(temp_dir)
    os.environ["JOBLIB_TEMP_FOLDER"] = temp_dir

    exp_arg = args.exp.lower() if args.exp else "all"
    if args.smoke or exp_arg == "e0":
        results = run_smoke(config, args.out)
    else:
        exp_for_tasks = exp_arg if exp_arg != "all" else None
        tasks = _build_tasks(config, exp=exp_for_tasks)
        runner = joblib.Parallel(n_jobs=config.get("jobs", 6), backend="loky", temp_folder=temp_dir)
        work = (joblib.delayed(_run_task)(task, config, args.out) for task in tasks)
        results = runner(work)

    results_path = os.path.join(args.out, "results.csv")
    df_new = pd.DataFrame(results)
    if not df_new.empty:
        df_new["git_rev_short"] = _git_rev_short()
    if os.path.exists(results_path):
        df_prev = pd.read_csv(results_path)
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
    else:
        df_all = df_new

    dedup_keys = [
        col
        for col in [
            "dataset",
            "method",
            "eps_tot",
            "seed",
            "k",
            "d",
            "r",
            "T",
            "n",
            "eps_proxy",
            "eps_proxy_ratio",
            "algo_variant",
            "config_hash",
            "git_rev_short",
        ]
        if col in df_all.columns
    ]
    if dedup_keys:
        df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")

    df_all.to_csv(results_path, index=False)

    figs_dir = os.path.join(args.out, "figures")
    ensure_dir(figs_dir)
    generate_figures(df_all, figs_dir, config)

    summary_path = os.path.join(args.out, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# SA-HDPCA outputs\n")
        f.write(f"- results: {results_path}\n")
        f.write(f"- figures dir: {figs_dir}\n")
        f.write(f"- histories: {os.path.join(args.out, 'history')}/*.csv\n")


if __name__ == "__main__":
    main()
