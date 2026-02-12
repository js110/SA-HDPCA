                      

import argparse

import json

import sys

import time

from pathlib import Path

from typing import Dict, List, Tuple



import numpy as np

import pandas as pd

from sklearn.decomposition import PCA



ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:

    sys.path.append(str(ROOT))



from baselines.dbdp import fit_predict as dbdp_fit_predict

from baselines.dpdp import fit_predict as dpdp_fit_predict

from baselines.gapbas_scheduler import optimize_schedule

from src.data import load_gas_sensor, load_uci_har

from src.feedback import make_fixed_tail_schedule

from src.init_av import fuzzy_av_init, kmeanspp_init, random_init

from src.kmeans_dp import dp_kmeans

from src.metrics import ari, cluster_stats, macro_f1, nmi

from src.preprocess import preprocess

from src.utils import ensure_dir





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run 2024/2025 baseline comparisons.")

    parser.add_argument("--datasets", nargs="+", default=["HAR", "GAS"], help="Datasets to run (HAR GAS).")

    parser.add_argument("--eps", nargs="+", type=float, default=[0.2, 0.5, 0.8, 1.0, 1.5], help="Privacy budgets.")

    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)), help="Seeds.")

    parser.add_argument("--T", type=int, default=20, help="Number of iterations for DP k-means.")

    parser.add_argument("--clip_B", type=float, default=3.0, help="Clipping bound for preprocessing.")

    parser.add_argument("--pca_r", type=int, default=50, help="Internal PCA dimension applied to all methods.")

    parser.add_argument("--out_dir", type=str, default="outputs/compare_2024_2025", help="Output directory.")

    parser.add_argument("--jobs", type=int, default=1, help="(Reserved) parallel jobs; currently sequential.")

    parser.add_argument("--ga_pop", type=int, default=30, help="GAPBAS population size.")

    parser.add_argument("--ga_gens", type=int, default=40, help="GAPBAS generations.")

    parser.add_argument("--ga_max_samples", type=int, default=3000, help="Max samples for GAPBAS fitness eval.")

    parser.add_argument("--max_points", type=int, default=None, help="Optional cap on number of points per dataset (for quick runs/debug).")

    parser.add_argument(

        "--dbdp_max_points",

        type=int,

        default=3000,

        help="Optional cap on points used inside DBDP (remaining points assigned by nearest centroid).",

    )

    parser.add_argument(

        "--methods",

        nargs="+",

        default=[

            "SA-HDPCA",

            "Uniform",

            "GAPBAS",

            "DBDP",

            "DPDP",

            "DP-KMEANS",

            "KMEANSPP-DP",

            "PCA-DP",

            "SAHDPCA_NO_MOD1",

            "SAHDPCA_NO_MOD2",

            "SAHDPCA_NO_MOD3",

        ],

        help="Method list to run.",

    )

    return parser.parse_args()





def load_dataset(name: str, clip_B: float) -> tuple[np.ndarray, np.ndarray, int]:

    if name.upper() == "HAR":

        X, y = load_uci_har()

        k = 6

    elif name.upper() == "GAS":

        X, y, _, _ = load_gas_sensor()

        k = 6

    else:

        raise ValueError(f"Unsupported dataset: {name}")

    Z = preprocess(X, clip_B=clip_B)

    return Z, y, k





def apply_pca(Z: np.ndarray, r: int, seed: int = 0) -> tuple[np.ndarray, PCA]:

    r_use = min(r, Z.shape[1])

    pca = PCA(n_components=r_use, random_state=seed, svd_solver="randomized")

    Z_pca = pca.fit_transform(Z)

    return Z_pca, pca





def relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:

    unique = np.unique(labels)

    mapping = {old: idx for idx, old in enumerate(unique)}

    new_labels = np.array([mapping[l] for l in labels], dtype=int)

    return new_labels, len(unique)





def centroids_from_labels(Z: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:

    centroids = []

    for cid in range(k):

        mask = labels == cid

        if not np.any(mask):

            centroids.append(np.zeros(Z.shape[1], dtype=float))

            continue

        centroids.append(Z[mask].mean(axis=0))

    return np.vstack(centroids)





def apply_feature_mask(Z: np.ndarray, m: int = 100) -> np.ndarray:

    m_use = min(m, Z.shape[1])

    vars_ = np.var(Z, axis=0)

    idx = np.argsort(vars_)[::-1][:m_use]

    return Z[:, idx]





def compute_metrics(Z_full: np.ndarray, labels: np.ndarray, y_true: np.ndarray | None) -> Dict[str, float]:

    labels_seq, k_found = relabel_sequential(labels)

    centroids = centroids_from_labels(Z_full, labels_seq, k_found)

    diff = Z_full - centroids[labels_seq]

    sse = float(np.sum(diff * diff))

    stats = cluster_stats(labels_seq, k_found)

    metrics: Dict[str, float] = {

        "sse": sse,

        "non_empty_clusters": stats["non_empty_k"],

        "max_cluster_ratio": stats["max_cluster_ratio"],

    }

    if y_true is not None:

        metrics["macro_f1"] = macro_f1(y_true, labels_seq)

        metrics["nmi"] = nmi(y_true, labels_seq)

        metrics["ari"] = ari(y_true, labels_seq)

    else:

        metrics["macro_f1"] = np.nan

        metrics["nmi"] = np.nan

        metrics["ari"] = np.nan

    return metrics





def run_sahdpca(Z: np.ndarray, k: int, eps_tot: float, T: int, clip_B: float, seed: int, use_feature_mask: bool = False) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:

    schedule = np.array(make_fixed_tail_schedule(eps_tot, T, low_mult=1.0, high_mult=3.0), dtype=float)

    rng = np.random.default_rng(seed)

    Z_use = apply_feature_mask(Z, m=100) if use_feature_mask else Z

    init = fuzzy_av_init(Z_use, k, rng)

    result = dp_kmeans(

        Z_use,

        init_centroids=init,

        k=k,

        T=T,

        eps_iter=eps_tot,

        clip_B=clip_B,

        rng=rng,

        eps_schedule=schedule,

        budget_mode="static",

        proxy_points=Z_use,

        collapse_boost=1.5,

        collapse_threshold=0.6,

    )

    labels = result["labels"]

    centroids = result["centroids"]

    extra = {"schedule": schedule.tolist()}

    return labels, centroids, extra, schedule





def run_uniform(Z: np.ndarray, k: int, eps_tot: float, T: int, clip_B: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:

    rng = np.random.default_rng(seed)

    init = kmeanspp_init(Z, k, rng)

    result = dp_kmeans(

        Z,

        init_centroids=init,

        k=k,

        T=T,

        eps_iter=eps_tot,

        clip_B=clip_B,

        rng=rng,

        budget_mode="static",

        proxy_points=Z,

    )

    labels = result["labels"]

    centroids = result["centroids"]

    schedule = np.full(T, eps_tot / max(T, 1), dtype=float)

    return labels, centroids, {}, schedule





def main():

    args = parse_args()

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    schedules_dir = out_dir / "schedules"

    ensure_dir(schedules_dir)



    datasets = [d.upper() for d in args.datasets]

    eps_list = [float(e) for e in args.eps]

    seeds = [int(s) for s in args.seeds]

    T = args.T

    clip_B = args.clip_B



    results: List[Dict] = []



                          

    data_cache: Dict[str, Dict] = {}

    for ds in datasets:

        Z, y, k = load_dataset(ds, clip_B=clip_B)

        if args.max_points is not None and Z.shape[0] > args.max_points:

            rng = np.random.default_rng(0)

            idx = rng.choice(Z.shape[0], size=args.max_points, replace=False)

            Z = Z[idx]

            y = y[idx] if y is not None else y

        Z_pca, pca = apply_pca(Z, r=args.pca_r, seed=0)

        data_cache[ds] = {"Z": Z, "Z_pca": Z_pca, "y": y, "k": k, "pca": pca}



    gapbas_cache: Dict[Tuple[str, float], np.ndarray] = {}



    methods = [m.upper() for m in args.methods]

    method_label = {

        "SA-HDPCA": "SA-HDPCA",

        "UNIFORM": "Uniform",

        "GAPBAS": "GAPBAS",

        "DBDP": "DBDP",

        "DPDP": "DPDP",

        "DP-KMEANS": "DP-KMEANS",

        "KMEANSPP-DP": "KMEANSPP-DP",

        "PCA-DP": "PCA-DP",

        "SAHDPCA_NO_MOD1": "SAHDPCA-wo-M1",

        "SAHDPCA_NO_MOD2": "SAHDPCA-wo-M2",

        "SAHDPCA_NO_MOD3": "SAHDPCA-wo-M3",

    }



    for ds in datasets:

        ctx = data_cache[ds]

        Z_full = ctx["Z"]

        Z_pca = ctx["Z_pca"]

        y = ctx["y"]

        k = ctx["k"]



        for eps_tot in eps_list:

                                                      

            _, _, _, sched_uniform = run_uniform(Z_pca, k, eps_tot, T, clip_B, seed=0)

            uniform_sched_path = schedules_dir / f"{ds}_uniform_eps{eps_tot}.json"

            uniform_sched_path.write_text(json.dumps({"schedule": sched_uniform.tolist(), "eps_tot": eps_tot, "T": T}))



            _, _, _, sched_sa = run_sahdpca(Z_pca, k, eps_tot, T, clip_B, seed=0, use_feature_mask=True)

            sahdpca_sched_path = schedules_dir / f"{ds}_sahdpca_eps{eps_tot}.json"

            sahdpca_sched_path.write_text(json.dumps({"schedule": sched_sa.tolist(), "eps_tot": eps_tot, "T": T}))



            if (ds, eps_tot) not in gapbas_cache:

                ga_res = optimize_schedule(

                    Z_pca,

                    k,

                    eps_tot,

                    T,

                    seed_eval=0,

                    eps_min=0.01,

                    eps_max=eps_tot,

                    pop_size=args.ga_pop,

                    generations=args.ga_gens,

                    crossover_rate=0.8,

                    mutation_rate=0.2,

                    elitism=2,

                    clip_B=clip_B,

                    max_ga_samples=args.ga_max_samples,

                )

                gapbas_cache[(ds, eps_tot)] = ga_res["schedule"]

                gapbas_path = schedules_dir / f"{ds}_gapbas_eps{eps_tot}.json"

                gapbas_path.write_text(json.dumps({"schedule": ga_res["schedule"].tolist(), "eps_tot": eps_tot, "T": T}))

            gapbas_schedule = gapbas_cache[(ds, eps_tot)]



            for seed in seeds:

                rng = np.random.default_rng(seed)

                for method in methods:

                    start = time.perf_counter()

                    method_key = method.upper()

                    if method_key == "SA-HDPCA":

                        labels, centroids_pca, extra, _ = run_sahdpca(Z_pca, k, eps_tot, T, clip_B, seed, use_feature_mask=True)

                    elif method_key == "UNIFORM":

                        labels, centroids_pca, extra, _ = run_uniform(Z_pca, k, eps_tot, T, clip_B, seed)

                    elif method_key == "GAPBAS":

                        init = kmeanspp_init(Z_pca, k, rng)

                        result = dp_kmeans(

                            Z_pca,

                            init_centroids=init,

                            k=k,

                            T=T,

                            eps_iter=eps_tot,

                            clip_B=clip_B,

                            rng=rng,

                            eps_schedule=gapbas_schedule,

                            budget_mode="static",

                            proxy_points=Z_pca,

                            collapse_boost=1.0,

                        )

                        labels = result["labels"]

                        centroids_pca = result["centroids"]

                        extra = {"schedule": gapbas_schedule.tolist()}

                    elif method_key == "DBDP":

                        labels, centroids_pca, extra = dbdp_fit_predict(

                            Z_pca, k, eps_tot, seed, max_points=args.dbdp_max_points

                        )

                    elif method_key == "DPDP":

                        labels, centroids_pca, extra = dpdp_fit_predict(Z_pca, k, eps_tot, seed, T=T, clip_B=clip_B)

                    elif method_key == "DP-KMEANS":

                        init = random_init(Z_pca, k, rng)

                        result = dp_kmeans(

                            Z_pca,

                            init_centroids=init,

                            k=k,

                            T=T,

                            eps_iter=eps_tot,

                            clip_B=clip_B,

                            rng=rng,

                            budget_mode="static",

                            proxy_points=Z_pca,

                        )

                        labels = result["labels"]

                        centroids_pca = result["centroids"]

                        extra = {}

                    elif method_key == "KMEANSPP-DP":

                        init = kmeanspp_init(Z_pca, k, rng)

                        result = dp_kmeans(

                            Z_pca,

                            init_centroids=init,

                            k=k,

                            T=T,

                            eps_iter=eps_tot,

                            clip_B=clip_B,

                            rng=rng,

                            budget_mode="static",

                            proxy_points=Z_pca,

                        )

                        labels = result["labels"]

                        centroids_pca = result["centroids"]

                        extra = {}

                    elif method_key == "PCA-DP":

                        init = kmeanspp_init(Z_pca, k, rng)

                        result = dp_kmeans(

                            Z_pca,

                            init_centroids=init,

                            k=k,

                            T=T,

                            eps_iter=eps_tot,

                            clip_B=clip_B,

                            rng=rng,

                            budget_mode="static",

                            proxy_points=Z_pca,

                        )

                        labels = result["labels"]

                        centroids_pca = result["centroids"]

                        extra = {}

                    elif method_key in {"SAHDPCA_NO_MOD1", "SAHDPCA_NO_MOD2", "SAHDPCA_NO_MOD3"}:

                                                                              

                        if not (ds == "HAR" and abs(eps_tot - 0.8) < 1e-9):

                            continue

                        sched = np.array(make_fixed_tail_schedule(eps_tot, T, low_mult=1.0, high_mult=3.0), dtype=float)

                        init_fn = fuzzy_av_init

                        if method_key == "SAHDPCA_NO_MOD2":

                            init_fn = kmeanspp_init

                        if method_key == "SAHDPCA_NO_MOD3":

                            sched = np.full(T, eps_tot / max(T, 1), dtype=float)

                        use_mask = False

                        if method_key == "SAHDPCA_NO_MOD1":

                            use_mask = False

                        elif method_key in {"SAHDPCA_NO_MOD2", "SAHDPCA_NO_MOD3"}:

                            use_mask = True

                        Z_use = apply_feature_mask(Z_pca, m=100) if use_mask else Z_pca

                        init = init_fn(Z_use, k, rng)

                        result = dp_kmeans(

                            Z_use,

                            init_centroids=init,

                            k=k,

                            T=T,

                            eps_iter=eps_tot,

                            clip_B=clip_B,

                            rng=rng,

                            eps_schedule=sched,

                            budget_mode="static",

                            proxy_points=Z_use,

                        )

                        labels = result["labels"]

                        centroids_pca = result["centroids"]

                        extra = {"schedule": sched.tolist()}

                    else:

                        raise ValueError(f"Unknown method: {method}")

                    runtime_seconds = time.perf_counter() - start



                                                                         

                    metrics = compute_metrics(Z_full, labels, y_true=y)



                    row = {

                        "dataset": ds,

                        "method": method_label.get(method_key, method_key),

                        "eps_tot": eps_tot,

                        "seed": seed,

                        "macro_f1": metrics["macro_f1"],

                        "nmi": metrics["nmi"],

                        "ari": metrics["ari"],

                        "sse": metrics["sse"],

                        "non_empty_clusters": metrics["non_empty_clusters"],

                        "max_cluster_ratio": metrics["max_cluster_ratio"],

                        "runtime_seconds": runtime_seconds,

                    }

                    results.append(row)



    df = pd.DataFrame(results)

    raw_path = out_dir / "raw_results.csv"

    df.to_csv(raw_path, index=False)



                      

    agg_funcs = {m: ["mean", "std"] for m in ["macro_f1", "nmi", "ari", "sse", "non_empty_clusters", "max_cluster_ratio", "runtime_seconds"]}

    summary = df.groupby(["dataset", "method", "eps_tot"]).agg(agg_funcs)

    summary.columns = ["_".join(col) for col in summary.columns]

    summary = summary.reset_index()

    summary_path = out_dir / "summary_results.csv"

    summary.to_csv(summary_path, index=False)



    print(f"[OK] wrote {raw_path} with {len(df)} rows")

    print(f"[OK] wrote {summary_path}")





if __name__ == "__main__":

    main()

