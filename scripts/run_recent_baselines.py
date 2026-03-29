#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from baselines.dbdp import fit_predict as dbdp_fit_predict
from baselines.dpdp import fit_predict as dpdp_fit_predict
from baselines.gapbas_scheduler import optimize_schedule
from src.data import load_gas_sensor, load_uci_har
from src.feedback import make_fixed_tail_schedule
from src.init_av import fuzzy_av_init, kmeanspp_init
from src.kmeans_dp import dp_kmeans
from src.metrics import macro_f1
from src.preprocess import preprocess
from src import methods as core_methods
from src.utils import ensure_dir

STRICT_BEST_LABEL = "SA-HDPCA(strict-best)"
METHOD_CFG_KEYS = {
    "feature_mode",
    "use_pca",
    "init_mode",
    "budget_mode",
    "schedule_kind",
    "feedback_overrides",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run secondary Tier B recent baselines while using the strict core implementation for SA-HDPCA/PCA-DP references."
    )
    parser.add_argument("--datasets", nargs="+", default=["HAR", "GAS"], help="Datasets to run (HAR GAS).")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.8, 1.0], help="Privacy budgets.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)), help="Seeds.")
    parser.add_argument("--T", type=int, default=20, help="Number of iterations for DP k-means.")
    parser.add_argument("--clip_B", type=float, default=3.0, help="Clipping bound for preprocessing.")
    parser.add_argument("--pca_r", type=int, default=50, help="Internal PCA dimension applied to all methods.")
    parser.add_argument("--k", type=int, default=None, help="Override cluster count for all datasets.")
    parser.add_argument("--out_dir", type=str, default="outputs/compare_recent_2024_2025", help="Output directory.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (threaded).")
    parser.add_argument("--ga_pop", type=int, default=30, help="GAPBAS population size.")
    parser.add_argument("--ga_gens", type=int, default=40, help="GAPBAS generations.")
    parser.add_argument("--ga_max_samples", type=int, default=3000, help="Max samples for GAPBAS fitness eval.")
    parser.add_argument("--dbdp_max_points", type=int, default=3000, help="Cap on points used inside DBDP.")
    parser.add_argument("--max_points", type=int, default=None, help="Optional cap on number of points per dataset.")
    parser.add_argument("--proxy_eps_ratio", type=float, default=0.1, help="Proxy budget ratio for SA-HDPCA/PCA-DP.")
    parser.add_argument("--proxy_eps_min", type=float, default=0.15, help="Minimum proxy budget for SA-HDPCA/PCA-DP.")
    parser.add_argument("--proxy_init_frac", type=float, default=0.1, help="Subsample fraction for proxy-based init.")
    parser.add_argument("--proxy_init_min", type=int, default=500, help="Minimum proxy points for init subsample.")
    parser.add_argument("--proxy_init_mode", type=str, default="noisy_means", help="Init mode for SA-HDPCA (noisy_means or point_noise).")
    parser.add_argument("--collapse_boost", type=float, default=1.3, help="Collapse boost for SA-HDPCA schedule.")
    parser.add_argument("--collapse_threshold", type=float, default=0.6, help="Collapse threshold for SA-HDPCA.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path for full SA-HDPCA/PCA-DP runs.")
    parser.add_argument(
        "--best-configs",
        type=str,
        default="outputs/tier_a_strict/best_configs.json",
        help="Path to the frozen Tier A best-config file used to select the strongest strict SA-HDPCA variant per dataset.",
    )
    parser.add_argument("--use_full_method", action="store_true", help="Use full pipeline for SA-HDPCA/PCA-DP from src.methods.")
    parser.add_argument("--history", type=str, default=None, help="Optional path to historical raw_results.csv.")
    parser.add_argument(
        "--reuse_history",
        action="store_true",
        help="If set, reuse any available methods from history (not just SA-HDPCA/PCA-DP).",
    )
    parser.add_argument(
        "--rerun_methods",
        nargs="+",
        default=[],
        help="Method names to rerun even if present in history.",
    )
    parser.set_defaults(use_full_method=True)
    return parser.parse_args()


def load_dataset(name: str, clip_B: float) -> Tuple[np.ndarray, np.ndarray, int]:
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


def apply_pca(Z: np.ndarray, r: int, seed: int = 0) -> Tuple[np.ndarray, PCA]:
    r_use = min(r, Z.shape[1])
    pca = PCA(n_components=r_use, random_state=seed, svd_solver="randomized")
    Z_pca = pca.fit_transform(Z)
    return Z_pca, pca


def relabel_sequential(labels: np.ndarray) -> Tuple[np.ndarray, int]:
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


def top_variance_idx(Z: np.ndarray, m: int = 100) -> np.ndarray:
    m_use = min(m, Z.shape[1])
    vars_ = np.var(Z, axis=0)
    return np.argsort(vars_)[::-1][:m_use]


def maybe_subsample_proxy(
    Z_proxy: np.ndarray,
    rng: np.random.Generator,
    frac: Optional[float],
    min_points: Optional[int],
) -> np.ndarray:
    if frac is None or frac <= 0 or frac >= 1:
        return Z_proxy
    n = Z_proxy.shape[0]
    target = int(n * frac)
    if min_points is not None:
        target = max(target, int(min_points))
    target = min(max(target, 1), n)
    if target >= n:
        return Z_proxy
    idx = rng.choice(n, size=target, replace=False)
    return Z_proxy[idx]


def dp_init_noisy_means(
    Z_init: np.ndarray,
    k: int,
    eps_proxy: float,
    clip_B: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if eps_proxy <= 0:
        raise ValueError("eps_proxy must be positive for DP init.")
    n, d = Z_init.shape
    if n == 0:
        return rng.uniform(-clip_B, clip_B, size=(k, d))
    labels = np.empty(n, dtype=int)
    perm = rng.permutation(n)
    base = n // k if k > 0 else 0
    remainder = n % k if k > 0 else 0
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        if size <= 0:
            continue
        idx = perm[start : start + size]
        labels[idx] = i
        start += size
    counts = np.bincount(labels, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    for i in range(k):
        if counts[i] > 0:
            sums[i] = Z_init[labels == i].sum(axis=0)
    noisy_counts = counts + rng.laplace(0.0, 1.0 / eps_proxy, size=counts.shape)
    noisy_sums = sums + rng.laplace(0.0, (2.0 * clip_B) / eps_proxy, size=sums.shape)
    counts_safe = np.maximum(noisy_counts, 1.0)
    centroids = noisy_sums / counts_safe[:, None]
    return np.clip(centroids, -clip_B, clip_B)


def resolve_proxy_budget(eps_tot: float, proxy_ratio: float, proxy_min: Optional[float]) -> Tuple[float, float]:
    eps_proxy = float(proxy_ratio) * eps_tot
    if proxy_min is not None:
        eps_proxy = max(eps_proxy, float(proxy_min))
    eps_proxy = max(eps_proxy, 0.0)
    if eps_proxy >= eps_tot:
        raise ValueError("proxy budget must be smaller than eps_tot.")
    return eps_proxy, eps_tot - eps_proxy


def make_proxy(Z: np.ndarray, eps_proxy: float, clip_B: float, rng: np.random.Generator) -> np.ndarray:
    if eps_proxy <= 0:
        return Z.copy()
    noise_scale = (2.0 * clip_B) / eps_proxy
    Z_tilde = Z + rng.laplace(0.0, noise_scale, size=Z.shape)
    return np.clip(Z_tilde, -clip_B, clip_B)


def compute_metrics(Z_full: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray]) -> Tuple[float, float]:
    labels_seq, k_found = relabel_sequential(labels)
    centroids = centroids_from_labels(Z_full, labels_seq, k_found)
    diff = Z_full - centroids[labels_seq]
    sse = float(np.sum(diff * diff))
    if y_true is None:
        return np.nan, sse
    return macro_f1(y_true, labels_seq), sse


def run_sahdpca(
    Z_pca: np.ndarray,
    k: int,
    eps_tot: float,
    T: int,
    clip_B: float,
    seed: int,
    proxy_eps_ratio: float,
    proxy_eps_min: Optional[float],
    proxy_init_frac: Optional[float],
    proxy_init_min: Optional[int],
    proxy_init_mode: str,
    collapse_boost: float,
    collapse_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    eps_proxy, eps_iter = resolve_proxy_budget(eps_tot, proxy_eps_ratio, proxy_eps_min)
    schedule = np.array(make_fixed_tail_schedule(eps_iter, T, low_mult=1.0, high_mult=3.0), dtype=float)
    rng = np.random.default_rng(seed)
    idx = top_variance_idx(Z_pca, m=100)
    Z_use = Z_pca[:, idx]
    init_mode = (proxy_init_mode or "point_noise").strip().lower()
    if init_mode == "noisy_means":
        Z_init = maybe_subsample_proxy(Z_use, rng, proxy_init_frac, proxy_init_min)
        init = dp_init_noisy_means(Z_init, k, eps_proxy, clip_B, rng)
        proxy_points = init
    else:
        Z_proxy = make_proxy(Z_pca, eps_proxy, clip_B, rng)
        Z_proxy_use = Z_proxy[:, idx]
        Z_proxy_init = maybe_subsample_proxy(Z_proxy_use, rng, proxy_init_frac, proxy_init_min)
        init = fuzzy_av_init(Z_proxy_init, k, rng)
        proxy_points = Z_proxy_use
    result = dp_kmeans(
        Z_use,
        init_centroids=init,
        k=k,
        T=T,
        eps_iter=eps_iter,
        clip_norm=clip_B,
        clip_B=clip_B,
        rng=rng,
        eps_schedule=schedule,
        budget_mode="static",
        proxy_points=proxy_points,
        collapse_boost=collapse_boost,
        collapse_threshold=collapse_threshold,
    )
    return result["labels"], result["centroids"]


def run_pcadp(
    Z_pca: np.ndarray,
    k: int,
    eps_tot: float,
    T: int,
    clip_B: float,
    seed: int,
    proxy_eps_ratio: float,
    proxy_eps_min: Optional[float],
    proxy_init_frac: Optional[float],
    proxy_init_min: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    eps_proxy, eps_iter = resolve_proxy_budget(eps_tot, proxy_eps_ratio, proxy_eps_min)
    rng = np.random.default_rng(seed)
    Z_proxy = make_proxy(Z_pca, eps_proxy, clip_B, rng)
    Z_proxy_init = maybe_subsample_proxy(Z_proxy, rng, proxy_init_frac, proxy_init_min)
    init = kmeanspp_init(Z_proxy_init, k, rng)
    result = dp_kmeans(
        Z_pca,
        init_centroids=init,
        k=k,
        T=T,
        eps_iter=eps_iter,
        clip_norm=clip_B,
        clip_B=clip_B,
        rng=rng,
        budget_mode="static",
        proxy_points=Z_proxy,
    )
    return result["labels"], result["centroids"]


def normalize_method(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


CANONICAL_METHODS = {
    "sahdpca": "SA-HDPCA(fb)",
    "sahdpcafb": "SA-HDPCA(fb)",
    "sahdpcafull": "SA-HDPCA(fb)",
    "sahdpcastrictbest": STRICT_BEST_LABEL,
    "sahdpcabest": STRICT_BEST_LABEL,
    "pcadp": "PCA-DP",
    "gapbas": "GAPBAS",
    "dbdp": "DBDP",
    "dpdp": "DPDP",
}


def canonicalize_method(name: str) -> str:
    norm = normalize_method(name)
    return CANONICAL_METHODS.get(norm, name)


def load_best_configs(path: str) -> Dict[str, Dict]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        raw = json.loads(cfg_path.read_text())
    except Exception:
        return {}
    out: Dict[str, Dict] = {}
    for dataset, methods_blob in raw.items():
        ds_key = dataset.upper()
        out[ds_key] = {}
        best_method = None
        best_payload = None
        best_score = float("-inf")
        for method, payload in methods_blob.items():
            out[ds_key][method] = {
                "candidate": dict(payload.get("candidate", {})),
                "summary": dict(payload.get("summary", {})),
            }
            score = float(payload.get("summary", {}).get("f1_mean", float("-inf")))
            if method.startswith("sahdpca") and score > best_score:
                best_score = score
                best_method = method
                best_payload = payload
        if best_method is not None and best_payload is not None:
            out[ds_key]["__strict_best__"] = {
                "internal_method": best_method,
                "candidate": dict(best_payload.get("candidate", {})),
                "summary": dict(best_payload.get("summary", {})),
            }
    return out


def split_dataset_overrides(internal_method: str, candidate: Dict) -> Dict:
    dataset_overrides: Dict = {}
    method_overrides: Dict = {}
    for key, value in candidate.items():
        if key in METHOD_CFG_KEYS:
            method_overrides[key] = value
        else:
            dataset_overrides[key] = value
    if method_overrides:
        dataset_overrides["method_overrides"] = {internal_method: method_overrides}
    return dataset_overrides


def find_column(df: pd.DataFrame, options: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for opt in options:
        col = lower_map.get(opt.lower())
        if col is not None:
            return col
    return None


def load_history(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    col_dataset = find_column(df, ["dataset"])
    col_method = find_column(df, ["method"])
    col_eps = find_column(df, ["eps", "eps_tot", "epsilon"])
    col_seed = find_column(df, ["seed"])
    col_macro = find_column(df, ["macro_f1", "macrof1"])
    col_sse = find_column(df, ["sse"])
    col_runtime = find_column(df, ["runtime", "runtime_seconds", "time"])

    if not all([col_dataset, col_method, col_eps, col_seed, col_macro, col_sse, col_runtime]):
        return None

    out = pd.DataFrame(
        {
            "dataset": df[col_dataset].astype(str).str.upper(),
            "method": df[col_method].astype(str),
            "eps": df[col_eps].astype(float),
            "seed": df[col_seed].astype(int),
            "macro_f1": df[col_macro].astype(float),
            "sse": df[col_sse].astype(float),
            "runtime": df[col_runtime].astype(float),
        }
    )
    out["method"] = out["method"].apply(lambda m: CANONICAL_METHODS.get(normalize_method(m), m))
    return out


def pick_history_path(user_path: Optional[str]) -> Optional[Path]:
    if user_path:
        path = Path(user_path)
        return path if path.exists() else None
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None
    candidates = (
        list(outputs_dir.rglob("raw_results_new.csv"))
        + list(outputs_dir.rglob("raw_results.csv"))
        + list(outputs_dir.rglob("results.csv"))
    )
    for path in candidates:
        df = load_history(path)
        if df is None:
            continue
        if {STRICT_BEST_LABEL, "PCA-DP"}.issubset(set(df["method"].unique())):
            return path
    return None


def mean_std_fmt(values: pd.Series) -> str:
    return f"{values.mean():.3f}+/-{values.std():.3f}"


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, eps), sub in df.groupby(["dataset", "method", "eps"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "eps": eps,
                "macro_f1": mean_std_fmt(sub["macro_f1"]),
                "sse": mean_std_fmt(sub["sse"]),
                "runtime": mean_std_fmt(sub["runtime"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    datasets = [d.upper() for d in args.datasets]
    eps_list = [float(e) for e in args.eps]
    seeds = [int(s) for s in args.seeds]
    full_config: Optional[Dict] = None
    best_cfg_by_dataset = load_best_configs(args.best_configs)
    if args.use_full_method:
        with open(args.config, "r") as f:
            full_config = yaml.safe_load(f)
        full_config["clip_B"] = args.clip_B
        full_config["T"] = args.T
        full_config["proxy_eps_ratio"] = args.proxy_eps_ratio
        full_config["proxy_eps_min"] = args.proxy_eps_min
        full_config["proxy_init_frac"] = args.proxy_init_frac
        full_config["proxy_init_min"] = args.proxy_init_min
        full_config["proxy_init_mode"] = args.proxy_init_mode
        budget_cfg = full_config.get("budget", {})
        budget_cfg["collapse_boost"] = args.collapse_boost
        budget_cfg["collapse_threshold"] = args.collapse_threshold
        if "datasets" in budget_cfg and "gas" in budget_cfg["datasets"]:
            budget_cfg["datasets"]["gas"]["collapse_boost"] = args.collapse_boost
            budget_cfg["datasets"]["gas"]["collapse_threshold"] = args.collapse_threshold
        full_config["budget"] = budget_cfg

    history_path = pick_history_path(args.history)
    history_df = load_history(history_path) if history_path else None

    history_rows: List[Dict] = []
    history_keys = set()
    if history_df is not None:
        method_filter = {STRICT_BEST_LABEL, "PCA-DP"}
        if args.reuse_history:
            method_filter |= {"GAPBAS", "DBDP", "DPDP"}
        history_df = history_df[history_df["method"].isin(method_filter)]
        rerun_set = {canonicalize_method(m) for m in args.rerun_methods}
        if rerun_set:
            history_df = history_df[~history_df["method"].isin(rerun_set)]
        history_df = history_df[history_df["dataset"].isin(datasets)]
        history_df = history_df[history_df["eps"].isin(eps_list)]
        history_df = history_df[history_df["seed"].isin(seeds)]
        history_rows = history_df.to_dict("records")
        history_keys = {(r["dataset"], r["method"], float(r["eps"]), int(r["seed"])) for r in history_rows}

    data_cache: Dict[str, Dict] = {}
    for ds in datasets:
        Z_full, y, k = load_dataset(ds, clip_B=args.clip_B)
        if args.k is not None:
            k = int(args.k)
        if args.max_points is not None and Z_full.shape[0] > args.max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(Z_full.shape[0], size=args.max_points, replace=False)
            Z_full = Z_full[idx]
            y = y[idx] if y is not None else y
        Z_pca, _ = apply_pca(Z_full, r=args.pca_r, seed=0)
        data_cache[ds] = {"Z_full": Z_full, "Z_pca": Z_pca, "y": y, "k": k}

    tasks: List[Tuple[str, float, int, str]] = []
    methods_to_run = ["GAPBAS", "DBDP", "DPDP"]

    for ds in datasets:
        for eps_tot in eps_list:
            for seed in seeds:
                for method in methods_to_run:
                    key = (ds, method, eps_tot, seed)
                    if args.reuse_history and key in history_keys:
                        continue
                    tasks.append((ds, eps_tot, seed, method))

    # Fallback runs for the strongest strict SA-HDPCA variant and PCA-DP if history is missing.
    for ds in datasets:
        for eps_tot in eps_list:
            for seed in seeds:
                for method in [STRICT_BEST_LABEL, "PCA-DP"]:
                    key = (ds, method, eps_tot, seed)
                    if key in history_keys:
                        continue
                    tasks.append((ds, eps_tot, seed, method))

    gapbas_cache: Dict[Tuple[str, float], np.ndarray] = {}
    gapbas_needed = {(ds, eps_tot) for (ds, eps_tot, _seed, method) in tasks if method == "GAPBAS"}
    for ds, eps_tot in sorted(gapbas_needed):
        ga_res = optimize_schedule(
            data_cache[ds]["Z_pca"],
            data_cache[ds]["k"],
            eps_tot,
            args.T,
            seed_eval=0,
            eps_min=0.01,
            eps_max=eps_tot,
            pop_size=args.ga_pop,
            generations=args.ga_gens,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elitism=2,
            clip_B=args.clip_B,
            max_ga_samples=args.ga_max_samples,
        )
        gapbas_cache[(ds, eps_tot)] = ga_res["schedule"]

    def run_task(task: Tuple[str, float, int, str]) -> Dict:
        ds, eps_tot, seed, method = task
        if args.use_full_method and method in {STRICT_BEST_LABEL, "PCA-DP"}:
            if full_config is None:
                raise RuntimeError("Full config not loaded for full method run.")
            dataset_key = ds.upper()
            if method == STRICT_BEST_LABEL:
                best_payload = best_cfg_by_dataset.get(dataset_key, {}).get("__strict_best__", {})
                core_method = best_payload.get("internal_method", "sahdpca")
                dataset_overrides = split_dataset_overrides(core_method, best_payload.get("candidate", {}))
            else:
                core_method = "pca_dp"
                pca_payload = best_cfg_by_dataset.get(dataset_key, {}).get("pca_dp", {})
                dataset_overrides = split_dataset_overrides(core_method, pca_payload.get("candidate", {}))
            result = core_methods.run_method(
                dataset=ds.lower(),
                method=core_method,
                eps_tot=eps_tot,
                seed=seed,
                config=full_config,
                out_dir=str(out_dir),
                proxy_eps_ratio=args.proxy_eps_ratio,
                dataset_overrides=dataset_overrides,
            )
            return {
                "dataset": ds,
                "method": method,
                "eps": eps_tot,
                "seed": seed,
                "macro_f1": result["f1"],
                "sse": result["sse_x"],
                "runtime": result["runtime_ms_total"] / 1000.0,
            }
        ctx = data_cache[ds]
        Z_full = ctx["Z_full"]
        Z_pca = ctx["Z_pca"]
        y = ctx["y"]
        k = ctx["k"]
        rng = np.random.default_rng(seed)
        start = time.perf_counter()

        if method == "GAPBAS":
            schedule = gapbas_cache[(ds, eps_tot)]
            init = kmeanspp_init(Z_pca, k, rng)
            result = dp_kmeans(
                Z_pca,
                init_centroids=init,
                k=k,
                T=args.T,
                eps_iter=eps_tot,
                clip_norm=args.clip_B,
                clip_B=args.clip_B,
                rng=rng,
                eps_schedule=schedule,
                budget_mode="static",
                proxy_points=Z_pca,
                collapse_boost=1.0,
            )
            labels = result["labels"]
        elif method == "DBDP":
            labels, _, _ = dbdp_fit_predict(Z_pca, k, eps_tot, seed, max_points=args.dbdp_max_points)
        elif method == "DPDP":
            labels, _, _ = dpdp_fit_predict(Z_pca, k, eps_tot, seed, T=args.T, clip_B=args.clip_B)
        elif method == "SA-HDPCA(fb)":
            labels, _ = run_sahdpca(
                Z_pca,
                k,
                eps_tot,
                args.T,
                args.clip_B,
                seed,
                args.proxy_eps_ratio,
                args.proxy_eps_min,
                args.proxy_init_frac,
                args.proxy_init_min,
                args.proxy_init_mode,
                args.collapse_boost,
                args.collapse_threshold,
            )
        elif method == "PCA-DP":
            labels, _ = run_pcadp(
                Z_pca,
                k,
                eps_tot,
                args.T,
                args.clip_B,
                seed,
                args.proxy_eps_ratio,
                args.proxy_eps_min,
                args.proxy_init_frac,
                args.proxy_init_min,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        runtime = time.perf_counter() - start
        macro, sse = compute_metrics(Z_full, labels, y)
        return {
            "dataset": ds,
            "method": method,
            "eps": eps_tot,
            "seed": seed,
            "macro_f1": macro,
            "sse": sse,
            "runtime": runtime,
        }

    if args.jobs > 1 and tasks:
        results = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(run_task)(t) for t in tasks)
    else:
        results = [run_task(t) for t in tasks]

    all_rows = history_rows + results
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset", "method", "eps", "seed"]).reset_index(drop=True)

    raw_path = out_dir / "raw_results_new.csv"
    df.to_csv(raw_path, index=False)

    summary_path = out_dir / "summary_new.csv"
    summary_df = make_summary(df)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] wrote {raw_path} with {len(df)} rows")
    print(f"[OK] wrote {summary_path}")
    if history_path is not None:
        print(f"[OK] history source: {history_path}")


if __name__ == "__main__":
    main()
