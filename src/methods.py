import os

import time

from typing import Any, Dict, Optional



import numpy as np

import pandas as pd

from sklearn.decomposition import PCA



from .data import load_gas_sensor, load_uci_har, make_synthetic, make_synthetic_stream

from .feedback import make_fixed_tail_schedule, make_strong_contrast_schedule

from .init_av import fuzzy_av_init, kmeanspp_init, kmeanspp_rr_init, random_init

from .kmeans_dp import dp_kmeans

from .metrics import ari, cluster_stats, macro_f1, nmi, sse_in_X, sse_in_Z

from .preprocess import preprocess

from .pus import apply_pus

from .utils import configure_environment, ensure_dir





def _load_dataset(

    dataset: str,

    dataset_cfg: Dict[str, Any],

    seed: int,

    d_sub: Optional[int] = None,

    n_override: Optional[int] = None,

    d_override: Optional[int] = None,

    k_override: Optional[int] = None,

) -> tuple[np.ndarray, np.ndarray, int]:

    if dataset == "har":

        X, y = load_uci_har(dataset_cfg["root"])

        k = dataset_cfg["k"]

        if d_sub is not None:

            X = X[:, :d_sub]

    elif dataset == "gas":

        X, y, _, _ = load_gas_sensor(dataset_cfg["root"])

        k = dataset_cfg["k"]

        if d_sub is not None:

            X = X[:, :d_sub]

    elif dataset == "synthetic":

        n = n_override if n_override is not None else dataset_cfg.get("n", dataset_cfg["n_list"][0])

        d = d_override if d_override is not None else dataset_cfg.get("d", dataset_cfg["d_list"][0])

        k = k_override if k_override is not None else dataset_cfg["k"]

        weights = dataset_cfg.get("cluster_weights")

        cluster_std = dataset_cfg.get("cluster_std", 1.0)

        X, y = make_synthetic(n=n, d=d, k=k, seed=seed, weights=weights, cluster_std=cluster_std)

    else:

        raise ValueError(f"Unsupported dataset: {dataset}")

    return X, y, k





def _get_feature_params(dataset: str, dataset_cfg: Dict[str, Any], d: int) -> tuple[int, int]:

    if dataset == "har":

        m = min(dataset_cfg.get("pus_top_m", 128), d)

        r = min(dataset_cfg.get("pca_r", 64), d)

    elif dataset == "gas":

        m = min(dataset_cfg.get("pus_top_m", 64), d)

        r = min(dataset_cfg.get("pca_r", 32), d)

    else:

        m = min(128, d)

        r = min(64, d)

    return m, r





PUS_METHODS = {

    "sahdpca",

    "sahdpca_wo_init",

    "sahdpca_wo_feedback",

    "sahdpca_strong",

    "sahdpca_fb_v3",

    "sahdpca_fb_v2",

    "sahdpca_proxy_kpp",

    "sahdpca_proxy_rr",

}



PCA_METHODS = {

    "pca_dp",

    "sahdpca",

    "sahdpca_wo_pus",

    "sahdpca_wo_init",

    "sahdpca_wo_feedback",

    "sahdpca_strong",

    "sahdpca_fb_v3",

    "sahdpca_fb_v2",

    "sahdpca_proxy_kpp",

    "sahdpca_proxy_rr",

    "sahdpca_pca_only",

    "sahdpca_rand_feat",

}





def _apply_dimensionality(

    method: str,

    dataset: str,

    Z: np.ndarray,

    Z_tilde: np.ndarray,

    y: np.ndarray,

    dataset_cfg: Dict[str, Any],

    seed: int,

    clip_B: float,

    feature_mode: Optional[str] = None,

    use_pca: Optional[bool] = None,

    rng: Optional[np.random.Generator] = None,

) -> tuple[np.ndarray, np.ndarray, int]:

    """
    Return Z_proc, Z_tilde_proc, final_dim.
    """

    m, r = _get_feature_params(dataset, dataset_cfg, Z.shape[1])

    if feature_mode is None:

        feature_mode = "pus" if method in PUS_METHODS else "none"

    if use_pca is None:

        use_pca = method in PCA_METHODS



    Z_proc = Z

    Z_tilde_proc = Z_tilde



    if feature_mode == "pus":

        Z_proc, idx, _ = apply_pus(Z_proc, y, m=m, seed=seed)

        Z_tilde_proc = Z_tilde_proc[:, idx]

    elif feature_mode == "random":

        if rng is None:

            rng = np.random.default_rng(seed)

        if m < Z_proc.shape[1]:

            idx = rng.choice(Z_proc.shape[1], size=m, replace=False)

            Z_proc = Z_proc[:, idx]

            Z_tilde_proc = Z_tilde_proc[:, idx]

    elif feature_mode != "none":

        raise ValueError(f"Unsupported feature_mode: {feature_mode}")



    final_dim = Z_proc.shape[1]



    if use_pca and r < final_dim:

        pca = PCA(n_components=r, random_state=seed, svd_solver="randomized")

        fit_data = Z_tilde_proc if method == "pca_dp" else Z_proc

        pca.fit(fit_data)

        Z_tilde_proc = pca.transform(Z_tilde_proc)

        Z_proc = pca.transform(Z_proc)

        Z_tilde_proc = np.clip(Z_tilde_proc, -clip_B, clip_B)

        Z_proc = np.clip(Z_proc, -clip_B, clip_B)

        final_dim = r



    return Z_proc, Z_tilde_proc, final_dim





SAHDPCA_METHODS = {

    "sahdpca",

    "sahdpca_wo_feedback",

    "sahdpca_wo_pus",

    "sahdpca_wo_init",

    "sahdpca_strong",

    "sahdpca_fb_v3",

    "sahdpca_fb_v2",

    "sahdpca_proxy_kpp",

    "sahdpca_proxy_rr",

    "sahdpca_pca_only",

    "sahdpca_rand_feat",

}





def _init_centroids(

    method: str,

    Z_proc: np.ndarray,

    Z_proxy_init: np.ndarray,

    k: int,

    rng: np.random.Generator,

    init_mode: Optional[str] = None,

    rr_restarts: int = 10,

) -> np.ndarray:

                                                                            

    init_data = Z_proxy_init if method in SAHDPCA_METHODS else Z_proc

    mode = init_mode

    if mode is None:

        if method == "dp_kmeans":

            mode = "random"

        elif method in {"kmeanspp_dp", "pca_dp", "sahdpca_wo_init"}:

            mode = "kpp"

        else:

            mode = "fuzzy"

    if mode == "random":

        return random_init(init_data, k, rng)

    if mode == "kpp":

        return kmeanspp_init(init_data, k, rng)

    if mode == "kpp_rr":

        return kmeanspp_rr_init(init_data, k, rng, restarts=rr_restarts)

    if mode == "fuzzy":

        return fuzzy_av_init(init_data, k, rng)

    raise ValueError(f"Unknown init_mode: {mode}")





def _feedback_params(

    budget_mode: str,

    budget_cfg: Dict[str, Any],

    per_ds: Dict[str, Any],

    overrides: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    mode_keys = {

        "feedback": ["warmup", "beta", "gamma", "drift_clip", "eps_min_ratio", "eps_max_ratio"],

        "feedback_v2": [

            "warmup",

            "beta",

            "gamma",

            "drift_clip",

            "eps_min_ratio",

            "eps_max_ratio",

            "p",

            "adj_clip_low",

            "adj_clip_high",

            "recovery_scale",

        ],

        "feedback_v3": [

            "warmup",

            "beta",

            "gamma",

            "drift_clip",

            "eps_min_ratio",

            "eps_max_ratio",

            "p",

            "time_power",

            "adj_clip_low",

            "adj_clip_high",

            "recovery_scale",

        ],

    }

    keys = mode_keys.get(budget_mode, [])

    params: Dict[str, Any] = {}

    for key in keys:

        if key in per_ds:

            params[key] = per_ds[key]

        elif key in budget_cfg:

            params[key] = budget_cfg[key]

    if overrides:

        params.update(overrides)

    return params





def _resolve_method_config(method: str) -> Dict[str, Any]:

    feature_mode = None

    use_pca = None

    init_mode = None

    budget_mode = "static"

    schedule_kind = None

    feedback_overrides = None



    if method == "sahdpca":

        schedule_kind = "fixed_tail"

    elif method == "sahdpca_strong":

        schedule_kind = "strong"

    elif method in {

        "sahdpca_fb_v3",

        "sahdpca_proxy_kpp",

        "sahdpca_proxy_rr",

        "sahdpca_pca_only",

        "sahdpca_rand_feat",

    }:

        budget_mode = "feedback_v3"

    elif method == "sahdpca_fb_v2":

        budget_mode = "feedback_v2"

        feedback_overrides = {"recovery_scale": 0.0}



    if method == "sahdpca_proxy_kpp":

        init_mode = "kpp"

    elif method == "sahdpca_proxy_rr":

        init_mode = "kpp_rr"



    if method == "sahdpca_rand_feat":

        feature_mode = "random"



    return {

        "feature_mode": feature_mode,

        "use_pca": use_pca,

        "init_mode": init_mode,

        "budget_mode": budget_mode,

        "schedule_kind": schedule_kind,

        "feedback_overrides": feedback_overrides,

    }





def _maybe_subsample_proxy(

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





def _dp_init_noisy_means(

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

    centroids = np.clip(centroids, -clip_B, clip_B)

    return centroids





def _resolve_proxy_budget(

    eps_tot: float,

    config: Dict[str, Any],

    proxy_ratio: Optional[float] = None,

    proxy_eps: Optional[float] = None,

) -> tuple[float, float]:

    if proxy_eps is not None:

        eps_proxy = float(proxy_eps)

    elif proxy_ratio is not None:

        eps_proxy = float(proxy_ratio) * eps_tot

    else:

        eps_proxy_cfg = config.get("proxy_eps", None)

        proxy_ratio_cfg = config.get("proxy_eps_ratio", None)

        if eps_proxy_cfg is not None:

            eps_proxy = float(eps_proxy_cfg)

        elif proxy_ratio_cfg is not None:

            eps_proxy = float(proxy_ratio_cfg) * eps_tot

        else:

            eps_proxy = 0.1 * eps_tot

    eps_proxy = max(eps_proxy, 0.0)

    proxy_min = config.get("proxy_eps_min", None)

    if proxy_min is not None:

        eps_proxy = max(eps_proxy, float(proxy_min))

    if eps_proxy >= eps_tot:

        raise ValueError("eps_proxy must be smaller than eps_tot.")

    eps_iter = eps_tot - eps_proxy

    return eps_proxy, eps_iter





def _run_on_data(

    X: np.ndarray,

    y: np.ndarray,

    k_val: int,

    dataset: str,

    method: str,

    eps_tot: float,

    seed: int,

    config: Dict[str, Any],

    out_dir: str,

    dataset_cfg: Dict[str, Any],

    d_orig: int,

    d_sub: Optional[int] = None,

    proxy_eps_ratio: Optional[float] = None,

    proxy_eps: Optional[float] = None,

    dataset_tag: Optional[str] = None,

    budget_overrides: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    rng = configure_environment(seed)

    T = config["T"]

    clip_B = config["clip_B"]

    Z = preprocess(X, clip_B=clip_B)

    method_cfg = _resolve_method_config(method)

    feature_mode = method_cfg["feature_mode"]

    use_pca = method_cfg["use_pca"]

    init_mode = method_cfg["init_mode"]

    budget_mode = method_cfg["budget_mode"]

    schedule_kind = method_cfg["schedule_kind"]

    feedback_overrides = method_cfg["feedback_overrides"]



    eps_proxy, eps_iter = _resolve_proxy_budget(

        eps_tot,

        config,

        proxy_ratio=proxy_eps_ratio,

        proxy_eps=proxy_eps,

    )

    if eps_iter <= 0:

        raise ValueError("eps_tot too small; eps_iter must be positive.")

    proxy_init_mode = config.get("proxy_init_mode", "point_noise")

    if isinstance(proxy_init_mode, str):

        proxy_init_mode = proxy_init_mode.lower()

    use_noisy_means = proxy_init_mode in {"noisy_means", "noisymean", "noisy-mean"} and method in SAHDPCA_METHODS

    proxy_after_reduction = bool(config.get("proxy_after_reduction", False))

    use_point_proxy = eps_proxy > 0 and not use_noisy_means

    use_post_noise = use_point_proxy and proxy_after_reduction and method in SAHDPCA_METHODS

    if use_post_noise:

        Z_tilde = Z.copy()

    elif use_point_proxy:

        noise_scale = (2.0 * clip_B) / eps_proxy

        Z_tilde = Z + rng.laplace(0.0, noise_scale, size=Z.shape)

        Z_tilde = np.clip(Z_tilde, -clip_B, clip_B)

    else:

        Z_tilde = Z.copy()



    Z_proc, Z_tilde_proc, final_dim = _apply_dimensionality(

        method=method,

        dataset=dataset,

        Z=Z,

        Z_tilde=Z_tilde,

        y=y,

        dataset_cfg=dataset_cfg,

        seed=seed,

        clip_B=clip_B,

        feature_mode=feature_mode,

        use_pca=use_pca,

        rng=rng,

    )

    if use_post_noise:

        if eps_proxy > 0:

            noise_scale = (2.0 * clip_B) / eps_proxy

            Z_tilde_proc = Z_proc + rng.laplace(0.0, noise_scale, size=Z_proc.shape)

            Z_tilde_proc = np.clip(Z_tilde_proc, -clip_B, clip_B)

        else:

            Z_tilde_proc = Z_proc.copy()



    proxy_init_frac = config.get("proxy_init_frac", None)

    proxy_init_min = config.get("proxy_init_min", None)

    if use_noisy_means:

        Z_proxy_init = _maybe_subsample_proxy(Z_proc, rng, proxy_init_frac, proxy_init_min)

        init_centroids = _dp_init_noisy_means(Z_proxy_init, k_val, eps_proxy, clip_B, rng)

        proxy_points = init_centroids

    else:

        Z_proxy_init = _maybe_subsample_proxy(Z_tilde_proc, rng, proxy_init_frac, proxy_init_min)

        rr_restarts = int(config.get("proxy_rr_restarts", 10))

        init_centroids = _init_centroids(

            method,

            Z_proc,

            Z_proxy_init,

            k_val,

            rng,

            init_mode=init_mode,

            rr_restarts=rr_restarts,

        )

        proxy_points = Z_tilde_proc

    budget_cfg = dict(config.get("budget", {}))

    if budget_overrides:

        budget_cfg.update(budget_overrides)

    per_ds = budget_cfg.get("datasets", {}).get(dataset, {})

    collapse_threshold = per_ds.get("collapse_threshold", budget_cfg.get("collapse_threshold", 0.55))

    collapse_min_ratio = per_ds.get("collapse_min_ratio", config.get("collapse_min_ratio", 0.01))



    start = time.perf_counter()

    if schedule_kind == "fixed_tail":

        sched_cfg = budget_cfg.get("fixed", {})

        sched_cfg = {**sched_cfg, **per_ds.get("fixed", {})}

        eps_schedule = make_fixed_tail_schedule(

            eps_tot=eps_iter,

            T=T,

            low_mult=sched_cfg.get("low_mult", 1.0),

            high_mult=sched_cfg.get("high_mult", 3.0),

        )

        collapse_boost = per_ds.get("collapse_boost", budget_cfg.get("collapse_boost", 1.5))

        eps_cap = per_ds.get("eps_cap", budget_cfg.get("eps_cap", None))

        result = dp_kmeans(

            Z_proc,

            init_centroids=init_centroids,

            k=k_val,

            T=T,

            eps_iter=eps_iter,

            clip_B=clip_B,

            rng=rng,

            eps_schedule=eps_schedule,

            budget_mode="static",

            proxy_points=proxy_points,

            collapse_boost=collapse_boost,

            collapse_threshold=collapse_threshold,

            eps_cap=eps_cap,

        )

    elif schedule_kind == "strong":

        eps_schedule = make_strong_contrast_schedule(

            eps_iter,

            T,

            low_frac=budget_cfg.get("strong_low_frac", 0.7),

            eps_min_ratio=budget_cfg.get("strong_eps_min_ratio", 0.7),

        )

        result = dp_kmeans(

            Z_proc,

            init_centroids=init_centroids,

            k=k_val,

            T=T,

            eps_iter=eps_iter,

            clip_B=clip_B,

            rng=rng,

            eps_schedule=eps_schedule,

            budget_mode="static",

            proxy_points=proxy_points,

        )

    else:

        extra_overrides = {}

        if budget_overrides:

            extra_overrides.update(budget_overrides)

        if feedback_overrides:

            extra_overrides.update(feedback_overrides)

        feedback_params = None

        if budget_mode.startswith("feedback"):

            feedback_params = _feedback_params(budget_mode, budget_cfg, per_ds, extra_overrides or None)

        result = dp_kmeans(

            Z_proc,

            init_centroids=init_centroids,

            k=k_val,

            T=T,

            eps_iter=eps_iter,

            clip_B=clip_B,

            rng=rng,

            budget_mode=budget_mode,

            feedback_params=feedback_params,

            proxy_points=proxy_points,

        )

    runtime_ms = (time.perf_counter() - start) * 1000.0



    labels = result["labels"]

    centroids = result["centroids"]

    stats = cluster_stats(labels, k_val)

    history = result["history"]

    collapse_flags = []

    for h in history:

        min_ratio = h.get("min_cluster_ratio")

        if min_ratio is None:

            min_ratio = 0.0

        collapse_flags.append((h.get("non_empty_k", k_val) < k_val) or (min_ratio < collapse_min_ratio))

    collapse_rate = float(np.mean(collapse_flags)) if collapse_flags else float("nan")

    metrics = {

        "sse_x": sse_in_X(Z, labels, k_val),

        "sse_z": sse_in_Z(Z_proc, labels, centroids),

        "ari": ari(y, labels),

        "nmi": nmi(y, labels),

        "f1": macro_f1(y, labels),

        "non_empty_k_final": stats["non_empty_k"],

        "max_cluster_ratio_final": stats["max_cluster_ratio"],

        "min_cluster_ratio_final": stats["min_cluster_ratio"],

        "cluster_entropy_final": stats["entropy"],

        "collapse_final": int(

            stats["non_empty_k"] < k_val or stats["max_cluster_ratio"] > collapse_threshold

        ),

        "collapse_rate": collapse_rate,

    }



    history_dir = os.path.join(out_dir, "history")

    ensure_dir(history_dir)

    tag = dataset_tag or dataset

    history_path = os.path.join(

        history_dir,

        f"{tag}_{method}_eps{eps_tot}_seed{seed}.csv",

    )

    pd.DataFrame(history).to_csv(history_path, index=False)



    out = {

        "dataset": dataset,

        "method": method,

        "eps_tot": eps_tot,

        "seed": seed,

        "k": k_val,

        "d": d_orig,

        "r": final_dim,

        "T": T,

        "n": Z_proc.shape[0],

        "eps_proxy": eps_proxy,

        "eps_proxy_ratio": (eps_proxy / eps_tot) if eps_tot > 0 else 0.0,

        "eps_iter": eps_iter,

        "runtime_ms_total": runtime_ms,

        "history_path": history_path,

    }

    out.update(metrics)

    return out





def run_method(

    dataset: str,

    method: str,

    eps_tot: float,

    seed: int,

    config: Dict[str, Any],

    out_dir: str,

    d_sub: Optional[int] = None,

    n_override: Optional[int] = None,

    d_override: Optional[int] = None,

    k_override: Optional[int] = None,

    proxy_eps_ratio: Optional[float] = None,

    proxy_eps: Optional[float] = None,

    dataset_overrides: Optional[Dict[str, Any]] = None,

    budget_overrides: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:

    dataset_cfg = dict(config["datasets"][dataset])

    if dataset_overrides:

        dataset_cfg.update(dataset_overrides)



    if dataset == "synthetic_stream":

        n = n_override if n_override is not None else dataset_cfg.get("n", 20000)

        d = d_override if d_override is not None else dataset_cfg.get("d", 512)

        k_val = k_override if k_override is not None else dataset_cfg.get("k", 10)

        batches = dataset_cfg.get("batches", 10)

        weights = dataset_cfg.get("cluster_weights")

        cluster_std = dataset_cfg.get("cluster_std", 1.0)

        X, y, batch_ids = make_synthetic_stream(

            n=n,

            d=d,

            k=k_val,

            seed=seed,

            batches=batches,

            weights=weights,

            cluster_std=cluster_std,

        )

        d_orig = X.shape[1]

        batch_metrics = []

        for b in range(int(batch_ids.max()) + 1):

            mask = batch_ids == b

            if not np.any(mask):

                continue

            res = _run_on_data(

                X[mask],

                y[mask],

                k_val,

                dataset=dataset,

                method=method,

                eps_tot=eps_tot,

                seed=seed,

                config=config,

                out_dir=out_dir,

                dataset_cfg=dataset_cfg,

                d_orig=d_orig,

                d_sub=d_sub,

                proxy_eps_ratio=proxy_eps_ratio,

                proxy_eps=proxy_eps,

                dataset_tag=f"{dataset}_b{b}",

                budget_overrides=budget_overrides,

            )

            batch_metrics.append(res)

        if not batch_metrics:

            raise ValueError("No batches produced for synthetic_stream.")

        df = pd.DataFrame(batch_metrics)

        out = batch_metrics[0].copy()

        out.update(

            {

                "f1": float(df["f1"].mean()),

                "sse_x": float(df["sse_x"].mean()),

                "runtime_ms_total": float(df["runtime_ms_total"].sum()),

                "stream_batches": int(df.shape[0]),

                "f1_std": float(df["f1"].std(ddof=0)),

                "sse_x_std": float(df["sse_x"].std(ddof=0)),

            }

        )

        return out



    X, y, k_val = _load_dataset(

        dataset,

        dataset_cfg,

        seed=seed,

        d_sub=d_sub,

        n_override=n_override,

        d_override=d_override,

        k_override=k_override,

    )

    d_orig = X.shape[1]

    return _run_on_data(

        X,

        y,

        k_val,

        dataset=dataset,

        method=method,

        eps_tot=eps_tot,

        seed=seed,

        config=config,

        out_dir=out_dir,

        dataset_cfg=dataset_cfg,

        d_orig=d_orig,

        d_sub=d_sub,

        proxy_eps_ratio=proxy_eps_ratio,

        proxy_eps=proxy_eps,

        budget_overrides=budget_overrides,

    )

