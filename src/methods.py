import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import load_gas_sensor, load_uci_har, make_synthetic, make_synthetic_stream
from .feedback import make_fixed_tail_schedule, make_strong_contrast_schedule
from .init_av import fuzzy_av_init, kmeanspp_init, kmeanspp_rr_init
from .kmeans_dp import dp_kmeans
from .metrics import ari, cluster_stats, macro_f1, nmi, sse_in_X, sse_in_Z
from .preprocess import preprocess, row_l2_clip
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
        k = k_override if k_override is not None else dataset_cfg["k"]
        if d_sub is not None:
            X = X[:, :d_sub]
    elif dataset == "gas":
        X, y, _, _ = load_gas_sensor(dataset_cfg["root"])
        k = k_override if k_override is not None else dataset_cfg["k"]
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
        m = min(int(dataset_cfg.get("pus_top_m", 128)), d)
        r = min(int(dataset_cfg.get("pca_r", 64)), d)
    elif dataset == "gas":
        m = min(int(dataset_cfg.get("pus_top_m", 64)), d)
        r = min(int(dataset_cfg.get("pca_r", 32)), d)
    else:
        m = min(int(dataset_cfg.get("pus_top_m", 128)), d)
        r = min(int(dataset_cfg.get("pca_r", 64)), d)
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
}

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

PROXY_INIT_METHODS = SAHDPCA_METHODS | {"kmeanspp_dp", "pca_dp"}
METHOD_CFG_KEYS = {
    "feature_mode",
    "use_pca",
    "init_mode",
    "budget_mode",
    "schedule_kind",
    "feedback_overrides",
}


def _stable_hash(payload: Dict[str, Any], length: int = 12) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _public_random_points(
    n_points: int,
    dim: int,
    clip_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pts = rng.normal(size=(n_points, dim))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-12)
    radii = rng.uniform(0.0, clip_norm, size=(n_points, 1))
    return pts * radii


def _default_init_mode(method: str) -> str:
    if method == "dp_kmeans":
        return "public_random"
    if method in {"kmeanspp_dp", "pca_dp", "sahdpca_wo_init"}:
        return "kpp"
    if method in SAHDPCA_METHODS:
        return "layered"
    return "public_random"


def _private_pca_transform(
    X: np.ndarray,
    r: int,
    eps_pca: float,
    clip_norm: float,
    seed: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    if r >= X.shape[1]:
        return row_l2_clip(X, clip_norm), {
            "eps_pca": 0.0,
            "pca_noise_scale": 0.0,
            "pca_sensitivity": 0.0,
        }
    if eps_pca <= 0:
        raise ValueError("eps_pca must be positive when private PCA is enabled.")
    sensitivity = 2.0 * clip_norm * np.sqrt(float(X.shape[1]))
    noise_scale = sensitivity / eps_pca
    X_private = X + rng.laplace(0.0, noise_scale, size=X.shape)
    pca = PCA(n_components=r, random_state=seed, svd_solver="randomized")
    pca.fit(X_private)
    X_proj = pca.transform(X)
    X_proj = row_l2_clip(X_proj, clip_norm)
    meta = {
        "eps_pca": float(eps_pca),
        "pca_noise_scale": float(noise_scale),
        "pca_sensitivity": float(sensitivity),
    }
    return X_proj, meta


def _apply_dimensionality(
    method: str,
    dataset: str,
    Z: np.ndarray,
    dataset_cfg: Dict[str, Any],
    seed: int,
    clip_B: float,
    clip_norm: float,
    rng: np.random.Generator,
    eps_topm: float,
    eps_pca: float,
    feature_mode: Optional[str] = None,
    use_pca: Optional[bool] = None,
) -> tuple[np.ndarray, dict]:
    m, r = _get_feature_params(dataset, dataset_cfg, Z.shape[1])
    if feature_mode is None:
        feature_mode = "pus" if method in PUS_METHODS else "none"
    if use_pca is None:
        use_pca = method in PCA_METHODS

    Z_proc = Z.copy()
    meta: Dict[str, Any] = {
        "feature_mode": feature_mode,
        "use_pca": bool(use_pca),
        "selected_dim": Z.shape[1],
        "eps_topm": float(eps_topm),
        "eps_pca": float(eps_pca),
        "score_sensitivity": 0.0,
        "score_max": 0.0,
        "selected_indices": None,
    }

    if feature_mode == "pus":
        Z_proc, idx, _, fs_meta = apply_pus(
            Z_proc,
            m=m,
            clip_B=clip_B,
            rng=rng,
            eps_fs=eps_topm,
        )
        meta["selected_indices"] = idx.tolist()
        meta["selected_dim"] = int(Z_proc.shape[1])
        meta["score_sensitivity"] = float(fs_meta["score_sensitivity"])
        meta["score_max"] = float(fs_meta["score_max"])
    elif feature_mode == "random":
        if m < Z_proc.shape[1]:
            idx = np.sort(rng.choice(Z_proc.shape[1], size=m, replace=False))
            Z_proc = Z_proc[:, idx]
            meta["selected_indices"] = idx.tolist()
            meta["selected_dim"] = int(Z_proc.shape[1])
    elif feature_mode != "none":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    Z_proc = row_l2_clip(Z_proc, clip_norm)
    if use_pca and r < Z_proc.shape[1]:
        Z_proc, pca_meta = _private_pca_transform(
            Z_proc,
            r=r,
            eps_pca=eps_pca,
            clip_norm=clip_norm,
            seed=seed,
            rng=rng,
        )
        meta.update(pca_meta)
        meta["selected_dim"] = int(Z_proc.shape[1])
    else:
        meta["pca_noise_scale"] = 0.0
        meta["pca_sensitivity"] = 0.0

    Z_proc = row_l2_clip(Z_proc, clip_norm)
    return Z_proc, meta


def _make_random_projection(
    q: int,
    q_prime: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q_prime_use = min(max(int(q_prime), 1), q)
    if q_prime_use == q:
        return np.eye(q, dtype=float)
    return rng.normal(0.0, 1.0 / np.sqrt(float(q_prime_use)), size=(q, q_prime_use))


def _make_dp_proxy(
    Z_proc: np.ndarray,
    q_prime: int,
    eps_proxy: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if eps_proxy <= 0:
        raise ValueError("eps_proxy must be positive for strict DP proxy construction.")
    R = _make_random_projection(Z_proc.shape[1], q_prime=q_prime, rng=rng)
    Y = Z_proc @ R
    op_norm = float(np.linalg.norm(R, ord=2))
    sensitivity = float(2.0 * clip_norm * op_norm * np.sqrt(float(R.shape[1])))
    noise_scale = sensitivity / eps_proxy
    Y_hat = Y + rng.laplace(0.0, noise_scale, size=Y.shape)
    R_pinv = np.linalg.pinv(R)
    meta = {
        "proxy_dim": int(R.shape[1]),
        "proxy_op_norm": op_norm,
        "proxy_sensitivity": sensitivity,
        "proxy_noise_scale": float(noise_scale),
    }
    return Y_hat, R_pinv, meta


def _lift_proxy_to_working(
    proxy_array: np.ndarray,
    R_pinv: np.ndarray,
    clip_norm: float,
) -> np.ndarray:
    working = proxy_array @ R_pinv
    return row_l2_clip(working, clip_norm)


def _init_centroids(
    method: str,
    proxy_data: Optional[np.ndarray],
    working_dim: int,
    k: int,
    clip_norm: float,
    rng: np.random.Generator,
    init_mode: Optional[str] = None,
    rr_restarts: int = 10,
    layered_cfg: Optional[Dict[str, Any]] = None,
    R_pinv: Optional[np.ndarray] = None,
) -> np.ndarray:
    mode = init_mode or _default_init_mode(method)
    if mode == "public_random":
        return _public_random_points(k, working_dim, clip_norm, rng)
    if proxy_data is None or R_pinv is None:
        raise ValueError("Proxy data and pseudo-inverse are required for strict proxy-based initialization.")

    def _proxy_stats(data: np.ndarray, centroids: np.ndarray) -> Dict[str, float]:
        labels = np.argmin(((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        diff = data - centroids[labels]
        sse = float(np.sum(diff * diff))
        counts = np.bincount(labels, minlength=k).astype(float)
        n = float(max(data.shape[0], 1))
        non_empty = int(np.count_nonzero(counts))
        min_ratio = float(np.min(counts) / n) if counts.size > 0 else 0.0
        max_ratio = float(np.max(counts) / n) if counts.size > 0 else 1.0
        return {
            "sse": sse,
            "non_empty": float(non_empty),
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
        }

    def _score(stats: Dict[str, float], cfg: Dict[str, Any]) -> float:
        min_ratio_thr = float(cfg.get("min_cluster_ratio", 1.0 / max(3 * k, 1)))
        max_ratio_thr = float(cfg.get("max_cluster_ratio", 0.75))
        penalty = 0.0
        penalty += float(max(0.0, min_ratio_thr - stats["min_ratio"])) * float(cfg.get("w_min_ratio", 2.0))
        penalty += float(max(0.0, stats["max_ratio"] - max_ratio_thr)) * float(cfg.get("w_max_ratio", 1.5))
        penalty += float(max(0.0, k - stats["non_empty"]) / max(k, 1)) * float(cfg.get("w_empty", 3.0))
        return stats["sse"] * (1.0 + penalty)

    if mode == "kpp":
        proxy_centroids = kmeanspp_init(proxy_data, k, rng)
    elif mode == "kpp_rr":
        proxy_centroids = kmeanspp_rr_init(proxy_data, k, rng, restarts=rr_restarts)
    elif mode == "layered":
        cfg = layered_cfg or {}
        light_rr = max(int(cfg.get("rr_restarts_light", rr_restarts)), 1)
        light = kmeanspp_rr_init(proxy_data, k, rng, restarts=light_rr)
        light_stats = _proxy_stats(proxy_data, light)
        collapse_like = (
            light_stats["non_empty"] < k
            or light_stats["min_ratio"] < float(cfg.get("min_cluster_ratio", 1.0 / max(3 * k, 1)))
            or light_stats["max_ratio"] > float(cfg.get("max_cluster_ratio", 0.75))
        )
        if (not bool(cfg.get("use_fuzzy_fallback", True))) or (not collapse_like):
            proxy_centroids = light
        else:
            heavy = fuzzy_av_init(proxy_data, k, rng)
            margin = float(cfg.get("switch_margin", 0.0))
            if _score(heavy_stats := _proxy_stats(proxy_data, heavy), cfg) + margin < _score(light_stats, cfg):
                proxy_centroids = heavy
            else:
                proxy_centroids = light
    else:
        raise ValueError(f"Unknown init_mode: {mode}")

    centroids = _lift_proxy_to_working(proxy_centroids, R_pinv=R_pinv, clip_norm=clip_norm)
    return centroids


def _feedback_params(
    budget_mode: str,
    budget_cfg: Dict[str, Any],
    per_ds: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mode_keys = {
        "feedback": [
            "warmup",
            "beta",
            "gamma",
            "drift_clip",
            "eps_min_ratio",
            "eps_max_ratio",
            "guarded_feedback",
            "guard_warmup_static",
            "guard_min_rel_improve",
            "guard_min_non_empty_ratio",
            "guard_max_cluster_ratio",
            "guard_raise_cap_ratio",
        ],
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
            "guarded_feedback",
            "guard_warmup_static",
            "guard_min_rel_improve",
            "guard_min_non_empty_ratio",
            "guard_max_cluster_ratio",
            "guard_raise_cap_ratio",
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
            "guarded_feedback",
            "guard_warmup_static",
            "guard_min_rel_improve",
            "guard_min_non_empty_ratio",
            "guard_max_cluster_ratio",
            "guard_raise_cap_ratio",
        ],
    }
    params: Dict[str, Any] = {}
    for key in mode_keys.get(budget_mode, []):
        if key in per_ds:
            params[key] = per_ds[key]
        elif key in budget_cfg:
            params[key] = budget_cfg[key]
    if overrides:
        params.update(overrides)
    return params


def _merge_nested(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_nested(out[key], value)
        else:
            out[key] = value
    return out


def _split_method_overrides(
    config: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    method: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw_sources = [
        config.get("method_overrides", {}).get(method, {}),
        dataset_cfg.get("method_overrides", {}).get(method, {}),
    ]
    method_cfg_overrides: Dict[str, Any] = {}
    dataset_param_overrides: Dict[str, Any] = {}
    for source in raw_sources:
        if not source:
            continue
        method_piece = {k: v for k, v in source.items() if k in METHOD_CFG_KEYS}
        dataset_piece = {k: v for k, v in source.items() if k not in METHOD_CFG_KEYS}
        method_cfg_overrides = _merge_nested(method_cfg_overrides, method_piece)
        dataset_param_overrides = _merge_nested(dataset_param_overrides, dataset_piece)
    return method_cfg_overrides, dataset_param_overrides


def _resolve_method_config(method: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    feature_mode = "pus" if method in PUS_METHODS else "none"
    use_pca = method in PCA_METHODS
    init_mode = None
    budget_mode = "static"
    schedule_kind = None
    feedback_overrides = None

    if method == "dp_kmeans":
        init_mode = "public_random"
    elif method in {"kmeanspp_dp", "pca_dp", "sahdpca_wo_init"}:
        init_mode = "kpp"
    elif method == "sahdpca_proxy_rr":
        init_mode = "kpp_rr"
    elif method == "sahdpca_proxy_kpp":
        init_mode = "kpp"
    else:
        init_mode = _default_init_mode(method)

    if method in {
        "sahdpca",
        "sahdpca_wo_pus",
        "sahdpca_wo_init",
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
    elif method == "sahdpca_wo_feedback":
        budget_mode = "static"
    elif method == "sahdpca_strong":
        budget_mode = "static"
        schedule_kind = "strong"

    if method == "sahdpca_rand_feat":
        feature_mode = "random"
        use_pca = False
    if method == "sahdpca_wo_pus":
        feature_mode = "none"
    if method == "sahdpca_pca_only":
        feature_mode = "none"
        use_pca = True

    resolved = {
        "feature_mode": feature_mode,
        "use_pca": use_pca,
        "init_mode": init_mode,
        "budget_mode": budget_mode,
        "schedule_kind": schedule_kind,
        "feedback_overrides": feedback_overrides,
    }
    if overrides:
        resolved = _merge_nested(resolved, overrides)
    return resolved


def _resolve_budget_split(
    eps_tot: float,
    config: Dict[str, Any],
    method_cfg: Dict[str, Any],
    proxy_ratio: Optional[float] = None,
    proxy_eps: Optional[float] = None,
    pre_pca_dim: Optional[int] = None,
    pca_dim: Optional[int] = None,
) -> Dict[str, float]:
    feature_mode = method_cfg["feature_mode"]
    use_pca = bool(method_cfg["use_pca"])
    effective_use_pca = use_pca
    if pre_pca_dim is not None and pca_dim is not None:
        effective_use_pca = use_pca and int(pca_dim) < int(pre_pca_dim)
    use_proxy = method_cfg["init_mode"] != "public_random"

    eps_dim_ratio = float(config.get("eps_fs_ratio", 0.05)) if (feature_mode == "pus" or effective_use_pca) else 0.0
    eps_topm_ratio = float(config.get("eps_topm_ratio", 0.6))
    eps_proxy_ratio = float(proxy_ratio if proxy_ratio is not None else config.get("proxy_eps_ratio", 0.10)) if use_proxy else 0.0

    eps_dim = eps_dim_ratio * eps_tot
    if proxy_eps is not None and use_proxy:
        eps_proxy_val = float(proxy_eps)
    else:
        eps_proxy_val = eps_proxy_ratio * eps_tot

    total_reserved = eps_dim + eps_proxy_val
    if total_reserved >= eps_tot:
        scale = 0.85 * eps_tot / max(total_reserved, 1e-12)
        eps_dim *= scale
        eps_proxy_val *= scale

    if feature_mode == "pus":
        eps_topm = eps_dim * eps_topm_ratio
        eps_pca = eps_dim - eps_topm if effective_use_pca else 0.0
    elif effective_use_pca:
        eps_topm = 0.0
        eps_pca = eps_dim
    else:
        eps_topm = 0.0
        eps_pca = 0.0

    eps_iter = eps_tot - eps_topm - eps_pca - eps_proxy_val
    if eps_iter <= 0:
        raise ValueError("Budget split leaves no iterative budget.")

    return {
        "eps_topm": float(max(eps_topm, 0.0)),
        "eps_pca": float(max(eps_pca, 0.0)),
        "eps_fs": float(max(eps_dim, 0.0)),
        "eps_proxy": float(max(eps_proxy_val, 0.0)),
        "eps_iter": float(eps_iter),
        "effective_use_pca": bool(effective_use_pca),
    }


def _maybe_subsample_proxy(
    proxy_data: np.ndarray,
    rng: np.random.Generator,
    frac: Optional[float],
    min_points: Optional[int],
) -> np.ndarray:
    if frac is None or frac <= 0 or frac >= 1:
        return proxy_data
    n = proxy_data.shape[0]
    target = int(n * frac)
    if min_points is not None:
        target = max(target, int(min_points))
    target = min(max(target, 1), n)
    if target >= n:
        return proxy_data
    idx = rng.choice(n, size=target, replace=False)
    return proxy_data[idx]


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
    dataset_tag: Optional[str] = None,
    budget_overrides: Optional[Dict[str, Any]] = None,
    proxy_eps_ratio: Optional[float] = None,
    proxy_eps: Optional[float] = None,
) -> Dict[str, Any]:
    rng = configure_environment(seed)
    T = int(config["T"])
    clip_B = float(config["clip_B"])
    clip_norm = float(config.get("clip_norm", clip_B))
    eps_cnt_ratio = float(config.get("eps_cnt_ratio", 0.2))
    Z = preprocess(X, clip_B=clip_B)

    method_cfg_overrides, dataset_param_overrides = _split_method_overrides(config, dataset_cfg, method)
    run_dataset_cfg = _merge_nested(dataset_cfg, dataset_param_overrides)
    method_cfg = _resolve_method_config(method, overrides=method_cfg_overrides)
    m, r = _get_feature_params(dataset, run_dataset_cfg, Z.shape[1])
    pre_pca_dim = Z.shape[1]
    if method_cfg["feature_mode"] in {"pus", "random"}:
        pre_pca_dim = min(m, Z.shape[1])

    proxy_ratio_eff = proxy_eps_ratio
    if proxy_ratio_eff is None and run_dataset_cfg.get("proxy_eps_ratio") is not None:
        proxy_ratio_eff = float(run_dataset_cfg["proxy_eps_ratio"])
    budget_split = _resolve_budget_split(
        eps_tot=eps_tot,
        config=config,
        method_cfg=method_cfg,
        proxy_ratio=proxy_ratio_eff,
        proxy_eps=proxy_eps,
        pre_pca_dim=pre_pca_dim,
        pca_dim=r,
    )

    Z_proc, dim_meta = _apply_dimensionality(
        method=method,
        dataset=dataset,
        Z=Z,
        dataset_cfg=run_dataset_cfg,
        seed=seed,
        clip_B=clip_B,
        clip_norm=clip_norm,
        rng=rng,
        eps_topm=budget_split["eps_topm"],
        eps_pca=budget_split["eps_pca"],
        feature_mode=method_cfg["feature_mode"],
        use_pca=budget_split["effective_use_pca"],
    )

    q_prime_default = min(int(config.get("proxy_dim", 16)), Z_proc.shape[1])
    q_prime = int(run_dataset_cfg.get("proxy_dim", q_prime_default))
    q_prime = max(min(q_prime, Z_proc.shape[1]), 1)
    proxy_init_frac = config.get("proxy_init_frac", None)
    proxy_init_min = config.get("proxy_init_min", None)
    rr_restarts = int(config.get("proxy_rr_restarts", 10))
    layered_cfg = dict(config.get("init_layered", {}))
    layered_cfg.update(run_dataset_cfg.get("init_layered", {}))

    proxy_meta = {
        "proxy_dim": 0,
        "proxy_op_norm": 0.0,
        "proxy_sensitivity": 0.0,
        "proxy_noise_scale": 0.0,
    }
    if method_cfg["init_mode"] == "public_random":
        init_centroids = _public_random_points(k_val, Z_proc.shape[1], clip_norm, rng)
        proxy_points = _public_random_points(max(128, 8 * k_val), Z_proc.shape[1], clip_norm, rng)
    else:
        proxy_data, R_pinv, proxy_meta = _make_dp_proxy(
            Z_proc,
            q_prime=q_prime,
            eps_proxy=budget_split["eps_proxy"],
            clip_norm=clip_norm,
            rng=rng,
        )
        proxy_data_init = _maybe_subsample_proxy(proxy_data, rng, proxy_init_frac, proxy_init_min)
        init_centroids = _init_centroids(
            method=method,
            proxy_data=proxy_data_init,
            working_dim=Z_proc.shape[1],
            k=k_val,
            clip_norm=clip_norm,
            rng=rng,
            init_mode=method_cfg["init_mode"],
            rr_restarts=rr_restarts,
            layered_cfg=layered_cfg,
            R_pinv=R_pinv,
        )
        proxy_points = _lift_proxy_to_working(proxy_data, R_pinv=R_pinv, clip_norm=clip_norm)

    budget_cfg = dict(config.get("budget", {}))
    if budget_overrides:
        budget_cfg.update(budget_overrides)
    per_ds = budget_cfg.get("datasets", {}).get(dataset, {})
    collapse_threshold = float(per_ds.get("collapse_threshold", budget_cfg.get("collapse_threshold", 0.55)))
    collapse_min_ratio = float(per_ds.get("collapse_min_ratio", config.get("collapse_min_ratio", 0.01)))

    extra_overrides: Dict[str, Any] = {}
    if budget_overrides:
        extra_overrides.update(budget_overrides)
    if method_cfg["feedback_overrides"]:
        extra_overrides.update(method_cfg["feedback_overrides"])
    feedback_params = None
    if method_cfg["budget_mode"].startswith("feedback"):
        feedback_params = _feedback_params(method_cfg["budget_mode"], budget_cfg, per_ds, extra_overrides or None)

    variant_payload = {
        "method": method,
        "dataset": dataset,
        "feature_mode": method_cfg["feature_mode"],
        "use_pca": bool(budget_split["effective_use_pca"]),
        "budget_mode": method_cfg["budget_mode"],
        "schedule_kind": method_cfg["schedule_kind"] or "none",
        "clip_norm": clip_norm,
        "eps_cnt_ratio": eps_cnt_ratio,
        "budget_split": budget_split,
        "proxy_dim": q_prime,
        "feedback_params": feedback_params or {},
    }
    config_hash = _stable_hash(variant_payload)
    algo_variant = (
        f"{method}|feature={method_cfg['feature_mode']}|pca={int(bool(budget_split['effective_use_pca']))}|"
        f"budget={method_cfg['budget_mode']}|proxy={q_prime}|strict=1"
    )

    start = time.perf_counter()
    if method_cfg["schedule_kind"] == "strong":
        eps_schedule = make_strong_contrast_schedule(
            budget_split["eps_iter"],
            T,
            low_frac=budget_cfg.get("strong_low_frac", 0.7),
            eps_min_ratio=budget_cfg.get("strong_eps_min_ratio", 0.3),
        )
        result = dp_kmeans(
            Z_proc,
            init_centroids=init_centroids,
            k=k_val,
            T=T,
            eps_iter=budget_split["eps_iter"],
            clip_norm=clip_norm,
            rng=rng,
            eps_schedule=eps_schedule,
            budget_mode="static",
            proxy_points=proxy_points,
            collapse_boost=float(per_ds.get("collapse_boost", budget_cfg.get("collapse_boost", 1.3))),
            collapse_threshold=collapse_threshold,
            eps_cap=per_ds.get("eps_cap", budget_cfg.get("eps_cap")),
            eps_cnt_ratio=eps_cnt_ratio,
        )
    else:
        result = dp_kmeans(
            Z_proc,
            init_centroids=init_centroids,
            k=k_val,
            T=T,
            eps_iter=budget_split["eps_iter"],
            clip_norm=clip_norm,
            rng=rng,
            budget_mode=method_cfg["budget_mode"],
            feedback_params=feedback_params,
            proxy_points=proxy_points,
            collapse_boost=float(per_ds.get("collapse_boost", budget_cfg.get("collapse_boost", 1.3))),
            collapse_threshold=collapse_threshold,
            eps_cap=per_ds.get("eps_cap", budget_cfg.get("eps_cap")),
            eps_cnt_ratio=eps_cnt_ratio,
        )
    runtime_ms = (time.perf_counter() - start) * 1000.0

    labels = result["labels"]
    centroids = result["centroids"]
    history = result["history"]
    stats = cluster_stats(labels, k_val)
    collapse_flags = []
    for h in history:
        min_ratio = h.get("min_cluster_ratio", 0.0)
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
        f"{tag}_{method}_eps{eps_tot}_seed{seed}_{config_hash[:8]}.csv",
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    out = {
        "dataset": dataset,
        "method": method,
        "eps_tot": eps_tot,
        "seed": seed,
        "k": k_val,
        "d": d_orig,
        "r": Z_proc.shape[1],
        "T": T,
        "n": Z_proc.shape[0],
        "eps_fs": budget_split["eps_fs"],
        "eps_topm": budget_split["eps_topm"],
        "eps_pca": budget_split["eps_pca"],
        "eps_proxy": budget_split["eps_proxy"],
        "eps_proxy_ratio": (budget_split["eps_proxy"] / eps_tot) if eps_tot > 0 else 0.0,
        "eps_iter": budget_split["eps_iter"],
        "clip_norm": clip_norm,
        "proxy_dim": proxy_meta["proxy_dim"],
        "algo_variant": algo_variant,
        "config_hash": config_hash,
        "runtime_ms_total": runtime_ms,
        "history_path": history_path,
        "score_sensitivity": dim_meta["score_sensitivity"],
        "score_max": dim_meta["score_max"],
        "proxy_sensitivity": proxy_meta["proxy_sensitivity"],
        "proxy_noise_scale": proxy_meta["proxy_noise_scale"],
        "pca_noise_scale": dim_meta.get("pca_noise_scale", 0.0),
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
                dataset_tag=f"{dataset}_b{b}",
                budget_overrides=budget_overrides,
                proxy_eps_ratio=proxy_eps_ratio,
                proxy_eps=proxy_eps,
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
        budget_overrides=budget_overrides,
        proxy_eps_ratio=proxy_eps_ratio,
        proxy_eps=proxy_eps,
    )
