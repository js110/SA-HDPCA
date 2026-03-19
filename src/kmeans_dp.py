import numpy as np
from sklearn.metrics import pairwise_distances_argmin

from .budget import FeedbackBudget, FeedbackBudgetV2, FeedbackBudgetV3, static_schedule
from .dp import privatize_clusters


def dp_kmeans(
    Z: np.ndarray,
    init_centroids: np.ndarray,
    k: int,
    T: int,
    eps_iter: float,
    clip_norm: float | None,
    rng: np.random.Generator,
    budget_mode: str = "static",
    feedback_params: dict | None = None,
    proxy_points: np.ndarray | None = None,
    eps_schedule: list[float] | np.ndarray | None = None,
    collapse_boost: float = 1.5,
    collapse_threshold: float = 0.6,
    eps_cap: float | None = None,
    eps_cnt_ratio: float = 0.2,
    clip_B: float | None = None,
):
    if clip_norm is None:
        if clip_B is None:
            raise ValueError("clip_norm or clip_B must be provided.")
        clip_norm = float(clip_B)
    n, d = Z.shape
    centroids = init_centroids.astype(float).copy()
    history = []

    if eps_schedule is None:
        if budget_mode == "static":
            eps_schedule_arr = static_schedule(eps_iter, T)
            budget = None
        elif budget_mode == "feedback":
            fb_params_raw = feedback_params or {}
            allowed = {"warmup", "beta", "gamma", "drift_clip", "eps_min_ratio", "eps_max_ratio"}
            fb_params = {k: v for k, v in fb_params_raw.items() if k in allowed}
            budget = FeedbackBudget(eps_iter=eps_iter, T=T, **fb_params)
            eps_schedule_arr = None
        elif budget_mode == "feedback_v2":
            fb_params_raw = feedback_params or {}
            allowed = {
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
            }
            fb_params = {k: v for k, v in fb_params_raw.items() if k in allowed}
            budget = FeedbackBudgetV2(eps_iter=eps_iter, T=T, **fb_params)
            eps_schedule_arr = None
        elif budget_mode == "feedback_v3":
            fb_params_raw = feedback_params or {}
            allowed = {
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
            }
            fb_params = {k: v for k, v in fb_params_raw.items() if k in allowed}
            budget = FeedbackBudgetV3(eps_iter=eps_iter, T=T, **fb_params)
            eps_schedule_arr = None
        else:
            raise ValueError(f"Unknown budget mode: {budget_mode}")
    else:
        eps_schedule_arr = np.array(eps_schedule, dtype=float)
        if eps_schedule_arr.shape[0] != T:
            raise ValueError("eps_schedule length must equal T.")
        budget = None
    remaining_eps = float(eps_iter)
    if eps_schedule is not None:
        suffix_sum = np.cumsum(eps_schedule_arr[::-1])[::-1]
    else:
        suffix_sum = None

    use_guarded_feedback = False
    guard_warmup_static = 0
    guard_min_rel_improve = 0.02
    guard_min_non_empty_ratio = 0.7
    guard_max_cluster_ratio = collapse_threshold
    guard_raise_cap_ratio = 1.2
    if eps_schedule is None and budget_mode.startswith("feedback"):
        fp = feedback_params or {}
        use_guarded_feedback = bool(fp.get("guarded_feedback", False))
        guard_warmup_static = int(fp.get("guard_warmup_static", fp.get("warmup", 0)))
        guard_min_rel_improve = float(fp.get("guard_min_rel_improve", 0.02))
        guard_min_non_empty_ratio = float(fp.get("guard_min_non_empty_ratio", 0.7))
        guard_max_cluster_ratio = float(fp.get("guard_max_cluster_ratio", collapse_threshold))
        guard_raise_cap_ratio = float(fp.get("guard_raise_cap_ratio", 1.2))
    prev_prev_drift = None
    prev_drift = None
    prev_non_empty_ratio = None
    prev_max_ratio = None

    for t in range(T):
        if remaining_eps <= 1e-12:
            break
        remaining_iters = max(T - t, 1)
        eps_static_ref = remaining_eps / float(remaining_iters)
        guard_state = "none"
        eps_candidate = np.nan
        if eps_schedule is not None:
            planned_remaining = float(suffix_sum[t]) if suffix_sum is not None else 0.0
            scale = (remaining_eps / planned_remaining) if planned_remaining > 0 else 0.0
            eps_base = float(eps_schedule_arr[t])
            eps_raw = eps_base * scale
            eps_use = eps_raw
            eps_candidate = eps_raw
            adj = 1.0
            guard_state = "scheduled"
        else:
            if budget_mode == "static":
                eps_use = float(eps_schedule_arr[t])
                eps_raw = eps_use
                eps_base = eps_use
                eps_candidate = eps_use
                adj = 1.0
                guard_state = "static"
            else:
                eps_candidate, eps_raw = budget.next_eps()
                if budget_mode == "feedback":
                    eps_base = eps_raw
                    adj = 1.0
                else:
                    eps_base = budget.eps_base_trace[-1]
                    adj = budget.adj_trace[-1]
                eps_use = eps_candidate
                if use_guarded_feedback:
                    if t < guard_warmup_static:
                        eps_use = eps_static_ref
                        guard_state = "warmup_static"
                    else:
                        drift_improved = False
                        if prev_drift is not None:
                            if prev_prev_drift is None:
                                drift_improved = True
                            else:
                                drift_improved = prev_drift <= prev_prev_drift * (1.0 - guard_min_rel_improve)
                        collapse_risk = False
                        if prev_non_empty_ratio is not None and prev_max_ratio is not None:
                            collapse_risk = (
                                prev_non_empty_ratio < guard_min_non_empty_ratio
                                or prev_max_ratio > guard_max_cluster_ratio
                            )
                        allow_raise = drift_improved and (not collapse_risk)
                        if allow_raise and eps_candidate > eps_static_ref:
                            eps_use = min(eps_candidate, eps_static_ref * guard_raise_cap_ratio)
                            guard_state = "feedback_raise"
                        else:
                            eps_use = eps_static_ref
                            guard_state = "fallback_static"
                else:
                    guard_state = "feedback"

        labels = pairwise_distances_argmin(Z, centroids)
        counts = np.bincount(labels, minlength=k).astype(float)
        sums = np.zeros((k, d), dtype=float)
        for i in range(k):
            if counts[i] > 0:
                sums[i] = Z[labels == i].sum(axis=0)
        non_empty = int(np.count_nonzero(counts))
        max_ratio = float(np.max(counts) / n) if n > 0 else 0.0
        min_ratio = float(np.min(counts) / n) if n > 0 else 0.0

        # Collapse-aware boost when explicit schedule is used; keep total budget bounded.
        if eps_schedule is not None and (non_empty < k or max_ratio > collapse_threshold):
            eps_use = eps_use * collapse_boost
        if eps_cap is not None:
            eps_use = min(eps_use, eps_cap if eps_cap is not None else eps_use)
        eps_use = min(max(eps_use, 1e-8), remaining_eps)

        eps_cnt = max(eps_use * eps_cnt_ratio, 1e-8)
        eps_sum = max(eps_use - eps_cnt, 1e-8)
        if eps_cnt + eps_sum > eps_use:
            eps_sum = max(eps_use - eps_cnt, 1e-8)

        noisy_counts, _, centroids_new, count_scale, sum_scale = privatize_clusters(
            counts=counts,
            sums=sums,
            eps_cnt=eps_cnt,
            eps_sum=eps_sum,
            clip_norm=clip_norm,
            rng=rng,
            proxy_points=proxy_points if proxy_points is not None else Z,
        )

        sse_z_prev_labels = float(np.sum((Z - centroids_new[labels]) ** 2))
        labels_reassign = pairwise_distances_argmin(Z, centroids_new)
        sse_z_reassign = float(np.sum((Z - centroids_new[labels_reassign]) ** 2))
        sse_z = sse_z_prev_labels
        delta = centroids_new - centroids
        drift_raw = float(np.linalg.norm(delta))
        n_eff = float(np.sum(noisy_counts))
        if not np.isfinite(n_eff) or n_eff <= 0:
            n_eff = float(n)
        w = np.sqrt(noisy_counts / (n_eff + 1e-12))
        drift = float(np.linalg.norm(delta * w[:, None]))
        history.append(
            {
                "iter": t,
                "sse_z": sse_z,
                "sse_z_prev_labels": sse_z_prev_labels,
                "sse_z_reassign": sse_z_reassign,
                "drift": drift,
                "drift_raw": drift_raw,
                "eps_used": eps_use,
                "eps_candidate": eps_candidate,
                "eps_raw": eps_raw,
                "eps_base": eps_base,
                "adj": adj,
                "eps_static_ref": eps_static_ref,
                "guard_state": guard_state,
                "eps_cnt": eps_cnt,
                "eps_sum": eps_sum,
                "non_empty_k": non_empty,
                "max_cluster_ratio": max_ratio,
                "min_cluster_ratio": min_ratio,
                "noise_scale_counts": count_scale,
                "noise_scale_sums": sum_scale,
            }
        )

        centroids = centroids_new
        remaining_eps = max(remaining_eps - eps_use, 0.0)
        prev_prev_drift = prev_drift
        prev_drift = drift
        prev_non_empty_ratio = float(non_empty) / float(max(k, 1))
        prev_max_ratio = max_ratio
        if budget_mode in {"feedback", "feedback_v2", "feedback_v3"} and eps_schedule is None:
            centroid_norm = float(np.linalg.norm(centroids_new * w[:, None]))
            kwargs = {}
            if budget_mode in {"feedback_v2", "feedback_v3"}:
                kwargs = {
                    "non_empty_k": non_empty,
                    "max_cluster_ratio": max_ratio,
                    "k": k,
                }
            budget.register_drift(drift, centroid_norm, **kwargs)

    final_labels = pairwise_distances_argmin(Z, centroids)
    for h in history:
        # Keep schedule trace equal to actual used budget per iteration.
        h["eps_t"] = h["eps_used"]

    return {
        "labels": final_labels,
        "centroids": centroids,
        "history": history,
    }
