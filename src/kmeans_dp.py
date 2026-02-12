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

    clip_B: float,

    rng: np.random.Generator,

    budget_mode: str = "static",

    feedback_params: dict | None = None,

    proxy_points: np.ndarray | None = None,

    eps_schedule: list[float] | np.ndarray | None = None,

    collapse_boost: float = 1.5,

    collapse_threshold: float = 0.6,

    eps_cap: float | None = None,

):

    n, d = Z.shape

    centroids = init_centroids.astype(float).copy()

    history = []



    if eps_schedule is None:

        if budget_mode == "static":

            eps_schedule_arr = static_schedule(eps_iter, T)

            budget = None

        elif budget_mode == "feedback":

            fb_params = feedback_params or {}

            budget = FeedbackBudget(eps_iter=eps_iter, T=T, **fb_params)

            eps_schedule_arr = None

        elif budget_mode == "feedback_v2":

            fb_params = feedback_params or {}

            budget = FeedbackBudgetV2(eps_iter=eps_iter, T=T, **fb_params)

            eps_schedule_arr = None

        elif budget_mode == "feedback_v3":

            fb_params = feedback_params or {}

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



    for t in range(T):

        if remaining_eps <= 1e-12:

            break

        if eps_schedule is not None:

            planned_remaining = float(suffix_sum[t]) if suffix_sum is not None else 0.0

            scale = (remaining_eps / planned_remaining) if planned_remaining > 0 else 0.0

            eps_base = float(eps_schedule_arr[t])

            eps_raw = eps_base * scale

            eps_use = eps_raw

            adj = 1.0

        else:

            if budget_mode == "static":

                eps_use = float(eps_schedule_arr[t])

                eps_raw = eps_use

                eps_base = eps_use

                adj = 1.0

            else:

                eps_use, eps_raw = budget.next_eps()

                if budget_mode == "feedback":

                    eps_base = eps_raw

                    adj = 1.0

                else:

                    eps_base = budget.eps_base_trace[-1]

                    adj = budget.adj_trace[-1]



        labels = pairwise_distances_argmin(Z, centroids)

        counts = np.bincount(labels, minlength=k).astype(float)

        sums = np.zeros((k, d), dtype=float)

        for i in range(k):

            if counts[i] > 0:

                sums[i] = Z[labels == i].sum(axis=0)

        non_empty = int(np.count_nonzero(counts))

        max_ratio = float(np.max(counts) / n) if n > 0 else 0.0

        min_ratio = float(np.min(counts) / n) if n > 0 else 0.0



                                                                                         

        if eps_schedule is not None and (non_empty < k or max_ratio > collapse_threshold):

            eps_use = eps_use * collapse_boost

        if eps_cap is not None:

            eps_use = min(eps_use, eps_cap if eps_cap is not None else eps_use)

        eps_use = min(max(eps_use, 1e-8), remaining_eps)



        noisy_counts, _, centroids_new = privatize_clusters(

            counts=counts,

            sums=sums,

            eps_t=eps_use,

            clip_B=clip_B,

            rng=rng,

            proxy_points=proxy_points if proxy_points is not None else Z,

        )



        sse_z = float(np.sum((Z - centroids_new[labels]) ** 2))

        delta = centroids_new - centroids

        drift_raw = float(np.linalg.norm(delta))

        n_eff = float(np.sum(noisy_counts))

        if not np.isfinite(n_eff) or n_eff <= 0:

            n_eff = float(n)

        w = np.sqrt(noisy_counts / (n_eff + 1e-12))

        drift = float(np.linalg.norm(delta * w[:, None]))

        noise_scale_counts = 1.0 / max(eps_use, 1e-12)

        noise_scale_sums = (2.0 * clip_B) / max(eps_use, 1e-12)

        history.append(

            {

                "iter": t,

                "sse_z": sse_z,

                "drift": drift,

                "drift_raw": drift_raw,

                "eps_used": eps_use,

                "eps_raw": eps_raw,

                "eps_base": eps_base,

                "adj": adj,

                "non_empty_k": non_empty,

                "max_cluster_ratio": max_ratio,

                "min_cluster_ratio": min_ratio,

                "noise_scale_counts": noise_scale_counts,

                "noise_scale_sums": noise_scale_sums,

            }

        )



        centroids = centroids_new

        remaining_eps = max(remaining_eps - eps_use, 0.0)

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



    if budget_mode in {"feedback", "feedback_v2"} and eps_schedule is None:

        scaled_eps = budget.finalize()

        for idx, eps_val in enumerate(scaled_eps):

            history[idx]["eps_t"] = eps_val

    else:

        for idx, h in enumerate(history):

            h["eps_t"] = h["eps_used"]



    return {

        "labels": final_labels,

        "centroids": centroids,

        "history": history,

    }

