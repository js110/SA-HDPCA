from __future__ import annotations

import numpy as np

from src.init_av import kmeanspp_init
from src.kmeans_dp import dp_kmeans

# DPDP (arXiv 2304.13886v2) Algorithm 1 mapping:
# Line 3 DPClustering -> _base_dp_kmeans (DP k-means proxy)
# Line 5 TEVGraph -> _edge_weight/_merge_clusters (distance + mass proxy)
# Line 6 MergeCluster -> hierarchical merges in _merge_clusters
# Simplification: no ODE-based TEV search; use centroid distance heuristic.

def _base_dp_kmeans(Z: np.ndarray, k_base: int, eps_tot: float, T: int, seed: int, clip_B: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    init = kmeanspp_init(Z, k_base, rng)
    result = dp_kmeans(
        Z,
        init_centroids=init,
        k=k_base,
        T=T,
        eps_iter=eps_tot,
        clip_norm=clip_B,
        clip_B=clip_B,
        rng=rng,
        budget_mode="static",
        proxy_points=Z,
    )
    return result["labels"], result["centroids"]


def _edge_weight(c1: dict, c2: dict) -> float:
    dist = float(np.linalg.norm(c1["centroid"] - c2["centroid"]))
    mass = float(min(c1["weight"], c2["weight"]))
    return dist / (mass + 1e-6)


def _merge_clusters(Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray, k_target: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    counts = np.bincount(labels, minlength=centroids.shape[0]).astype(float)
    clusters = {
        cid: {"centroid": centroids[cid].copy(), "weight": counts[cid]}
        for cid in range(centroids.shape[0])
        if counts[cid] > 0
    }
    active = set(clusters.keys())
    current_labels = labels.copy()
    next_id = centroids.shape[0]
    merges: list[tuple[int, int]] = []

    while len(active) > k_target:
        best_pair = None
        best_w = np.inf
        ids = list(active)
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                i = ids[i_idx]
                j = ids[j_idx]
                w = _edge_weight(clusters[i], clusters[j])
                if w < best_w:
                    best_w = w
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        ci = clusters[i]
        cj = clusters[j]
        new_weight = ci["weight"] + cj["weight"]
        if new_weight > 0:
            new_centroid = (ci["centroid"] * ci["weight"] + cj["centroid"] * cj["weight"]) / new_weight
        else:
            new_centroid = (ci["centroid"] + cj["centroid"]) / 2.0
        new_id = next_id
        next_id += 1
        clusters[new_id] = {"centroid": new_centroid, "weight": new_weight}
        active.discard(i)
        active.discard(j)
        active.add(new_id)
        merges.append((i, j))
        del clusters[i]
        del clusters[j]
        current_labels[(current_labels == i) | (current_labels == j)] = new_id

    # relabel to contiguous ids
    final_ids = sorted(list(active))
    id_map = {old: new for new, old in enumerate(final_ids)}
    relabeled = np.vectorize(lambda x: id_map.get(x, -1))(current_labels)
    centroids_final = []
    for new_id in range(len(final_ids)):
        mask = relabeled == new_id
        if not np.any(mask):
            centroids_final.append(np.zeros(Z.shape[1], dtype=float))
            continue
        centroids_final.append(Z[mask].mean(axis=0))
    return relabeled.astype(int), np.vstack(centroids_final), merges


def fit_predict(
    Z: np.ndarray,
    k: int,
    eps_tot: float,
    seed: int,
    T: int = 20,
    k_base_scale: float = 1.5,
    clip_B: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    DPDP: post-processing (no extra privacy) on top of DP k-means.
    """
    k_base = max(k + 2, int(np.ceil(k_base_scale * k)))
    labels_base, centroids_base = _base_dp_kmeans(Z, k_base, eps_tot, T, seed, clip_B)
    labels_final, centroids_final, merges = _merge_clusters(Z, labels_base, centroids_base, k_target=k)
    return labels_final, centroids_final, {
        "k_base": k_base,
        "merges": merges,
        "labels_base": labels_base,
    }
