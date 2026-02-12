from __future__ import annotations



import numpy as np

from sklearn.metrics import pairwise_distances

from sklearn.neighbors import NearestNeighbors



                                          

                                                                               

                                                                                 

                                                                                     



def _make_eps_candidates(Z: np.ndarray, rng: np.random.Generator, quantiles: list[float], sample_size: int = 2000) -> list[float]:

    n = Z.shape[0]

    if n <= 1:

        return [0.1]

    take = min(n, sample_size)

    idx = rng.choice(n, size=take, replace=False)

    subset = Z[idx]

    dists = pairwise_distances(subset, subset, metric="euclidean", n_jobs=1)

    triu = dists[np.triu_indices_from(dists, k=1)]

    cand = []

    for q in quantiles:

        cand.append(float(np.quantile(triu, q)))

                                

    cand = sorted(list({max(c, 1e-6) for c in cand}))

    return cand





def _label_clusters(Z: np.ndarray, eps_radius: float, min_pts: int, eps1: float, eps2: float, k_hint: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:

    n = Z.shape[0]

    labels = -np.ones(n, dtype=int)

    nbrs = NearestNeighbors(radius=eps_radius, algorithm="ball_tree").fit(Z)

    neighbor_indices = nbrs.radius_neighbors(return_distance=False)



    eps2_slice = eps2 / max(2 * k_hint, 1)

    eps2_slice = max(eps2_slice, 1e-8)



    visited = np.zeros(n, dtype=bool)

    cluster_id = 0



    for i in range(n):

        if visited[i]:

            continue

        visited[i] = True

        neighbors = neighbor_indices[i]

        noisy_density = len(neighbors) + rng.laplace(0.0, 1.0 / max(eps1, 1e-8))

        if noisy_density < min_pts:

            labels[i] = -1                

            continue



                           

        queue = [i]

        labels[i] = cluster_id

        while queue:

            p = queue.pop()

            neigh = neighbor_indices[p]

            for nb in neigh:

                if not visited[nb]:

                    visited[nb] = True

                    neigh_nb = neighbor_indices[nb]

                    noisy_density_nb = len(neigh_nb) + rng.laplace(0.0, 1.0 / eps2_slice)

                    if noisy_density_nb >= min_pts:

                        queue.extend(list(neigh_nb))

                if labels[nb] == -1:

                    labels[nb] = cluster_id

        cluster_id += 1



                                 

    if cluster_id == 0:

        labels[:] = 0

        cluster_id = 1

    return labels, cluster_id





def fit_predict(

    Z: np.ndarray,

    k: int,

    eps_tot: float,

    seed: int,

    min_pts: int = 10,

    eps_split: float = 0.5,

    eps_quantiles: list[float] | None = None,

    max_points: int | None = None,

) -> tuple[np.ndarray, np.ndarray, dict]:

    """
    Density-based DP clustering (DBDP).
    Returns labels, centroids (in Z space), extra_info.
    """

    rng = np.random.default_rng(seed)

    eps1 = eps_tot * eps_split

    eps2 = max(eps_tot - eps1, 1e-8)

    eps_quantiles = eps_quantiles or [0.01, 0.02, 0.05]



    Z_work = Z

    full_indices = np.arange(Z.shape[0])

    if max_points is not None and Z.shape[0] > max_points:

        idx = rng.choice(Z.shape[0], size=max_points, replace=False)

        Z_work = Z[idx]

        full_indices = idx



    eps_candidates = _make_eps_candidates(Z_work, rng, eps_quantiles)

    best = None

    best_score = None

    for eps_radius in eps_candidates:

        labels, n_clusters = _label_clusters(Z_work, eps_radius, min_pts, eps1, eps2, k_hint=k, rng=rng)

                                                                           

        counts = np.bincount(labels[labels >= 0], minlength=n_clusters)

        non_empty = int(np.count_nonzero(counts))

        if non_empty == 0:

            score = np.inf

        else:

            centroids = []

            sse = 0.0

            for cid in range(n_clusters):

                mask = labels == cid

                if not np.any(mask):

                    centroids.append(np.zeros(Z_work.shape[1], dtype=float))

                    continue

                cluster = Z_work[mask]

                mu = cluster.mean(axis=0)

                centroids.append(mu)

                diff = cluster - mu

                sse += float(np.sum(diff * diff))

            centroids_arr = np.vstack(centroids)

            score = (-non_empty, sse)

        if best_score is None or score < best_score:

            best_score = score

            best = (labels, n_clusters, eps_radius)



    labels_sub, n_clusters, eps_radius = best

    centroids = []

    for cid in range(n_clusters):

        mask = labels_sub == cid

        if not np.any(mask):

            centroids.append(np.zeros(Z_work.shape[1], dtype=float))

            continue

        centroids.append(Z_work[mask].mean(axis=0))

    centroids_arr = np.vstack(centroids)



                                              

    if Z_work.shape[0] != Z.shape[0]:

        full_labels = -np.ones(Z.shape[0], dtype=int)

        full_labels[full_indices] = labels_sub

        if n_clusters > 0:

            dists = pairwise_distances(Z, centroids_arr)

            nearest = np.argmin(dists, axis=1)

                          

            full_labels[full_labels < 0] = nearest[full_labels < 0]

    else:

        full_labels = labels_sub



    return full_labels, centroids_arr, {

        "eps_radius": eps_radius,

        "eps1": eps1,

        "eps2": eps2,

        "num_clusters": n_clusters,

        "used_points": int(Z_work.shape[0]),

    }

