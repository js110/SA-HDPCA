import numpy as np
from sklearn.metrics import pairwise_distances_argmin


def random_init(Z_tilde: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(Z_tilde.shape[0], size=k, replace=False)
    return Z_tilde[idx].copy()


def kmeanspp_init(Z_tilde: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples, _ = Z_tilde.shape
    centroids = np.empty((k, Z_tilde.shape[1]), dtype=float)
    first_idx = rng.integers(0, n_samples)
    centroids[0] = Z_tilde[first_idx]
    closest_dist_sq = np.sum((Z_tilde - centroids[0]) ** 2, axis=1)

    for i in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        next_idx = rng.choice(n_samples, p=probs)
        centroids[i] = Z_tilde[next_idx]
        new_dist_sq = np.sum((Z_tilde - centroids[i]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
    return centroids


def _proxy_sse(Z_tilde: np.ndarray, centroids: np.ndarray) -> float:
    labels = pairwise_distances_argmin(Z_tilde, centroids)
    diff = Z_tilde - centroids[labels]
    return float(np.sum(diff * diff))


def kmeanspp_rr_init(
    Z_tilde: np.ndarray,
    k: int,
    rng: np.random.Generator,
    restarts: int = 10,
) -> np.ndarray:
    """
    Random-restart k-means++ on proxy data, choose the best by proxy SSE.
    """
    restarts = max(int(restarts), 1)
    best = None
    best_sse = None
    for _ in range(restarts):
        centroids = kmeanspp_init(Z_tilde, k, rng)
        sse = _proxy_sse(Z_tilde, centroids)
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best = centroids
    return best.copy()


def fuzzy_av_init(
    Z_tilde: np.ndarray,
    k: int,
    rng: np.random.Generator,
    pop_size: int = 30,
    iters: int = 50,
    mf: float = 2.0,
) -> np.ndarray:
    """
    Simplified fuzzy initialization inspired by Fuzzy-AVOA++.
    Runs on a subsample (<=3000) of Z_tilde for speed.
    """
    if Z_tilde.shape[0] > 3000:
        idx = rng.choice(Z_tilde.shape[0], size=3000, replace=False)
        data = Z_tilde[idx]
    else:
        data = Z_tilde
    n, d = data.shape

    # More stable start: kmeans++ centroids then fuzzy memberships from distances.
    centroids = kmeanspp_init(data, k, rng)
    dist = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2) + 1e-8
    inv_dist = dist ** (-2 / (mf - 1))
    U = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    for _ in range(iters):
        um = U ** mf
        centroids = (um.T @ data) / (um.sum(axis=0)[:, None] + 1e-9)
        dist = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2) + 1e-8
        inv_dist = dist ** (-2 / (mf - 1))
        U = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    return centroids.copy()
