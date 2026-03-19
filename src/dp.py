import numpy as np


def add_noise_counts(
    counts: np.ndarray,
    eps_cnt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if eps_cnt <= 0:
        raise ValueError("eps_cnt must be positive for DP noise.")
    scale = 1.0 / eps_cnt
    noise = rng.laplace(0.0, scale, size=counts.shape)
    return counts.astype(float) + noise, scale


def add_noise_sums(
    sums: np.ndarray,
    eps_sum: float,
    clip_norm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if eps_sum <= 0:
        raise ValueError("eps_sum must be positive for DP noise.")
    d = sums.shape[1] if sums.ndim == 2 else 1
    scale = (2.0 * clip_norm * np.sqrt(float(d))) / eps_sum
    noise = rng.laplace(0.0, scale, size=sums.shape)
    return sums.astype(float) + noise, float(scale)


def privatize_clusters(
    counts: np.ndarray,
    sums: np.ndarray,
    eps_cnt: float,
    eps_sum: float,
    clip_norm: float,
    rng: np.random.Generator,
    proxy_points: np.ndarray,
) -> tuple:
    noisy_counts, count_scale = add_noise_counts(counts, eps_cnt, rng)
    noisy_sums, sum_scale = add_noise_sums(sums, eps_sum, clip_norm, rng)
    centroids = noisy_sums / np.maximum(noisy_counts[:, None], 1.0)

    empty = np.where(noisy_counts <= 0)[0]
    if empty.size > 0:
        if proxy_points is None or proxy_points.shape[0] == 0:
            raise ValueError("Proxy points required to reset empty clusters.")
        picks = rng.choice(proxy_points.shape[0], size=empty.size, replace=True)
        centroids[empty] = proxy_points[picks]
        noisy_counts[empty] = 1.0
        noisy_sums[empty] = centroids[empty]

    return noisy_counts, noisy_sums, centroids, count_scale, sum_scale
