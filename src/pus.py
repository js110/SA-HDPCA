import numpy as np


def variance_scores(X: np.ndarray) -> np.ndarray:
    """
    Unsupervised per-feature utility proxy used in the paper revision.
    """
    return np.var(X, axis=0, ddof=0)


def variance_sensitivity(clip_B: float, n_rows: int) -> float:
    """
    Conservative replace-one sensitivity bound for empirical variance on [-clip_B, clip_B].
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    return 8.0 * (clip_B ** 2) / float(n_rows)


def pus_public_bounds(clip_B: float, tau: float = 1e-6) -> dict:
    """
    Public bounds used by the strict DP Top-m mechanism.
    """
    sens_cost = 2.0 * clip_B
    score_max = (clip_B ** 2) / (sens_cost + tau)
    return {
        "sens_cost": sens_cost,
        "score_max": score_max,
    }


def pus_scores(
    X: np.ndarray,
    clip_B: float,
    tau: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    PUS with variance utility and a public sensitivity-cost term.
    """
    bounds = pus_public_bounds(clip_B=clip_B, tau=tau)
    utility = variance_scores(X)
    scores = utility / (bounds["sens_cost"] + tau)
    scores = np.clip(scores, 0.0, bounds["score_max"])
    return scores, bounds


def dp_top_m_indices(
    scores: np.ndarray,
    m: int,
    eps: float,
    score_sensitivity: float,
    score_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strict DP Top-m via sequential report-noisy-max with equal budget split.
    Returns (selected_indices, clipped_scores).
    """
    n_features = scores.shape[0]
    if m <= 0:
        return np.array([], dtype=int), np.clip(scores, 0.0, score_max)
    if m >= n_features:
        return np.arange(n_features, dtype=int), np.clip(scores, 0.0, score_max)
    if eps <= 0:
        raise ValueError("eps must be positive for DP Top-m selection.")
    clipped = np.clip(scores, 0.0, score_max)
    remaining = list(range(n_features))
    chosen: list[int] = []
    eps_step = eps / float(m)
    noise_scale = 2.0 * score_sensitivity / max(eps_step, 1e-12)

    for _ in range(m):
        rem_idx = np.array(remaining, dtype=int)
        noisy = clipped[rem_idx] + rng.laplace(0.0, noise_scale, size=rem_idx.shape[0])
        pick_local = int(np.argmax(noisy))
        pick = int(rem_idx[pick_local])
        chosen.append(pick)
        remaining.remove(pick)

    return np.array(chosen, dtype=int), clipped


def apply_pus(
    X: np.ndarray,
    m: int,
    clip_B: float,
    rng: np.random.Generator,
    eps_fs: float,
    tau: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Strict DP Top-m feature selection used by SA-HDPCA.
    Returns (X_reduced, selected_indices, clipped_scores, metadata).
    """
    n_rows, n_features = X.shape
    if m >= n_features:
        scores, bounds = pus_scores(X, clip_B=clip_B, tau=tau)
        meta = {
            "score_sensitivity": variance_sensitivity(clip_B=clip_B, n_rows=n_rows) / (bounds["sens_cost"] + tau),
            "score_max": bounds["score_max"],
            "eps_fs": float(max(eps_fs, 0.0)),
        }
        return X, np.arange(n_features, dtype=int), scores, meta

    scores, bounds = pus_scores(X, clip_B=clip_B, tau=tau)
    score_sens = variance_sensitivity(clip_B=clip_B, n_rows=n_rows) / (bounds["sens_cost"] + tau)
    idx, clipped_scores = dp_top_m_indices(
        scores=scores,
        m=m,
        eps=eps_fs,
        score_sensitivity=score_sens,
        score_max=bounds["score_max"],
        rng=rng,
    )
    X_reduced = X[:, idx]
    meta = {
        "score_sensitivity": score_sens,
        "score_max": bounds["score_max"],
        "eps_fs": float(eps_fs),
    }
    return X_reduced, idx, clipped_scores, meta
