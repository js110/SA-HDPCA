import numpy as np

from sklearn.feature_selection import mutual_info_classif





def pus_scores(

    X: np.ndarray,

    y: np.ndarray,

    seed: int,

    alpha: float = 1.0,

    xi: float = 1e-6,

    delta: np.ndarray | None = None,

) -> np.ndarray:

    if delta is None:

        delta = np.ptp(X, axis=0)

    delta = np.asarray(delta, dtype=float)

    ig = mutual_info_classif(X, y, random_state=seed)

    denom = (np.power(delta, alpha) + xi)

    return ig / denom





def apply_pus(

    X: np.ndarray,

    y: np.ndarray,

    m: int,

    seed: int,

    alpha: float = 1.0,

    xi: float = 1e-6,

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Compute PUS scores and select top-m features.
    Returns (X_reduced, selected_indices, scores_sorted).
    """

    n_features = X.shape[1]

    if m >= n_features:

        indices = np.arange(n_features)

        scores = pus_scores(X, y, seed, alpha=alpha, xi=xi)

        return X, indices, scores



    scores = pus_scores(X, y, seed, alpha=alpha, xi=xi)

    indices = np.argsort(scores)[::-1][:m]

    X_reduced = X[:, indices]

    return X_reduced, indices, scores

