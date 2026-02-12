import numpy as np

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score





def sse_in_Z(Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:

    """SSE measured in the (possibly reduced) embedding space."""

    return float(np.sum((Z - centroids[labels]) ** 2))





def sse_in_X(X: np.ndarray, labels: np.ndarray, k: int) -> float:

    """
    SSE measured in the original preprocessed space X.
    Empty clusters are skipped to keep the metric stable when collapse happens.
    """

    if X.shape[0] != labels.shape[0]:

        raise ValueError("X and labels must have the same number of samples.")

    total = 0.0

    for c in range(k):

        mask = labels == c

        if not np.any(mask):

            continue

        cluster = X[mask]

        mu = np.mean(cluster, axis=0)

        diff = cluster - mu

        total += float(np.sum(diff * diff))

    return total





def cluster_stats(labels: np.ndarray, k: int) -> dict:

    """
    Diagnose collapse by counting non-empty clusters and dominance of the largest cluster.
    """

    counts = np.bincount(labels, minlength=k).astype(float)

    n = counts.sum()

    non_empty = int(np.count_nonzero(counts))

    max_ratio = float(np.max(counts) / n) if n > 0 else 0.0

    min_ratio = float(np.min(counts) / n) if n > 0 else 0.0

    probs = counts / n if n > 0 else counts

    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))) if n > 0 else 0.0

    return {

        "non_empty_k": non_empty,

        "max_cluster_ratio": max_ratio,

        "min_cluster_ratio": min_ratio,

        "entropy": entropy,

    }





def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return float(adjusted_rand_score(y_true, y_pred))





def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return float(normalized_mutual_info_score(y_true, y_pred))





def hungarian_match(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    """
    Map predicted cluster ids to true labels via Hungarian matching.
    """

    classes = np.unique(y_true)

    clusters = np.unique(y_pred)

    n_classes = classes.shape[0]

    n_clusters = clusters.shape[0]

    max_size = max(n_classes, n_clusters)

                                                             

    conf = np.zeros((max_size, max_size), dtype=int)

    for i, c_true in enumerate(classes):

        for j, c_pred in enumerate(clusters):

            conf[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))

    row_ind, col_ind = linear_sum_assignment(-conf)

    mapping = {}

    for i, j in zip(row_ind, col_ind):

        if i < n_classes and j < n_clusters:

            mapping[clusters[j]] = classes[i]

    mapped = np.array([mapping.get(c, c) for c in y_pred], dtype=int)

    return mapped





def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    aligned = hungarian_match(y_true, y_pred)

    return float(f1_score(y_true, aligned, average="macro"))

