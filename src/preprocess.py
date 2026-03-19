import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(X: np.ndarray, clip_B: float = 3.0) -> np.ndarray:
    """
    Standardize features then clip to [-clip_B, clip_B].
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_clipped = np.clip(X_scaled, -clip_B, clip_B)
    return X_clipped


def row_l2_clip(X: np.ndarray, clip_norm: float) -> np.ndarray:
    """
    Public row-wise L2 clipping used to obtain an explicit per-record norm bound.
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))
    return X * scale
