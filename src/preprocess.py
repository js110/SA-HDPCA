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

