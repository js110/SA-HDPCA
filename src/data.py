import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def load_uci_har(root: str = "./UCI HAR Dataset") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI HAR dataset from the given root directory.
    Returns X in shape (10299, 561) and zero-based labels y in shape (10299,).
    """
    x_train_path = os.path.join(root, "train", "X_train.txt")
    y_train_path = os.path.join(root, "train", "y_train.txt")
    x_test_path = os.path.join(root, "test", "X_test.txt")
    y_test_path = os.path.join(root, "test", "y_test.txt")

    for path in [x_train_path, y_train_path, x_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing HAR file: {path}")

    x_train = pd.read_csv(x_train_path, sep=r"\s+", header=None).to_numpy(dtype=float)
    x_test = pd.read_csv(x_test_path, sep=r"\s+", header=None).to_numpy(dtype=float)
    y_train = pd.read_csv(y_train_path, header=None).iloc[:, 0].to_numpy(dtype=int)
    y_test = pd.read_csv(y_test_path, header=None).iloc[:, 0].to_numpy(dtype=int)

    X = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test]) - 1  # convert to 0..5

    expected_shape = (10299, 561)
    if X.shape != expected_shape:
        raise ValueError(f"HAR shape mismatch: expected {expected_shape}, got {X.shape}")
    if y.shape[0] != expected_shape[0]:
        raise ValueError(f"HAR labels mismatch: expected {expected_shape[0]}, got {y.shape[0]}")

    return X, y


def load_gas_sensor(root: str = "./gas") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the gas sensor dataset.
    Returns X (N, 128), y (N,), concentration (N,), batch_id (N,).
    """
    batch_files = sorted(glob.glob(os.path.join(root, "batch*.dat")))
    if not batch_files:
        raise FileNotFoundError(f"No batch*.dat files found under {root}")

    X_list, y_list, conc_list, batch_id_list = [], [], [], []
    for batch_idx, file_path in enumerate(batch_files, start=1):
        with open(file_path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                header = parts[0]
                if ";" not in header:
                    raise ValueError(f"Malformed header in {file_path}:{line_no} -> {header}")
                cls_str, conc_str = header.split(";")
                cls = int(cls_str)
                conc = float(conc_str)
                values = np.zeros(128, dtype=float)
                for token in parts[1:]:
                    if ":" not in token:
                        continue
                    idx_str, val_str = token.split(":")
                    idx = int(idx_str) - 1
                    if idx < 0 or idx >= 128:
                        raise ValueError(f"Index out of range in {file_path}:{line_no}: {idx}")
                    values[idx] = float(val_str)
                if values.shape[0] != 128:
                    raise ValueError(f"Unexpected feature length in {file_path}:{line_no}")
                X_list.append(values)
                y_list.append(cls - 1)
                conc_list.append(conc)
                batch_id_list.append(batch_idx)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=int)
    conc = np.asarray(conc_list, dtype=float)
    batch_ids = np.asarray(batch_id_list, dtype=int)

    if X.shape[1] != 128:
        raise ValueError(f"Gas sensor feature dimension mismatch: {X.shape}")

    return X, y, conc, batch_ids


def _normalize_weights(weights, k: int) -> np.ndarray:
    if weights is None:
        return np.ones(k, dtype=float) / float(k)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != k:
        raise ValueError(f"weights must have length k={k}, got {w.size}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights sum must be positive")
    return w / s


def make_synthetic(n: int, d: int, k: int, seed: int, weights=None, cluster_std=1.0):
    """
    Generate synthetic blobs for testing. Supports imbalanced clusters via weights.
    """
    rng = np.random.default_rng(seed)
    w = _normalize_weights(weights, k)
    counts = np.floor(w * n).astype(int)
    counts[-1] = max(counts[-1], 0) + (n - counts.sum())
    counts = np.clip(counts, 1, None)
    diff = n - counts.sum()
    counts[-1] = max(counts[-1] + diff, 1)
    centers = rng.uniform(-10.0, 10.0, size=(k, d))
    X, y = make_blobs(
        n_samples=counts.tolist(),
        centers=centers,
        n_features=d,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def make_synthetic_stream(
    n: int,
    d: int,
    k: int,
    seed: int,
    batches: int = 10,
    weights=None,
    cluster_std=1.0,
):
    """
    Generate synthetic blobs and split into batches to emulate streaming arrivals.
    Returns X, y, batch_ids.
    """
    X, y = make_synthetic(n=n, d=d, k=k, seed=seed, weights=weights, cluster_std=cluster_std)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]
    batches = max(int(batches), 1)
    batch_sizes = np.full(batches, n // batches, dtype=int)
    batch_sizes[: n % batches] += 1
    batch_ids = np.empty(n, dtype=int)
    start = 0
    for b, size in enumerate(batch_sizes):
        end = start + size
        batch_ids[start:end] = b
        start = end
    return X, y, batch_ids
