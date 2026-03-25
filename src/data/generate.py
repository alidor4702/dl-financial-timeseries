"""
Generate synthetic fractional Brownian motion (fBM) time series
for Hurst exponent estimation.
"""

import numpy as np
from fbm import FBM
from tqdm import tqdm
import joblib
from pathlib import Path


def generate_fbm_dataset(
    H_min: float = 0.01,
    H_max: float = 0.99,
    n_H_values: int = 100,
    samples_per_H: int = 210,
    series_length: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate fBM time series for a grid of Hurst exponent values.

    Returns:
        X: array of shape (n_H_values * samples_per_H, series_length) — the fBM increments
        y: array of shape (n_H_values * samples_per_H,) — the true H values
        H_values: array of shape (n_H_values,) — the grid of H values used
    """
    np.random.seed(seed)

    H_values = np.linspace(H_min, H_max, n_H_values)
    total_samples = n_H_values * samples_per_H

    X = np.zeros((total_samples, series_length))
    y = np.zeros(total_samples)

    idx = 0
    for h in tqdm(H_values, desc="Generating fBM data"):
        for _ in range(samples_per_H):
            fbm_gen = FBM(n=series_length, hurst=h, method="daviesharte")
            # fbm_gen.fbm() returns length n+1, take increments (length n)
            path = fbm_gen.fbm()
            increments = np.diff(path)  # fBM increments, length = series_length
            X[idx] = increments
            y[idx] = h
            idx += 1

    return X, y, H_values


def save_raw_data(X, y, H_values, save_dir: str = "data/raw"):
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump({"X": X, "y": y, "H_values": H_values}, path / "fbm_dataset.joblib")
    print(f"Saved raw dataset to {path / 'fbm_dataset.joblib'}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  H range: [{H_values[0]:.4f}, {H_values[-1]:.4f}]")


def load_raw_data(save_dir: str = "data/raw") -> dict:
    path = Path(save_dir) / "fbm_dataset.joblib"
    data = joblib.load(path)
    print(f"Loaded raw dataset from {path}")
    print(f"  X shape: {data['X'].shape}, y shape: {data['y'].shape}")
    return data


if __name__ == "__main__":
    from src.utils.config import load_config

    cfg = load_config()["data"]
    X, y, H_values = generate_fbm_dataset(
        H_min=cfg["H_min"],
        H_max=cfg["H_max"],
        n_H_values=cfg["n_H_values"],
        samples_per_H=cfg["samples_per_H"],
        series_length=cfg["series_length"],
        seed=cfg["seed"],
    )
    save_raw_data(X, y, H_values)
