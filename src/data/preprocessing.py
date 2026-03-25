"""
Rescaling and train/val/test splitting for fBM data.

Key insight on rescaling:
- Each fBM sample is a time series of increments.
- We rescale EACH SAMPLE independently (per-sample standardization)
  so that the network learns from the shape/autocorrelation structure,
  not the raw scale which depends on H.
- This is critical: if you rescale globally, the network can cheat
  by using the variance to infer H (since Var(fBM increments) depends on H).
"""

import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from pathlib import Path


def rescale_per_sample(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize each time series independently (zero mean, unit variance).
    This removes scale information and forces the model to learn from
    the autocorrelation structure.

    Returns:
        X_scaled, means, stds
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # avoid division by zero
    X_scaled = (X - means) / stds
    return X_scaled, means.squeeze(), stds.squeeze()


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    ratios: list[float] = [0.333, 0.333, 0.334],
    seed: int = 42,
) -> dict:
    """
    Split into train/val/test with 1/3 each.

    Why 1/3 / 1/3 / 1/3 instead of 60/20/20?
    With synthetic data we can generate as much as we want, so there's no
    scarcity of training data. Having equal-sized val and test sets gives
    more reliable evaluation estimates. The validation set is large enough
    to detect overfitting early, and the test set is large enough for
    statistically significant performance metrics.
    """
    val_test_ratio = ratios[1] + ratios[2]
    test_ratio_of_remainder = ratios[2] / val_test_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=seed, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_of_remainder, random_state=seed
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def save_processed_data(splits: dict, save_dir: str = "data/processed"):
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(splits, path / "splits.joblib")

    # Also save as parquet for easy inspection
    for key in ["train", "val", "test"]:
        X_key, y_key = f"X_{key}", f"y_{key}"
        df = pd.DataFrame(splits[X_key])
        df["H_true"] = splits[y_key]
        df.to_parquet(path / f"{key}.parquet", index=False)

    print(f"Saved processed data to {path}")
    for key in ["train", "val", "test"]:
        print(f"  {key}: {splits[f'X_{key}'].shape[0]} samples")


def load_processed_data(save_dir: str = "data/processed") -> dict:
    path = Path(save_dir) / "splits.joblib"
    splits = joblib.load(path)
    print(f"Loaded processed data from {path}")
    for key in ["train", "val", "test"]:
        print(f"  {key}: {splits[f'X_{key}'].shape[0]} samples")
    return splits
