"""
Comprehensive visualization of the synthetic fBM dataset.

Generates all exploratory plots for Part 1 of the project.
Run this after generating and preprocessing the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.plotting import setup_style, save_fig, COLORS, get_h_color, add_h_regions
from src.data.generate import load_raw_data
from src.data.preprocessing import rescale_per_sample, split_data, load_processed_data


def plot_sample_paths(X, y, H_values, save_dir="plots/data_exploration"):
    """Plot example fBM increment paths for different H values."""
    setup_style()

    h_showcase = [0.1, 0.25, 0.5, 0.75, 0.9]
    fig, axes = plt.subplots(len(h_showcase), 1, figsize=(12, 3 * len(h_showcase)),
                             sharex=True)

    for ax, h_target in zip(axes, h_showcase):
        # Find samples closest to target H
        mask = np.abs(y - h_target) < 0.02
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0][:5]

        for i, idx in enumerate(indices):
            color = get_h_color(y[idx])
            alpha = 0.8 if i == 0 else 0.4
            ax.plot(X[idx], color=color, alpha=alpha, linewidth=0.8)

        ax.set_ylabel("Increment")
        label = "Mean-reverting" if h_target < 0.5 else ("Trending" if h_target > 0.5 else "Random walk")
        ax.set_title(f"H = {h_target:.2f} ({label})", fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.suptitle("fBM Increment Paths for Different Hurst Exponents",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, f"{save_dir}/01_sample_paths.png")


def plot_cumulative_paths(X, y, save_dir="plots/data_exploration"):
    """Plot cumulative fBM paths (the actual fBM, not increments)."""
    setup_style()

    h_values_show = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, len(h_values_show), figsize=(4 * len(h_values_show), 4),
                             sharey=False)

    for ax, h_target in zip(axes, h_values_show):
        mask = np.abs(y - h_target) < 0.02
        indices = np.where(mask)[0][:8]
        color = get_h_color(h_target)

        for idx in indices:
            cumulative = np.cumsum(X[idx])
            ax.plot(cumulative, color=color, alpha=0.5, linewidth=0.7)

        ax.set_title(f"H = {h_target:.1f}", fontweight="bold")
        ax.set_xlabel("Time step")
        if ax == axes[0]:
            ax.set_ylabel("Cumulative value")

    fig.suptitle("Cumulative fBM Paths — From Mean-Reverting to Trending",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/02_cumulative_paths.png")


def plot_variance_vs_h(X, y, H_values, save_dir="plots/data_exploration"):
    """Show how variance of increments depends on H (motivates per-sample rescaling)."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-sample variance
    variances = X.var(axis=1)
    unique_H = np.unique(y)
    mean_var = [variances[y == h].mean() for h in unique_H]
    std_var = [variances[y == h].std() for h in unique_H]

    colors = [get_h_color(h) for h in unique_H]
    axes[0].scatter(unique_H, mean_var, c=colors, s=20, alpha=0.8)
    axes[0].fill_between(unique_H,
                         np.array(mean_var) - np.array(std_var),
                         np.array(mean_var) + np.array(std_var),
                         alpha=0.15, color=COLORS["primary"])
    axes[0].set_xlabel("H (true)")
    axes[0].set_ylabel("Mean sample variance")
    axes[0].set_title("Variance of fBM Increments vs H", fontweight="bold")
    axes[0].set_yscale("log")
    add_h_regions(axes[0])

    # After rescaling
    X_scaled, _, _ = rescale_per_sample(X)
    variances_scaled = X_scaled.var(axis=1)
    mean_var_scaled = [variances_scaled[y == h].mean() for h in unique_H]

    axes[1].scatter(unique_H, mean_var_scaled, c=colors, s=20, alpha=0.8)
    axes[1].set_xlabel("H (true)")
    axes[1].set_ylabel("Mean sample variance (after rescaling)")
    axes[1].set_title("After Per-Sample Standardization", fontweight="bold")
    axes[1].set_ylim(0.95, 1.05)
    add_h_regions(axes[1])

    fig.suptitle("Why Per-Sample Rescaling Matters",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/03_variance_vs_h.png")


def plot_autocorrelation_analysis(X, y, save_dir="plots/data_exploration"):
    """Show how autocorrelation structure changes with H."""
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    X_scaled, _, _ = rescale_per_sample(X)

    h_groups = [(0.1, 0.2, "H ~ 0.15 (mean-reverting)"),
                (0.45, 0.55, "H ~ 0.50 (random walk)"),
                (0.8, 0.9, "H ~ 0.85 (trending)")]

    max_lag = 30
    lags = np.arange(1, max_lag + 1)

    for ax, (h_lo, h_hi, title) in zip(axes, h_groups):
        mask = (y >= h_lo) & (y <= h_hi)
        X_subset = X_scaled[mask]

        # Compute autocorrelation for each sample
        autocorrs = np.zeros((X_subset.shape[0], max_lag))
        for i in range(X_subset.shape[0]):
            x = X_subset[i]
            for lag in range(1, max_lag + 1):
                autocorrs[i, lag - 1] = np.corrcoef(x[:-lag], x[lag:])[0, 1]

        mean_acf = autocorrs.mean(axis=0)
        std_acf = autocorrs.std(axis=0)

        h_mid = (h_lo + h_hi) / 2
        color = get_h_color(h_mid)

        ax.bar(lags, mean_acf, color=color, alpha=0.6, width=0.8)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.15, color=color)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0.5, max_lag + 0.5)

    fig.suptitle("Autocorrelation of fBM Increments — The Signal the Network Learns",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/04_autocorrelation.png")


def plot_h_distribution(y, save_dir="plots/data_exploration"):
    """Show the distribution of H values in the dataset."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(y, bins=100, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    axes[0].axvline(0.5, color=COLORS["danger"], linestyle="--", linewidth=1.5,
                    label="H = 0.5 (random walk)")
    axes[0].set_xlabel("H (true)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of H in Dataset", fontweight="bold")
    axes[0].legend()

    # Samples per H value
    unique_H, counts = np.unique(y, return_counts=True)
    colors = [get_h_color(h) for h in unique_H]
    axes[1].bar(unique_H, counts, width=0.008, color=colors, alpha=0.8)
    axes[1].set_xlabel("H (true)")
    axes[1].set_ylabel("Samples per H")
    axes[1].set_title("Samples per Hurst Value (Uniform by Design)", fontweight="bold")

    fig.suptitle("Dataset Composition",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/05_h_distribution.png")


def plot_rescaling_effect(X, y, save_dir="plots/data_exploration"):
    """Before vs after rescaling for a few samples."""
    setup_style()

    X_scaled, means, stds = rescale_per_sample(X)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    h_targets = [0.15, 0.5, 0.85]

    for col, h_target in enumerate(h_targets):
        mask = np.abs(y - h_target) < 0.02
        idx = np.where(mask)[0][0]
        color = get_h_color(h_target)

        # Before
        axes[0, col].plot(X[idx], color=color, linewidth=0.9)
        axes[0, col].set_title(f"Raw (H={h_target:.2f})", fontweight="bold")
        axes[0, col].axhline(0, color="black", linewidth=0.3, alpha=0.5)
        if col == 0:
            axes[0, col].set_ylabel("Raw increment value")

        # After
        axes[1, col].plot(X_scaled[idx], color=color, linewidth=0.9)
        axes[1, col].set_title(f"Rescaled (H={h_target:.2f})", fontweight="bold")
        axes[1, col].axhline(0, color="black", linewidth=0.3, alpha=0.5)
        if col == 0:
            axes[1, col].set_ylabel("Standardized value")
        axes[1, col].set_xlabel("Time step")

    fig.suptitle("Effect of Per-Sample Standardization on fBM Increments",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/06_rescaling_effect.png")


def plot_split_distribution(splits, save_dir="plots/data_exploration"):
    """Show H distribution across train/val/test splits."""
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    split_names = ["train", "val", "test"]
    split_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    for ax, name, color in zip(axes, split_names, split_colors):
        y_split = splits[f"y_{name}"]
        ax.hist(y_split, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.set_xlabel("H (true)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name.capitalize()} set (n={len(y_split):,})", fontweight="bold")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("H Distribution Across Train/Val/Test Splits (Uniform in Each)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/07_split_distributions.png")


def plot_covariance_heatmap(X, y, save_dir="plots/data_exploration"):
    """Show the empirical covariance matrix of increments for different H."""
    setup_style()

    h_targets = [0.15, 0.5, 0.85]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    X_scaled, _, _ = rescale_per_sample(X)

    for ax, h_target in zip(axes, h_targets):
        mask = np.abs(y - h_target) < 0.02
        X_subset = X_scaled[mask]

        cov_matrix = np.cov(X_subset.T)
        im = ax.imshow(cov_matrix[:30, :30], cmap="RdBu_r", aspect="equal",
                       vmin=-0.5, vmax=1.0)
        ax.set_title(f"H = {h_target:.2f}", fontweight="bold")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Time index")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Empirical Covariance Matrix of fBM Increments (First 30 Steps)",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/08_covariance_heatmaps.png")


def plot_spectral_analysis(X, y, save_dir="plots/data_exploration"):
    """Power spectral density for different H values."""
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    X_scaled, _, _ = rescale_per_sample(X)
    h_targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=0, vmax=1)

    for h_target in h_targets:
        mask = np.abs(y - h_target) < 0.02
        X_subset = X_scaled[mask]

        # Compute average power spectral density
        psds = np.abs(np.fft.rfft(X_subset, axis=1)) ** 2
        mean_psd = psds.mean(axis=0)
        freqs = np.fft.rfftfreq(X_subset.shape[1])

        color = cmap(norm(h_target))
        ax.loglog(freqs[1:], mean_psd[1:], color=color, linewidth=1.5,
                  label=f"H = {h_target:.1f}", alpha=0.8)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Power Spectrum of fBM Increments by Hurst Exponent",
                 fontweight="bold")
    ax.legend(title="H value")

    save_fig(fig, f"{save_dir}/09_spectral_analysis.png")


def plot_dataset_summary_grid(X, y, H_values, save_dir="plots/data_exploration"):
    """
    A single comprehensive summary figure with multiple panels.
    """
    setup_style()

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    X_scaled, _, _ = rescale_per_sample(X)

    # Panel 1: Sample paths (3 H values)
    for col, (h_target, label) in enumerate([
        (0.15, "Mean-reverting"), (0.5, "Random walk"), (0.85, "Trending")
    ]):
        ax = fig.add_subplot(gs[0, col])
        mask = np.abs(y - h_target) < 0.02
        indices = np.where(mask)[0][:5]
        color = get_h_color(h_target)
        for idx in indices:
            ax.plot(np.cumsum(X[idx]), color=color, alpha=0.5, linewidth=0.7)
        ax.set_title(f"H = {h_target:.2f} ({label})", fontweight="bold", fontsize=10)
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("Cumulative path")

    # Panel 2: Variance vs H
    ax = fig.add_subplot(gs[1, 0])
    variances = X.var(axis=1)
    unique_H = np.unique(y)
    mean_var = [variances[y == h].mean() for h in unique_H]
    colors = [get_h_color(h) for h in unique_H]
    ax.scatter(unique_H, mean_var, c=colors, s=15, alpha=0.8)
    ax.set_xlabel("H")
    ax.set_ylabel("Mean variance")
    ax.set_title("Variance vs H (log scale)", fontweight="bold", fontsize=10)
    ax.set_yscale("log")

    # Panel 3: Autocorrelation lag-1 vs H
    ax = fig.add_subplot(gs[1, 1])
    lag1_corrs = []
    for h in unique_H:
        mask = y == h
        subset = X_scaled[mask]
        corrs = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in subset[:50]]
        lag1_corrs.append(np.mean(corrs))
    ax.scatter(unique_H, lag1_corrs, c=colors, s=15, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("H")
    ax.set_ylabel("Lag-1 autocorrelation")
    ax.set_title("Lag-1 Autocorrelation vs H", fontweight="bold", fontsize=10)

    # Panel 4: H distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(y, bins=100, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax.set_xlabel("H")
    ax.set_ylabel("Count")
    ax.set_title("H Distribution (Uniform)", fontweight="bold", fontsize=10)

    # Panel 5-7: Covariance heatmaps
    for col, h_target in enumerate([0.15, 0.5, 0.85]):
        ax = fig.add_subplot(gs[2, col])
        mask = np.abs(y - h_target) < 0.02
        subset = X_scaled[mask]
        cov = np.cov(subset.T)
        im = ax.imshow(cov[:25, :25], cmap="RdBu_r", aspect="equal",
                       vmin=-0.5, vmax=1.0)
        ax.set_title(f"Cov matrix (H={h_target:.2f})", fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Synthetic fBM Dataset — Comprehensive Summary",
                 fontsize=16, fontweight="bold", y=1.01)
    save_fig(fig, f"{save_dir}/10_summary_grid.png")


def plot_rescaling_statistics(X, y, save_dir="plots/data_exploration"):
    """Distribution of per-sample means and stds before rescaling."""
    setup_style()

    means = X.mean(axis=1)
    stds = X.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Means vs H
    scatter = axes[0].scatter(y, means, c=y, cmap="coolwarm", s=3, alpha=0.3)
    axes[0].set_xlabel("H (true)")
    axes[0].set_ylabel("Sample mean")
    axes[0].set_title("Per-Sample Mean vs H", fontweight="bold")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(scatter, ax=axes[0], label="H")

    # Stds vs H
    scatter = axes[1].scatter(y, stds, c=y, cmap="coolwarm", s=3, alpha=0.3)
    axes[1].set_xlabel("H (true)")
    axes[1].set_ylabel("Sample std")
    axes[1].set_title("Per-Sample Std vs H (Network Could Cheat on This!)",
                      fontweight="bold")
    plt.colorbar(scatter, ax=axes[1], label="H")

    fig.suptitle("Why Rescaling is Necessary — Scale Leaks Information About H",
                 fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, f"{save_dir}/11_rescaling_statistics.png")


def plot_hurst_colormap_paths(X, y, save_dir="plots/data_exploration"):
    """All paths colored by their H value on a continuous colormap."""
    setup_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=0, vmax=1)

    # Plot a subset of paths
    indices = np.random.RandomState(42).choice(len(X), size=500, replace=False)
    indices = indices[np.argsort(y[indices])]

    for idx in indices:
        cumpath = np.cumsum(X[idx])
        ax.plot(cumpath, color=cmap(norm(y[idx])), alpha=0.15, linewidth=0.4)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Hurst Exponent (H)", fontsize=12)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative value")
    ax.set_title("500 fBM Paths Colored by Hurst Exponent",
                 fontsize=14, fontweight="bold")
    save_fig(fig, f"{save_dir}/12_colormap_paths.png")


def run_all_visualizations():
    """Generate all Part 1 visualizations."""
    print("=" * 60)
    print("GENERATING ALL DATA EXPLORATION VISUALIZATIONS")
    print("=" * 60)

    # Load data
    raw = load_raw_data()
    X, y, H_values = raw["X"], raw["y"], raw["H_values"]

    save_dir = "plots/data_exploration"

    print("\n[1/12] Sample paths...")
    plot_sample_paths(X, y, H_values, save_dir)

    print("[2/12] Cumulative paths...")
    plot_cumulative_paths(X, y, save_dir)

    print("[3/12] Variance vs H...")
    plot_variance_vs_h(X, y, H_values, save_dir)

    print("[4/12] Autocorrelation analysis...")
    plot_autocorrelation_analysis(X, y, save_dir)

    print("[5/12] H distribution...")
    plot_h_distribution(y, save_dir)

    print("[6/12] Rescaling effect...")
    plot_rescaling_effect(X, y, save_dir)

    print("[7/12] Split distributions...")
    X_scaled, _, _ = rescale_per_sample(X)
    splits = split_data(X_scaled, y)
    plot_split_distribution(splits, save_dir)

    print("[8/12] Covariance heatmaps...")
    plot_covariance_heatmap(X, y, save_dir)

    print("[9/12] Spectral analysis...")
    plot_spectral_analysis(X, y, save_dir)

    print("[10/12] Summary grid...")
    plot_dataset_summary_grid(X, y, H_values, save_dir)

    print("[11/12] Rescaling statistics...")
    plot_rescaling_statistics(X, y, save_dir)

    print("[12/12] Colormap paths...")
    plot_hurst_colormap_paths(X, y, save_dir)

    print("\n" + "=" * 60)
    print(f"ALL PLOTS SAVED TO: {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    run_all_visualizations()
