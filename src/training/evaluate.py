"""
Evaluation metrics and visualization for Hurst exponent estimation.
Computes bias, MAD, MAE, and generates comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.plotting import setup_style, save_fig, COLORS, get_h_color, add_h_regions


def compute_metrics_by_h(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute bias and MAD grouped by true H value."""
    unique_H = np.unique(y_true)
    bias = np.array([np.mean(y_pred[y_true == h] - h) for h in unique_H])
    mad = np.array([np.mean(np.abs(y_pred[y_true == h] - h)) for h in unique_H])
    std = np.array([np.std(y_pred[y_true == h] - h) for h in unique_H])

    overall_mae = np.mean(np.abs(y_pred - y_true))
    overall_bias = np.mean(y_pred - y_true)
    overall_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    return {
        "H_values": unique_H,
        "bias": bias,
        "mad": mad,
        "std": std,
        "overall_mae": overall_mae,
        "overall_bias": overall_bias,
        "overall_rmse": overall_rmse,
    }


def plot_bias(metrics: dict, title: str, save_path: str):
    """Plot average bias as a function of H_true."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    H = metrics["H_values"]
    colors = [get_h_color(h) for h in H]
    ax.scatter(H, metrics["bias"], c=colors, s=20, alpha=0.8, zorder=3)
    ax.fill_between(H, metrics["bias"] - metrics["std"], metrics["bias"] + metrics["std"],
                    alpha=0.15, color=COLORS["primary"])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    add_h_regions(ax)

    ax.set_xlabel("H (true)")
    ax.set_ylabel("Bias (H_predicted - H_true)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)

    overall = f"Overall bias: {metrics['overall_bias']:.4f}"
    ax.text(0.02, 0.95, overall, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    save_fig(fig, save_path)


def plot_mad(metrics: dict, title: str, save_path: str):
    """Plot mean absolute deviation as a function of H_true."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    H = metrics["H_values"]
    colors = [get_h_color(h) for h in H]
    ax.scatter(H, metrics["mad"], c=colors, s=20, alpha=0.8, zorder=3)
    add_h_regions(ax)

    ax.set_xlabel("H (true)")
    ax.set_ylabel("Mean Absolute Deviation")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    overall = f"Overall MAE: {metrics['overall_mae']:.4f}"
    ax.text(0.02, 0.95, overall, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    save_fig(fig, save_path)


def plot_predictions_scatter(y_true, y_pred, title, save_path):
    """Scatter plot of predicted vs true H."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [get_h_color(h) for h in y_true]
    ax.scatter(y_true, y_pred, c=colors, s=3, alpha=0.2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("H (true)")
    ax.set_ylabel("H (predicted)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend()

    save_fig(fig, save_path)


def plot_training_history(history: dict, title: str, save_path: str):
    """Plot training and validation loss curves."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train", color=COLORS["primary"], linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], label="Val", color=COLORS["danger"], linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Learning rate
    axes[1].plot(epochs, history["lr"], color=COLORS["secondary"], linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule", fontweight="bold")
    axes[1].set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, save_path)


def plot_comparison(all_metrics: dict, save_path: str):
    """
    Compare multiple methods (classical + NN) on bias and MAD.
    all_metrics: dict of {name: metrics_dict}
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    method_colors = {
        "R/S": "#9CA3AF",
        "DFA": "#6B7280",
        "Dense (small)": "#93C5FD",
        "Dense (medium)": "#2563EB",
        "Dense (large)": "#1E40AF",
    }

    for name, metrics in all_metrics.items():
        color = method_colors.get(name, COLORS["primary"])
        H = metrics["H_values"]

        # Bias
        axes[0].plot(H, metrics["bias"], label=name, color=color, linewidth=1.5, alpha=0.8)
        # MAD
        axes[1].plot(H, metrics["mad"], label=name, color=color, linewidth=1.5, alpha=0.8)

    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_xlabel("H (true)")
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Bias Comparison: Classical vs Neural Networks", fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("H (true)")
    axes[1].set_ylabel("Mean Absolute Deviation")
    axes[1].set_title("MAD Comparison: Classical vs Neural Networks", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(bottom=0)

    save_fig(fig, save_path)


def plot_comparison_summary_table(all_metrics: dict, save_path: str):
    """Create a summary bar chart comparing overall MAE across methods."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(all_metrics.keys())
    maes = [all_metrics[n]["overall_mae"] for n in names]
    rmses = [all_metrics[n]["overall_rmse"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, maes, width, label="MAE", color=COLORS["primary"], alpha=0.8)
    bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE", color=COLORS["secondary"], alpha=0.8)

    ax.set_ylabel("Error")
    ax.set_title("Overall Performance Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    save_fig(fig, save_path)


def plot_error_analysis(y_true, y_pred, title, save_path):
    """Detailed error analysis: error distribution, residuals, etc."""
    setup_style()
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # 1. Error distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(errors, bins=80, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Error (predicted - true)")
    ax1.set_ylabel("Count")
    ax1.set_title("Error Distribution", fontweight="bold")
    ax1.text(0.02, 0.95, f"Mean: {errors.mean():.4f}\nStd: {errors.std():.4f}",
             transform=ax1.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 2. Absolute error vs H
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_true, abs_errors, c=[get_h_color(h) for h in y_true], s=3, alpha=0.2)
    ax2.set_xlabel("H (true)")
    ax2.set_ylabel("|Error|")
    ax2.set_title("Absolute Error vs True H", fontweight="bold")

    # 3. Predicted vs True (density)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hexbin(y_true, y_pred, gridsize=50, cmap="Blues", mincnt=1)
    ax3.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax3.set_xlabel("H (true)")
    ax3.set_ylabel("H (predicted)")
    ax3.set_title("Prediction Density", fontweight="bold")
    ax3.set_aspect("equal")
    plt.colorbar(ax3.collections[0], ax=ax3, label="Count")

    # 4. Residual quantiles by H
    ax4 = fig.add_subplot(gs[1, 1])
    unique_H = np.unique(y_true)
    q25 = [np.percentile(errors[y_true == h], 25) for h in unique_H]
    q50 = [np.percentile(errors[y_true == h], 50) for h in unique_H]
    q75 = [np.percentile(errors[y_true == h], 75) for h in unique_H]
    ax4.fill_between(unique_H, q25, q75, alpha=0.2, color=COLORS["primary"], label="IQR (25-75%)")
    ax4.plot(unique_H, q50, color=COLORS["primary"], linewidth=1.5, label="Median error")
    ax4.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.set_xlabel("H (true)")
    ax4.set_ylabel("Error")
    ax4.set_title("Error Quantiles by H", fontweight="bold")
    ax4.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, save_path)


def plot_uncertainty_analysis(y_true, y_pred_mean, y_pred_std, title, save_path):
    """Visualize MC Dropout uncertainty estimates."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Uncertainty vs H
    axes[0].scatter(y_true, y_pred_std, c=[get_h_color(h) for h in y_true], s=3, alpha=0.3)
    axes[0].set_xlabel("H (true)")
    axes[0].set_ylabel("Prediction Std (uncertainty)")
    axes[0].set_title("Uncertainty vs True H", fontweight="bold")

    # 2. Uncertainty vs absolute error
    abs_err = np.abs(y_pred_mean - y_true)
    axes[1].scatter(y_pred_std, abs_err, c=[get_h_color(h) for h in y_true], s=3, alpha=0.3)
    axes[1].set_xlabel("Prediction Std (uncertainty)")
    axes[1].set_ylabel("|Error|")
    axes[1].set_title("Is Uncertainty Calibrated?", fontweight="bold")
    # Perfect calibration: high uncertainty ↔ high error
    corr = np.corrcoef(y_pred_std, abs_err)[0, 1]
    axes[1].text(0.02, 0.95, f"Correlation: {corr:.3f}",
                 transform=axes[1].transAxes, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 3. Predictions with error bars (subset)
    idx = np.random.RandomState(42).choice(len(y_true), size=200, replace=False)
    idx = idx[np.argsort(y_true[idx])]
    axes[2].errorbar(y_true[idx], y_pred_mean[idx], yerr=2 * y_pred_std[idx],
                     fmt="none", ecolor="lightblue", alpha=0.5, elinewidth=0.5)
    axes[2].scatter(y_true[idx], y_pred_mean[idx], c=[get_h_color(h) for h in y_true[idx]],
                    s=10, zorder=3)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    axes[2].set_xlabel("H (true)")
    axes[2].set_ylabel("H (predicted)")
    axes[2].set_title("Predictions with 2-sigma Intervals", fontweight="bold")
    axes[2].set_aspect("equal")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, save_path)
