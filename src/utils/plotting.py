import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# Project-wide plot style
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "success": "#10B981",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "mean_reverting": "#3B82F6",
    "random_walk": "#6B7280",
    "trending": "#EF4444",
}


def setup_style():
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("husl")


def save_fig(fig, path: str, tight: bool = True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")


def get_h_color(h: float) -> str:
    if h < 0.4:
        return COLORS["mean_reverting"]
    elif h > 0.6:
        return COLORS["trending"]
    return COLORS["random_walk"]


def add_h_regions(ax, alpha: float = 0.08):
    ylim = ax.get_ylim()
    ax.axhspan(ylim[0], ylim[0], alpha=0)  # dummy to set limits
    ax.axvspan(0, 0.5, alpha=alpha, color=COLORS["mean_reverting"], label="_nolegend_")
    ax.axvspan(0.5, 1.0, alpha=alpha, color=COLORS["trending"], label="_nolegend_")
    ax.axvline(0.5, color=COLORS["random_walk"], linestyle="--", alpha=0.5, linewidth=1)
