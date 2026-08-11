"""Dashboard shared utilities.

Consolidates reusable functions (plot styling, etc.) that were
previously duplicated across multiple page scripts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


DARK_THEME = {
    "axes.facecolor": "#0E1117",
    "figure.facecolor": "#0E1117",
    "axes.edgecolor": "#403E3E",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#404040",
    "legend.facecolor": "#1E1E1E",
    "legend.edgecolor": "gray",
}


def set_plot_style() -> None:
    """Set a consistent, dark-themed style for all matplotlib plots."""
    plt.style.use("dark_background")
    plt.rcParams.update(DARK_THEME)
