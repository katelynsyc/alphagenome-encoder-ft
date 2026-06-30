"""Visualization utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats


def plot_scatterplot(predictions: np.ndarray, targets: np.ndarray, split: str) -> None:
    """Plot predicted vs actual values per tissue with a regression line.

    Saves to results/plots/{split}predictvsactual.png.
    """
    tissues = ["Leaf", "MG", "Br", "RR"]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (tissue, color, ax) in enumerate(zip(tissues, colors, axes)):
        pred_i = predictions[:, i]
        tgt_i = targets[:, i]
        ax.scatter(pred_i, tgt_i, color=color, s=8, alpha=0.4)

        m, b, r_value, _, _ = stats.linregress(pred_i, tgt_i)
        x_line = np.array([pred_i.min(), pred_i.max()])
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1)

        ax.set_title(f"{tissue}  (r={r_value:.3f})")
        ax.set_xlabel("Predicted log2(RNA/DNA)")
        ax.set_ylabel("Actual log2(RNA/DNA)")
        ax.annotate(f"r² = {r_value**2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
        ax.annotate(f"y = {m:.2f}x + {b:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=9)

    fig.suptitle(f"{split} Predictions vs Actual", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/plots/{split}predictvsactual.png", dpi=300)
    plt.close(fig)
