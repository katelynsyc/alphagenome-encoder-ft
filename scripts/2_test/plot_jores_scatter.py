#!/usr/bin/env python
"""Plot actual vs. predicted enrichment per condition for a Jores test-set evaluation.

Consumes the test_predictions.csv written by evaluate_jores.py (columns
"{condition}_true"/"{condition}_pred" for cold/dark/light/warm/maize) so the
model does not need to be re-run to make this plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Order must match evaluate_jores.py's CONDITION_NAMES / JoresMPRADataset._targets.
CONDITION_NAMES = ["cold", "dark", "light", "warm", "maize"]

# Fixed categorical order (skill-validated palette, slots 1-5), one hue per condition.
CONDITION_COLORS = {
    "cold": "#2a78d6",   # blue
    "dark": "#008300",   # green
    "light": "#e87ba4",  # magenta
    "warm": "#eda100",   # yellow
    "maize": "#1baf7a",  # aqua
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot actual vs. predicted test-set enrichment per Jores condition"
    )
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help='Prepended to the figure title, e.g. --model_name plantGREP -> "plantGREP -- Test Set: ..."',
    )
    parser.add_argument(
        "--condition_order",
        type=str,
        nargs="+",
        default=None,
        help="Left-to-right panel order, e.g. --condition_order light dark warm cold maize "
             "(default: %s)" % " ".join(CONDITION_NAMES),
    )
    return parser


def load_predictions(path: Path, condition_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    by_condition: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in condition_names:
        true_vals = np.array([float(row[f"{name}_true"]) for row in rows], dtype=np.float64)
        pred_vals = np.array([float(row[f"{name}_pred"]) for row in rows], dtype=np.float64)
        by_condition[name] = (true_vals, pred_vals)
    return by_condition


def _point_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gaussian-KDE density at each (x, y) point, for density-based scatter coloring."""
    if x.size < 3:
        return np.zeros(x.size)
    return gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))


def plot_actual_vs_predicted(
    by_condition: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    condition_names: list[str],
    model_name: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(condition_names), figsize=(4 * len(condition_names), 4), sharex=True, sharey=True)

    # Compute every condition's density up front so the color scale (vmin/vmax)
    # is shared across all five panels -- otherwise each panel auto-normalizes to
    # its own min/max and identical colors end up meaning different densities.
    densities = {name: _point_density(*by_condition[name]) for name in condition_names}
    all_density_values = np.concatenate([d for d in densities.values() if d.size])
    vmin = float(all_density_values.min()) if all_density_values.size else 0.0
    vmax = float(all_density_values.max()) if all_density_values.size else 1.0

    scatter = None
    for ax, name in zip(axes, condition_names):
        y_true, y_pred = by_condition[name]
        color = CONDITION_COLORS[name]

        # Sequential (magnitude) encoding for local point density -- distinct from the
        # per-condition categorical color, which stays on the fit line as identity.
        density = densities[name]
        order = np.argsort(density)  # low density first, so high-density points render on top
        scatter = ax.scatter(
            y_true[order], y_pred[order],
            c=density[order], cmap="viridis", vmin=vmin, vmax=vmax,
            s=10, alpha=0.7, edgecolors="none",
        )

        finite = np.concatenate([y_true, y_pred])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="black")

        m, b = np.polyfit(y_true, y_pred, 1)
        denom = np.std(y_true) * np.std(y_pred)
        r = float(np.corrcoef(y_true, y_pred)[0, 1]) if denom > 0 else float("nan")
        x_line = np.array([y_true.min(), y_true.max()])
        ax.plot(x_line, m * x_line + b, color=color, linewidth=1.5)

        ax.set_title(f"{name} (r={r:.3f})")
        ax.set_xlabel("Actual log2 enrichment")
        ax.set_ylabel("Predicted log2 enrichment")
        ax.annotate(f"n = {y_true.size}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)

    # One shared colorbar (same vmin/vmax as every panel above) instead of five
    # redundant, independently-scaled ones.
    fig.colorbar(scatter, ax=axes, label="density", fraction=0.02, pad=0.02)

    title = "Test Set: Actual vs. Predicted Enrichment by Condition"
    if model_name:
        title = f"{model_name} -- {title}"
    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv).resolve()
    if not predictions_path.exists():
        parser.error(f"Predictions CSV not found: {predictions_path}")

    default_output_path = predictions_path.parent / "y_vs_y_pred_by_condition.png"
    output_path = Path(args.output_path).resolve() if args.output_path is not None else default_output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    condition_names = args.condition_order or CONDITION_NAMES
    if sorted(condition_names) != sorted(CONDITION_NAMES):
        parser.error(f"--condition_order must be a permutation of {CONDITION_NAMES}, got {condition_names}")

    by_condition = load_predictions(predictions_path, condition_names)
    plot_actual_vs_predicted(by_condition, output_path, condition_names, model_name=args.model_name)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
