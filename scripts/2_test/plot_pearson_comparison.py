#!/usr/bin/env python
"""Grouped bar chart comparing per-condition test-set Pearson r between two
models' test_metrics.json files (the format evaluate_jores.py writes)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set_style('whitegrid')

CONDITIONS = ["light", "dark", "warm", "cold", "maize"]

# First two categorical slots of the project's validated palette -- two series
# is always safe (see dataviz skill's color-formula check), no need to go further.
MODEL_COLORS = ["#36669c", "#3ec995"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-condition Pearson r between two models")
    parser.add_argument(
        "--alphagenome_metrics", type=str,
        default="/grid/koo/home/kachu/projects/alphagenome-encoder-ft/results/ray_tune/ag_hpsweep_1000/"
                "checkpoints/e898939e/df4406c4716cd2cf/stage2/best_test_eval/test_metrics.json",
    )
    parser.add_argument(
        "--plantgrep_metrics", type=str,
        default="/grid/koo/home/kachu/projects/plantGREP/data/results/plantGREP/test_metrics.json",
    )
    parser.add_argument("--alphagenome_label", type=str, default="AlphaGenome")
    parser.add_argument("--plantgrep_label", type=str, default="plantGREP")
    parser.add_argument("--output_path", type=str, default="results/plots/pearson_comparison.png")
    return parser


def load_pearsonr_by_condition(metrics_path: str) -> dict[str, float]:
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {condition: metrics["per_condition"][condition]["pearsonr"] for condition in CONDITIONS}


def plot_pearson_comparison(series: dict[str, dict[str, float]], output_path: str | None = None):
    """series: {model_label: {condition: pearsonr}}, each covering all of CONDITIONS.

    Bars start at 0 (never truncate a bar chart's baseline -- it exaggerates
    differences); direct value labels on each bar carry the precise comparison
    instead.
    """
    labels = list(series)
    n_models = len(labels)
    x = np.arange(len(CONDITIONS))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 6))
    all_values = []
    for i, label in enumerate(labels):
        values = [series[label][c] for c in CONDITIONS]
        all_values.extend(values)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label,
                       color=MODEL_COLORS[i % len(MODEL_COLORS)])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Pearson r")
    ax.set_title("Test-set Pearson r by Condition")
    ax.set_ylim(0, max(all_values) * 1.15)  # starts at 0 -- headroom above is just for the value labels
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    return fig


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    series = {
        args.alphagenome_label: load_pearsonr_by_condition(args.alphagenome_metrics),
        args.plantgrep_label: load_pearsonr_by_condition(args.plantgrep_metrics),
    }
    plot_pearson_comparison(series, output_path=args.output_path)


if __name__ == "__main__":
    main()
