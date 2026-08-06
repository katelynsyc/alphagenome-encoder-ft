#!/usr/bin/env python
"""Grouped bar chart comparing per-condition test-set Pearson r between two
models' test_metrics.json files (the format evaluate_jores.py writes)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CONDITIONS = ["light", "dark", "warm", "cold", "maize"]
CONDITIONS_CAPS = ["Light", "Dark", "Warm", "Cold", "Dark"]

# Species grouping shown as a second label row below the x-axis: the first
# four conditions were measured in tobacco, the last (relabeled "Dark" above,
# it's the maize condition) in maize.
SPECIES_GROUPS = [(0, 3, "Tobacco"), (4, 4, "Maize")]

# Positional palette slots assigned to each series (in main()'s series order):
# AG stage-1 probing, AG stage-2 fine-tuned, plantGREP baseline -- matching the
# convention in plot_benchmark_results.py (AG Probing/Fine-tuned = pal[2]/pal[3],
# external baseline models = pal[9]).
SERIES_COLOR_SLOTS = [2, 3, 7]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-condition Pearson r between two models")
    parser.add_argument(
            "--alphagenome_stage_1", type=str,
            default="/grid/koo/home/kachu/projects/alphagenome-encoder-ft/results/e898939e/df4406c4716cd2cf/"
                "stage1/best_test_eval/test_metrics.json"
    )
    parser.add_argument(
        "--alphagenome_stage_2", type=str,
        default="/grid/koo/home/kachu/projects/alphagenome-encoder-ft/results/e898939e/df4406c4716cd2cf/"
                "stage2/best_test_eval/test_metrics.json",
    )
    parser.add_argument(
        "--plantgrep_metrics", type=str,
        default="/grid/koo/home/kachu/projects/plantGREP/data/results/plantGREP/test_metrics.json",
    )
    parser.add_argument("--alphagenome_stage1_label", type=str, default="AlphaGenome (Probing)")
    parser.add_argument("--alphagenome_stage2_label", type=str, default="AlphaGenome (Fine-tuned)")
    parser.add_argument("--plantgrep_label", type=str, default="plantGREP")
    parser.add_argument("--output_path", type=str, default="results/plots/pearson_comparison.png")
    parser.add_argument(
        "--alphagenome_category_metrics", type=str,
        default="/grid/koo/home/kachu/projects/alphagenome-encoder-ft/results/e898939e/df4406c4716cd2cf/"
                "stage2/best_category_eval/evaluation_metrics.json",
        help="evaluate_jores_categories.py output for the fine-tuned AlphaGenome model, used for the "
             "3 per-category (evolved_condition_specific/off_target_evolved/perturbed_library) bar charts.",
    )
    parser.add_argument(
        "--plantgrep_category_metrics", type=str,
        default="/grid/koo/home/kachu/projects/plantGREP/data/results/plantGREP_category_eval/"
                "evaluation_metrics.json",
    )
    parser.add_argument("--category_output_dir", type=str, default="results/plots")
    return parser


def load_pearsonr_by_condition(metrics_path: str) -> dict[str, float]:
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {condition: metrics["per_condition"][condition]["pearsonr"] for condition in CONDITIONS}


# Maps each of plot_jores_categories.py's first 2 figures to the key path evaluate_jores_
# categories.py nests its matching per-condition Pearson r under in evaluation_metrics.json.
# "perturbed_library" isn't here -- unlike these 2, it doesn't pool into one series per
# model (see plot_perturbation_type_pearson_comparison), since insertion vs. shuffling are
# mechanistically different edits worth comparing separately rather than combined.
CATEGORY_KEY_PATHS = {
    "evolved_condition_specific": ("evolution_on_target",),
    "off_target_evolved": ("evolution_off_target",),
}
CATEGORY_TITLES = {
    "evolved_condition_specific": "Evolved Condition-Specific Sequences",
    "off_target_evolved": "Off-Target Evolved Sequences",
    "perturbed_library": "Perturbed Library Sequences",
}

# "insertion" pools insertion_1/2/3 (evaluate_jores_categories.py computes this the same
# way it pools shuffling+insertion into "combined" -- pooled BEFORE computing Pearson r,
# not an average of the 3 insertion-count r's). Order controls left-to-right hatch order
# within each model's pair of bars in plot_perturbation_type_pearson_comparison.
PERTURBATION_TYPE_ORDER = ["insertion", "shuffling"]
PERTURBATION_TYPE_LABELS = {"insertion": "TFBS Insertion", "shuffling": "TFBS Shuffling"}
PERTURBATION_TYPE_HATCHES = {"insertion": None, "shuffling": "//"}

# off_target_evolved's rightmost (maize) bar tends to be the tallest of the 3 category
# charts, so its rotated value label runs into the default upper-right legend -- nudged
# up past the axes here. The other 2 categories' tallest bars sit further from the
# legend corner and don't need this.
CATEGORY_LEGEND_BBOX = {
    "off_target_evolved": (1.0, 1.12),
}

# Reuses the stage-2 fine-tuned / plantGREP slots from SERIES_COLOR_SLOTS (skipping the
# stage-1 probing slot) so these category bars stay the same color as their counterparts
# in the overall pearson_comparison.png chart.
CATEGORY_SERIES_COLOR_SLOTS = [SERIES_COLOR_SLOTS[1], SERIES_COLOR_SLOTS[2]]


def load_category_pearsonr_by_condition(metrics_path: str, key_path: tuple[str, ...]) -> dict[str, float]:
    """Like load_pearsonr_by_condition, but for one of evaluate_jores_categories.py's
    nested category blocks (see CATEGORY_KEY_PATHS) instead of the top-level "overall"
    per-condition Pearson r."""
    with open(metrics_path) as f:
        metrics = json.load(f)
    node = metrics
    for key in key_path:
        node = node[key]
    return {condition: node["per_condition"][condition]["pearsonr"] for condition in CONDITIONS}


def plot_category_pearson_comparisons(
    alphagenome_metrics_path: str,
    plantgrep_metrics_path: str,
    pal,
    output_dir: str,
    alphagenome_label: str = "AlphaGenome (Fine-tuned)",
    plantgrep_label: str = "plantGREP",
) -> None:
    """Builds the 3 per-category bar charts (one per CATEGORY_KEY_PATHS entry) comparing
    only AlphaGenome fine-tuning vs. plantGREP -- no stage-1 probing series, unlike the
    overall comparison in main() -- across all of CONDITIONS. `alphagenome_metrics_path`
    and `plantgrep_metrics_path` are each an evaluate_jores_categories.py
    evaluation_metrics.json (e.g. .../best_category_eval/evaluation_metrics.json)."""
    for category, key_path in CATEGORY_KEY_PATHS.items():
        series = {
            alphagenome_label: load_category_pearsonr_by_condition(alphagenome_metrics_path, key_path),
            plantgrep_label: load_category_pearsonr_by_condition(plantgrep_metrics_path, key_path),
        }
        output_path = Path(output_dir) / f"{category}_pearson_comparison.png"
        plot_pearson_comparison(
            series, pal, output_path=str(output_path),
            title=CATEGORY_TITLES[category], color_slots=CATEGORY_SERIES_COLOR_SLOTS,
            legend_bbox_to_anchor=CATEGORY_LEGEND_BBOX.get(category),
        )

    output_path = Path(output_dir) / "perturbed_library_pearson_comparison.png"
    plot_perturbation_type_pearson_comparison(
        alphagenome_metrics_path, plantgrep_metrics_path, pal, output_path=str(output_path),
        alphagenome_label=alphagenome_label, plantgrep_label=plantgrep_label,
    )


def plot_perturbation_type_pearson_comparison(
    alphagenome_metrics_path: str,
    plantgrep_metrics_path: str,
    pal,
    output_path: str,
    figsize=(9, 6),
    alphagenome_label: str = "AlphaGenome (Fine-tuned)",
    plantgrep_label: str = "plantGREP",
) -> None:
    """Perturbed Library chart, split into insertion vs. shuffling bars per model instead
    of plot_pearson_comparison's pooled "combined" series -- 4 bars per condition (model x
    perturbation type). Color encodes model (same palette slots as every other chart in
    this figure set); hatch encodes perturbation type, so the legend/color mapping a reader
    has already learned from pearson_comparison.png and the other 2 category charts still
    holds here. Bars are grouped by perturbation type first (both models' insertion bars,
    then both models' shuffling bars) rather than by model, so the two same-hatch bars in
    each condition's group of 4 sit next to each other instead of being split apart by the
    other model's opposite-hatch bar."""
    color_slots = CATEGORY_SERIES_COLOR_SLOTS
    models = [(alphagenome_label, pal[color_slots[0]], alphagenome_metrics_path),
              (plantgrep_label, pal[color_slots[1]], plantgrep_metrics_path)]

    # series[model_label][perturbation_type] = {condition: pearsonr}
    series = {
        label: {
            ptype: load_category_pearsonr_by_condition(metrics_path, ("perturbation", ptype))
            for ptype in PERTURBATION_TYPE_ORDER
        }
        for label, _color, metrics_path in models
    }

    x = np.arange(len(CONDITIONS))
    n_bars = len(models) * len(PERTURBATION_TYPE_ORDER)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    all_values = []
    legend_handles = []
    bar_i = 0
    for ptype in PERTURBATION_TYPE_ORDER:
        for label, color, _metrics_path in models:
            values = [series[label][ptype][c] for c in CONDITIONS]
            all_values.extend(values)
            offset = (bar_i - (n_bars - 1) / 2) * width
            hatch = PERTURBATION_TYPE_HATCHES[ptype]
            bars = ax.bar(
                x + offset, values, width, color=color, alpha=0.9,
                edgecolor="black", linewidth=1, hatch=hatch, zorder=3,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                    rotation=90, zorder=3,
                )
            legend_handles.append(mpatches.Patch(
                facecolor=color, edgecolor="black", hatch=hatch,
                label=f"{label} ({PERTURBATION_TYPE_LABELS[ptype]})",
            ))
            bar_i += 1

    ax.set_ylabel("Pearson's r", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS_CAPS)
    ax.set_ylim([min(0.5, min(all_values) - 0.05), max(all_values) + 0.15])
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles, loc="upper right", frameon=True, facecolor="white",
        edgecolor="none", framealpha=1.0, fontsize=9, ncol=1,
    )

    condition_label_fontsize = ax.get_xticklabels()[0].get_fontsize()
    trans = ax.get_xaxis_transform()
    line_y, text_y = -0.10, -0.155
    half_group_width = 0.8 / 2 + 0.05
    for start, end, species_label in SPECIES_GROUPS:
        x0 = x[start] - half_group_width
        x1 = x[end] + half_group_width
        ax.plot([x0, x1], [line_y, line_y], color="black", linewidth=1, transform=trans, clip_on=False)
        ax.text((x0 + x1) / 2, text_y, species_label, transform=trans, ha="center", va="top",
                fontsize=condition_label_fontsize)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.22)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def setup_plot_style():
    """Set up the plotting style and color palette."""
    sns.set(font_scale=1.2)
    sns.set_style("white")
    
    # Color palette (matching plot_cagi5_results.py)
    pal = ["#A65141", "#E7CDC2", "#80A0C7", "#394165","#B1934A", "#DCA258", "#100F14", "#8B9DAF", "#EEDA9D", "#E8DCCF"]
    
    return pal

def plot_pearson_comparison(series: dict[str, dict[str, float]], pal, output_path: str | None = None,
                             figsize=(9, 6), title: str = "Jores et al. 2026", color_slots=None,
                             legend_bbox_to_anchor=None):
    """series: {model_label: {condition: pearsonr}}, each covering all of CONDITIONS.

    Styled to match plot_lentimpra_benchmark/plot_starrseq_benchmark: palette
    colors, bold rotated value labels tinted to their bar, dashed y-grid only,
    no top/right spines, legend pinned just above the axes. `color_slots` overrides
    SERIES_COLOR_SLOTS (e.g. plot_category_pearson_comparisons skips the stage-1
    probing slot since it only ever plots 2 series). `legend_bbox_to_anchor`, in
    axes-fraction coordinates, nudges the legend past its default upper-right corner
    for a chart whose tallest bar's rotated value label would otherwise run into it
    (e.g. off_target_evolved's maize bar -- see CATEGORY_LEGEND_BBOX).
    """
    labels = list(series)
    n_models = len(labels)
    color_slots = color_slots if color_slots is not None else SERIES_COLOR_SLOTS
    model_colors = {label: pal[color_slots[i % len(color_slots)]] for i, label in enumerate(labels)}

    x = np.arange(len(CONDITIONS))
    width = 0.8 / n_models

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    all_values = []
    for i, label in enumerate(labels):
        values = [series[label][c] for c in CONDITIONS]
        all_values.extend(values)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=model_colors[label],
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            bar_color = bar.get_facecolor()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.0025,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=bar_color,
                rotation=90,
            )

    #ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Pearson's r", fontsize=12)
    #ax.set_title(title, fontsize=14)
    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS_CAPS)
    # Floor at 0.5 like the benchmark plots, but drop lower if a series dips
    # below that so no bar (e.g. a weak condition) gets clipped away.
    ax.set_ylim([min(0.5, min(all_values) - 0.05), max(all_values) + 0.15])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    ax.legend(loc="upper right", bbox_to_anchor=legend_bbox_to_anchor, frameon=False, fontsize=10, ncol=1)

    # Second label row below the condition ticks: a bracket line + species
    # name spanning the tobacco conditions, and another for the maize one.
    condition_label_fontsize = ax.get_xticklabels()[0].get_fontsize()
    trans = ax.get_xaxis_transform()
    line_y, text_y = -0.10, -0.155
    half_group_width = 0.8 / 2 + 0.05
    for start, end, species_label in SPECIES_GROUPS:
        x0 = x[start] - half_group_width
        x1 = x[end] + half_group_width
        ax.plot([x0, x1], [line_y, line_y], color="black", linewidth=1, transform=trans, clip_on=False)
        ax.text((x0 + x1) / 2, text_y, species_label, transform=trans, ha="center", va="top",
                fontsize=condition_label_fontsize)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, bottom=0.22)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    return fig


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    pal = setup_plot_style()
    series = {
        args.alphagenome_stage1_label: load_pearsonr_by_condition(args.alphagenome_stage_1),
        args.alphagenome_stage2_label: load_pearsonr_by_condition(args.alphagenome_stage_2),
        args.plantgrep_label: load_pearsonr_by_condition(args.plantgrep_metrics),
    }
    plot_pearson_comparison(series, pal, output_path=args.output_path)

    plot_category_pearson_comparisons(
        args.alphagenome_category_metrics,
        args.plantgrep_category_metrics,
        pal,
        args.category_output_dir,
        alphagenome_label=args.alphagenome_stage2_label,
        plantgrep_label=args.plantgrep_label,
    )


if __name__ == "__main__":
    main()
