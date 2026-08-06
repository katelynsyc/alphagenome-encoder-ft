#!/usr/bin/env python
"""Plot per-round enrichment trajectories from ism_greedy_evolution.py's output.

For every `<seq_id>_history.tsv` found in --input_dir, draws one panel with all
five CONDITION levels (cold, dark, light, warm, maize) as separate colored lines
against round index, plus:
  - a dashed horizontal line at that sequence's target_warm (from the matching
    `<seq_id>_summary.json`), so how far the run got is visible at a glance.
  - a vertical marker at target_reached_round if warm ever crossed the target
    (None means it never did -- e.g. because the run's stop_reason was
    "max_iterations" before getting there, not because no mutation could help;
    "converged" is the actual local-optimum stop, see ism_greedy_evolution.py's
    module docstring).

One column of panels, one row per sequence, sharing a single legend (the five
CONDITIONs plus the target_warm reference line).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import blended_transform_factory
import seaborn; seaborn.set_style('whitegrid')

# Fixed categorical order, slots 1-5 of this project's validated palette (see
# scripts/3_analyze/plot_hpsweep_val_pearson.py for the same "reuse the fixed
# palette, don't re-derive colors" convention). Order matches CONDITION_NAMES
# in saturation_mutagenesis.py.
CONDITION_COLORS = {
    "cold": "#2a78d6",
    "dark": "#eb6834",
    "light": "#1baf7a",
    "warm": "#eda100",
    "maize": "#e87ba4",
}
CONDITIONS = list(CONDITION_COLORS)
TARGET_COLOR = "#52514e"  # secondary-ink gray -- for the target-reached marker only,
                          # which isn't tied to any one condition's color
TARGET_ALPHA = 0.55  # dashed target lines: readable against the solid trajectory
                     # lines without competing with them

GENES = {
    "Zm-16206_fwd": "ZmCRN",
    "Zm-1631_rev": "Hsp9"
}


def read_history(history_path: str) -> list[dict]:
    with open(history_path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["round"] = int(row["round"])
        for cond in CONDITIONS:
            row[cond] = float(row[cond])
        row["loss"] = float(row["loss"])
    return rows


def find_sequences(input_dir: str) -> list[str]:
    history_paths = sorted(glob.glob(os.path.join(input_dir, "*_history.tsv")))
    return [os.path.basename(p)[: -len("_history.tsv")] for p in history_paths]


def gene_title(seq_id: str) -> str:
    """`seq_id`, optionally suffixed with `_path###` (multi-path runs), to a display title."""
    base, _, path_suffix = seq_id.partition("_path")
    gene = GENES.get(base, base)
    return f"{gene} ({seq_id[len(base) + 1:]})" if path_suffix else gene


def plot_evolution_history(input_dir: str, output_path: str) -> None:
    seq_ids = find_sequences(input_dir)
    if not seq_ids:
        raise SystemExit(f"No *_history.tsv files found under {input_dir}")

    fig, axes = plt.subplots(len(seq_ids), 1, figsize=(7.5, 4.8 * len(seq_ids)), squeeze=False)
    axes = axes[:, 0]

    trajectory_handles: dict[str, object] = {}
    target_handles: dict[str, object] = {}

    for ax, seq_id in zip(axes, seq_ids):
        history = read_history(os.path.join(input_dir, f"{seq_id}_history.tsv"))
        with open(os.path.join(input_dir, f"{seq_id}_summary.json")) as handle:
            summary = json.load(handle)

        rounds = [row["round"] for row in history]

        # Each condition's target line: for warm this is the actual optimization
        # target (summary["target_warm"]); for the other four, the loss function's
        # "keep near baseline" term makes their round-0 level the implicit target
        # (see ism_greedy_evolution.py's module docstring). Dashed and color-matched
        # to that condition's trajectory line so the pair reads as one unit.
        for cond in CONDITIONS:
            target_value = summary["target_warm"] if cond == "warm" else history[0][cond]
            target_handles[cond] = ax.axhline(
                target_value, color=CONDITION_COLORS[cond], linestyle="--",
                alpha=TARGET_ALPHA, linewidth=1.5, zorder=1,
            )

        for cond in CONDITIONS:
            trajectory_handles[cond], = ax.plot(
                rounds, [row[cond] for row in history],
                color=CONDITION_COLORS[cond],
                linewidth=2.5 if cond == "warm" else 1.5,
                marker="o", markersize=4,
                zorder=3 if cond == "warm" else 2,
            )

        # Sits right at the line's own right end (x=1.0 in axes-fraction, since axhline
        # spans the full axes width) and just above it, rather than floating in the
        # margin -- reads as a label on the line's tip, not a separate annotation.
        ax.text(
            1.0, summary["target_warm"], f"{summary['target_warm']:g}",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            color=CONDITION_COLORS["warm"], fontsize=9, va="bottom", ha="right", clip_on=False,
        )
        if summary["target_reached_round"] is not None:
            ax.axvline(summary["target_reached_round"], color=TARGET_COLOR, linestyle=":", linewidth=1, zorder=1)

        ax.set_title(
            f"{gene_title(seq_id)} ({summary['n_rounds']} rounds)",
            fontsize=11,
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Predicted Enrichment")
        ax.set_xticks(rounds[::5])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # One shared legend for the whole (now vertically stacked) figure, not just
    # whichever panel happens to be last -- every panel's handles are identical
    # (same 5 conditions), so axes[0]'s (captured above) are representative.
    # Each condition gets ONE row pairing its trajectory line with its target line
    # (via HandlerTuple drawing both handles on top of each other) rather than
    # a separate "target_<cond>" entry per condition -- five rows instead of ten.
    handles = [(trajectory_handles[cond], target_handles[cond]) for cond in CONDITIONS]
    fig.legend(
        handles, CONDITIONS, handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0, frameon=True, fontsize=9,
        title="solid = trajectory\ndashed = target", title_fontsize=8,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dir", nargs="?",
        default="results/e898939e/df4406c4716cd2cf/stage2/best_greedy_evolution",
        help="Directory containing <seq_id>_history.tsv / <seq_id>_summary.json pairs "
             "written by ism_greedy_evolution.py.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Defaults to enrichment_trajectories.png inside input_dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output_path or os.path.join(args.input_dir, "enrichment_trajectories.png")
    plot_evolution_history(args.input_dir, output_path)


if __name__ == "__main__":
    main()
