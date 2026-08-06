#!/usr/bin/env python
"""Plot the warm-target-vs-preservation tradeoff traced out by
ism_greedy_evolution_weight_sweep.py's --other_condition_weights sweep.

For each sequence, reads every <seq_id>_w<weight>_summary.json in --input_dir and
plots two stacked panels against other_condition_weight (the x-axis, shared, since
it's the one knob being tuned -- see plot_hpsweep_val_pearson.py for the same
shared-x-axis-instead-of-dual-axis convention used elsewhere in this project):
  1. final warm level reached, with target_warm marked as a dashed reference line
     -- how much weight on preserving cold/dark/light/maize costs you in warm.
  2. total drift = sum of squared deviation of cold/dark/light/maize from their own
     round-0 baseline (the same quantity ism_greedy_evolution.py's loss penalizes)
     -- how much that weight actually buys you in preservation.
The weight actually used for a prior single-weight run (--highlight_weight, default
1.0, ism_greedy_evolution.py's own default) is marked with a vertical reference line
in both panels for context.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

CONDITIONS_OTHER = ["cold", "dark", "light", "maize"]
WARM_COLOR = "#eda100"     # same "warm" slot as plot_greedy_evolution_history.py
DRIFT_COLOR = "#2a78d6"    # categorical slot 1 -- drift is a different metric, own color
TARGET_COLOR = "#52514e"   # secondary-ink gray, reference line
HIGHLIGHT_COLOR = "#898781"  # muted gray, reference line for the highlighted weight


@dataclass
class SweepPoint:
    weight: float
    warm_final: float
    target_warm: float
    drift: float
    stop_reason: str
    n_rounds: int


def load_sweep(input_dir: str) -> dict[str, list[SweepPoint]]:
    points_by_id: dict[str, list[SweepPoint]] = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*_summary.json"))):
        with open(path) as handle:
            summary = json.load(handle)
        initial = summary["initial_levels"]
        final = summary["final_levels"]
        drift = sum((final[cond] - initial[cond]) ** 2 for cond in CONDITIONS_OTHER)
        point = SweepPoint(
            weight=summary["other_condition_weight"],
            warm_final=final["warm"],
            target_warm=summary["target_warm"],
            drift=drift,
            stop_reason=summary["stop_reason"],
            n_rounds=summary["n_rounds"],
        )
        points_by_id.setdefault(summary["id"], []).append(point)
    for points in points_by_id.values():
        points.sort(key=lambda p: p.weight)
    return points_by_id


def plot_tradeoff(points_by_id: dict[str, list[SweepPoint]], output_path: str, highlight_weight: float | None) -> None:
    seq_ids = list(points_by_id)
    fig, axes = plt.subplots(
        2, len(seq_ids), figsize=(6.5 * len(seq_ids), 8), squeeze=False, sharex="col",
    )

    for col, seq_id in enumerate(seq_ids):
        points = points_by_id[seq_id]
        weights = [p.weight for p in points]
        ax_warm, ax_drift = axes[0][col], axes[1][col]

        ax_warm.plot(weights, [p.warm_final for p in points], color=WARM_COLOR, marker="o", linewidth=2)
        ax_warm.axhline(
            points[0].target_warm, color=TARGET_COLOR, linestyle="--", linewidth=1.5,
            label=f"target_warm ({points[0].target_warm:g})",
        )
        ax_warm.set_title(seq_id, fontsize=12)
        ax_warm.legend(loc="best", fontsize=9, frameon=True)

        ax_drift.plot(weights, [p.drift for p in points], color=DRIFT_COLOR, marker="o", linewidth=2)
        ax_drift.set_xlabel("other_condition_weight")

        if highlight_weight is not None:
            for ax in (ax_warm, ax_drift):
                ax.axvline(highlight_weight, color=HIGHLIGHT_COLOR, linestyle=":", linewidth=1.5, zorder=1)

        for ax in (ax_warm, ax_drift):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0][0].set_ylabel("final warm level")
    axes[1][0].set_ylabel("total drift in\ncold/dark/light/maize\n(sum of squared deviation from baseline)")

    if highlight_weight is not None:
        fig.text(
            0.5, 0.995, f"dotted line: other_condition_weight={highlight_weight:g} (prior single-weight run)",
            ha="center", va="top", fontsize=9, color=HIGHLIGHT_COLOR,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    for seq_id, points in points_by_id.items():
        for p in points:
            print(f"  {seq_id} w={p.weight:g}: warm={p.warm_final:.3f} (target {p.target_warm:g}) "
                  f"drift={p.drift:.3f} stop={p.stop_reason} n_rounds={p.n_rounds}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dir", nargs="?",
        default="results/e898939e/df4406c4716cd2cf/stage2/best_greedy_evolution_weight_sweep",
        help="Directory of <seq_id>_w<weight>_summary.json files written by "
             "ism_greedy_evolution_weight_sweep.py.",
    )
    parser.add_argument("--output_path", type=str, default=None,
                         help="Defaults to weight_sweep_tradeoff.png inside input_dir.")
    parser.add_argument("--highlight_weight", type=float, default=1.0,
                         help="Marks this other_condition_weight with a vertical reference line "
                              "(default 1.0, ism_greedy_evolution.py's own default). Pass a value "
                              "outside the sweep, e.g. -1, to omit it.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    points_by_id = load_sweep(args.input_dir)
    if not points_by_id:
        raise SystemExit(f"No *_summary.json files found under {args.input_dir}")
    output_path = args.output_path or os.path.join(args.input_dir, "weight_sweep_tradeoff.png")
    highlight_weight = args.highlight_weight if args.highlight_weight >= 0 else None
    plot_tradeoff(points_by_id, output_path, highlight_weight)


if __name__ == "__main__":
    main()
