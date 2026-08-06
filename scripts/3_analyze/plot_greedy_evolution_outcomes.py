#!/usr/bin/env python
"""Report unique-sequence counts and plot per-condition outcome distributions for
ism_greedy_evolution.py's output.

For each CRE found in --input_dir (identified by its `<cre_id>_path###_summary.json`
files), this:
  1. Prints how many of the designed `final_sequence` values are actually unique
     (multiple greedy-evolution paths can converge on the same local optimum), plus
     the average number of rounds a path took to converge.
  2. Draws a box-violin plot per CRE of the final predicted enrichment across all
     five CONDITIONs, one point per *unique* final sequence (so a sequence several
     paths converged on isn't overrepresented). "warm" -- the condition the greedy
     search was actually optimizing toward target_warm -- is placed on the far left;
     the remaining conditions ("other_condition_weight" in ism_greedy_evolution.py's
     loss) follow in CONDITIONS order.
  3. Marks each box with a dotted reference line spanning just that box: target_warm
     for "warm", and the unmutated initial sequence's level for every other
     condition (the value the search was penalized for drifting away from).

One row of panels, one column per CRE.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

# Reuses this project's validated hex palette (see plot_greedy_evolution_history.py) --
# same categorical colors per condition, same secondary-ink gray for reference lines.
CONDITION_COLORS = {
    "cold": "#2a78d6",
    "dark": "#eb6834",
    "light": "#1baf7a",
    "warm": "#eda100",
    "maize": "#e87ba4",
}
CONDITIONS = list(CONDITION_COLORS)
DISPLAY_ORDER = ["warm"] + [c for c in CONDITIONS if c != "warm"]
TARGET_COLOR = "#52514e"

GENES = {
    "Zm-16206_fwd": "ZmCRN",
    "Zm-1631_rev": "Hsp9",
}


def find_cre_ids(input_dir: str) -> list[str]:
    pattern = re.compile(r"^(.+)_path\d+_summary\.json$")
    cre_ids = set()
    for path in glob.glob(os.path.join(input_dir, "*_path*_summary.json")):
        match = pattern.match(os.path.basename(path))
        if match:
            cre_ids.add(match.group(1))
    return sorted(cre_ids)


def load_path_summaries(input_dir: str, cre_id: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(input_dir, f"{cre_id}_path*_summary.json")))
    summaries = []
    for path in paths:
        with open(path) as handle:
            summaries.append(json.load(handle))
    return summaries


def summarize_deviations(cre_id: str, cond: str, ref_value: float, values: list[float]) -> dict:
    """Summarize how far each unique final sequence's predicted level for `cond`
    fell from `ref_value` (the dashed reference line: target_warm for "warm",
    the unmutated starting level for every other condition)."""
    deviations = [v - ref_value for v in values]
    abs_deviations = [abs(d) for d in deviations]
    return {
        "cre_id": cre_id,
        "condition": cond,
        "n": len(values),
        "reference": ref_value,
        "mean_deviation": statistics.mean(deviations),
        "std_deviation": statistics.stdev(deviations) if len(deviations) > 1 else 0.0,
        "median_deviation": statistics.median(deviations),
        "mean_abs_deviation": statistics.mean(abs_deviations),
        "min_deviation": min(deviations),
        "max_deviation": max(deviations),
    }


def format_deviation_report(stats_rows: list[dict]) -> str:
    lines = [
        "Deviation of final predicted enrichment from the dashed reference line",
        "(deviation = predicted value - reference; reference = target_warm for "
        "\"warm\", unmutated starting level for other conditions)",
        "",
    ]
    rows_by_cre: dict[str, list[dict]] = {}
    for row in stats_rows:
        rows_by_cre.setdefault(row["cre_id"], []).append(row)

    for cre_id, rows in rows_by_cre.items():
        lines.append(f"{GENES.get(cre_id, cre_id)} ({cre_id}):")
        for row in rows:
            lines.append(
                f"  {row['condition']:<6s} (n={row['n']}, ref={row['reference']:.3f}): "
                f"mean={row['mean_deviation']:+.3f}  std={row['std_deviation']:.3f}  "
                f"median={row['median_deviation']:+.3f}  mean_abs={row['mean_abs_deviation']:.3f}  "
                f"min={row['min_deviation']:+.3f}  max={row['max_deviation']:+.3f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def plot_outcomes(input_dir: str, output_path: str) -> None:
    cre_ids = find_cre_ids(input_dir)
    if not cre_ids:
        raise SystemExit(f"No <cre_id>_path###_summary.json files found under {input_dir}")

    fig, axes = plt.subplots(1, len(cre_ids), figsize=(6.5 * len(cre_ids), 5), squeeze=False)
    axes = axes[0]
    stats_rows = []

    for ax, cre_id in zip(axes, cre_ids):
        summaries = load_path_summaries(input_dir, cre_id)

        # Dedupe by final_sequence -- several paths can converge on the same local
        # optimum, and that sequence's predicted levels are deterministic, so keeping
        # every path would just overweight whichever optimum got rediscovered most.
        unique_by_sequence = {s["final_sequence"]: s["final_levels"] for s in summaries}
        n_total = len(summaries)
        n_unique = len(unique_by_sequence)
        avg_rounds = sum(s["n_rounds"] for s in summaries) / n_total
        print(f"{cre_id}: {n_unique} unique designed sequences out of {n_total} paths")
        print(f"{cre_id}: average {avg_rounds:.1f} rounds to design a sequence (n={n_total} paths)")

        target_warm = summaries[0]["target_warm"]
        initial_levels = summaries[0]["initial_levels"]

        for i, cond in enumerate(DISPLAY_ORDER):
            values = [levels[cond] for levels in unique_by_sequence.values()]

            parts = ax.violinplot([values], positions=[i], widths=0.8, showmedians=False, showextrema=False)
            parts["bodies"][0].set_facecolor(CONDITION_COLORS[cond])
            parts["bodies"][0].set_edgecolor(CONDITION_COLORS[cond])
            parts["bodies"][0].set_alpha(0.35)

            bp = ax.boxplot([values], positions=[i], widths=0.15, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor("white")
            bp["boxes"][0].set_edgecolor(CONDITION_COLORS[cond])
            bp["boxes"][0].set_linewidth(1.3)
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(CONDITION_COLORS[cond])
            bp["medians"][0].set_color(CONDITION_COLORS[cond])
            bp["medians"][0].set_linewidth(1.5)

            # warm's reference is the target it was optimized toward; every other
            # condition's reference is the unmutated sequence's level, since the loss
            # penalized this search for drifting away from it.
            ref_value = target_warm if cond == "warm" else initial_levels[cond]
            stats_rows.append(summarize_deviations(cre_id, cond, ref_value, values))
            ax.hlines(ref_value, i - 0.4, i + 0.4, color=TARGET_COLOR, linestyle=":", linewidth=1.8, zorder=4)
            ax.text(
                i + 0.42, ref_value, f"{ref_value:.2f}",
                color=TARGET_COLOR, fontsize=8, va="center", ha="left", clip_on=False,
            )

        ax.set_xticks(range(len(DISPLAY_ORDER)))
        ax.set_xticklabels(DISPLAY_ORDER)
        ax.set_title(GENES[cre_id], fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Predicted Enrichment")
    axes[-1].legend(
        handles=[
            plt.Line2D([0], [0], color=TARGET_COLOR, linestyle=":", linewidth=1.8,
                       label="warm: target  |  others: unmutated starting level"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=True, fontsize=9,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")

    report = format_deviation_report(stats_rows)
    print()
    print(report)
    stats_path = os.path.splitext(output_path)[0] + "_deviation_stats.txt"
    with open(stats_path, "w") as handle:
        handle.write(report)
    print(f"Saved {stats_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dir", nargs="?",
        default="results/e898939e/df4406c4716cd2cf/stage2/best_greedy_evolution",
        help="Directory containing <cre_id>_path###_summary.json files written by "
             "ism_greedy_evolution.py.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Defaults to final_expression_distribution.png inside input_dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output_path or os.path.join(args.input_dir, "final_expression_distribution.png")
    plot_outcomes(args.input_dir, output_path)


if __name__ == "__main__":
    main()
