#!/usr/bin/env python
"""Plot the distribution of per-round candidate-mutation losses recorded by
ism_round_candidate_losses.py.

For every `<seq_id>_round_candidate_losses.tsv` found in --input_dir, draws one
panel with a violin per round showing the loss of every (position, base) candidate
saturation_mutagenesis tried that round, plus a marker on the single candidate
evolve_sequence actually accepted (the round's minimum) -- so it's visible whether
the accepted move was a clear outlier or barely better than the rest of the
distribution.

One row of panels, one column per sequence found in --input_dir.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')

# Reuses this project's validated hex palette (see plot_greedy_evolution_history.py) --
# "warm" gold for the accepted candidate (the one move that gets taken), a blue for
# the bulk distribution it was drawn from, and the same secondary-ink gray for the
# median reference line.
DISTRIBUTION_COLOR = "#2a78d6"
ACCEPTED_COLOR = "#eda100"
MEDIAN_COLOR = "#52514e"


def read_round_losses(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["round"] = int(row["round"])
        row["position"] = int(row["position"])
        row["loss"] = float(row["loss"])
        row["accepted"] = row["accepted"] == "True"
    return rows


def find_sequences(input_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, "*_round_candidate_losses.tsv")))
    return [os.path.basename(p)[: -len("_round_candidate_losses.tsv")] for p in paths]


def plot_round_loss_distribution(input_dir: str, output_path: str, n_rounds: int | None) -> None:
    seq_ids = find_sequences(input_dir)
    if not seq_ids:
        raise SystemExit(f"No *_round_candidate_losses.tsv files found under {input_dir}")

    fig, axes = plt.subplots(1, len(seq_ids), figsize=(6.5 * len(seq_ids), 5), squeeze=False)
    axes = axes[0]

    for ax, seq_id in zip(axes, seq_ids):
        rows = read_round_losses(os.path.join(input_dir, f"{seq_id}_round_candidate_losses.tsv"))
        rounds = sorted(set(row["round"] for row in rows))
        if n_rounds is not None:
            rounds = [r for r in rounds if r < n_rounds]
            rows = [row for row in rows if row["round"] < n_rounds]

        losses_by_round = [[row["loss"] for row in rows if row["round"] == r] for r in rounds]
        accepted_by_round = [next(row["loss"] for row in rows if row["round"] == r and row["accepted"]) for r in rounds]

        parts = ax.violinplot(losses_by_round, positions=rounds, showmedians=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor(DISTRIBUTION_COLOR)
            body.set_edgecolor(DISTRIBUTION_COLOR)
            body.set_alpha(0.35)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_edgecolor(DISTRIBUTION_COLOR)
            parts[key].set_alpha(0.6)
        parts["cmedians"].set_edgecolor(MEDIAN_COLOR)
        parts["cmedians"].set_linewidth(1.5)

        ax.scatter(
            rounds, accepted_by_round, color=ACCEPTED_COLOR, edgecolor="white",
            linewidth=0.8, s=45, zorder=3, label="accepted mutation",
        )

        ax.set_title(f"{seq_id}\n{len(rounds)} round(s) of candidates", fontsize=11)
        ax.set_xlabel("round")
        ax.set_xticks(rounds)
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("candidate loss (log scale)")
    axes[-1].legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCEPTED_COLOR,
                       markeredgecolor="white", markersize=7, label="accepted mutation"),
            plt.Rectangle((0, 0), 1, 1, facecolor=DISTRIBUTION_COLOR, alpha=0.35, label="candidate losses"),
            plt.Line2D([0], [0], color=MEDIAN_COLOR, linewidth=1.5, label="median"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, frameon=True, fontsize=9,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dir", nargs="?",
        default="results/e898939e/df4406c4716cd2cf/stage2/best_round_candidate_losses",
        help="Directory containing <seq_id>_round_candidate_losses.tsv files written by "
             "ism_round_candidate_losses.py.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Defaults to round_loss_distribution.png inside input_dir.",
    )
    parser.add_argument(
        "--n_rounds", type=int, default=10,
        help="Only plot rounds before this index (the recorder may have converged and stopped "
             "earlier than this, in which case fewer rounds are plotted anyway).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output_path or os.path.join(args.input_dir, "round_loss_distribution.png")
    plot_round_loss_distribution(args.input_dir, output_path, args.n_rounds)


if __name__ == "__main__":
    main()
