#!/usr/bin/env python
"""Plot the distribution of val_pearson across every trial of an ag_hpsweep Ray Tune run.

Two stacked panels sharing the same x-axis (val_pearson):
  1. Histogram of every trial's final val_pearson, with the pre-sweep baseline marked as a
     vertical reference line and the single best trial marked as a star.
  2. A strip plot of every trial (points, y-jittered only for readability) colored by whether
     the trial ran to its own completion (finished stage1+stage2, whether via
     stage.early_stopping_patience or by reaching stage.num_epochs/second_stage_epochs) or was
     pruned mid-training by the ASHAScheduler in train_ag_tune.py.

Why we can't just read the --csv that tune_status_combined.py writes: that CSV keeps only each
trial's LAST reported row, which is enough for val_pearson (see below) but not for the
completed-vs-pruned split -- that needs each trial's *whole* history. So this script re-scans
every trial's progress.csv directly instead of shelling out to tune_status_combined.py.

How completed-vs-pruned is actually determined (see train_ag.py:run() / train_ag_tune.py):
  - train_ag.run() reports one metrics dict per real training epoch (train_pearson etc. all
    populated) via epoch_callback, for however many epochs stage1 and then stage2 actually run.
  - Only if BOTH stages finish on their own (early_stopping_patience triggers, or num_epochs /
    second_stage_epochs is reached) does run() go on to run the held-out test evaluation and
    report one extra "final_test" row per stage -- these rows have every train_* field empty
    (see train_ag.py's test_payload, which only carries stage/epoch/val_pearson/test_*).
  - ASHAScheduler (time_attr="stage2_epoch", grace_period=15, reduction_factor=4) can instead
    kill the trial's Ray actor mid-training. That happens strictly between two ordinary epoch
    reports, so a pruned trial's progress.csv simply stops -- it never gets a final_test row.
  So: last row's train_pearson is empty  => the trial completed on its own.
      last row's train_pearson is populated (a real epoch row) => the trial was cut off, almost
      certainly by ASHA (validated against this run: of the trials not marked completed, every
      single one stops with stage2_epoch exactly at 15 or 60 -- ASHA's own grace_period/rung
      schedule under reduction_factor=4 -- not scattered at arbitrary epoch counts, which is
      what a crash/preemption/still-running trial would look like instead).

Either way, a trial's reported val_pearson (last row, whichever kind) is exactly what
tune_status_combined.py's table already shows per trial -- for a completed trial this is the
backfilled best_monitor (the stage's best validation pearson, e.g. e898939e stage2 79 0.8425),
for a pruned trial it's the last real epoch's val_pearson before ASHA cut it off.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set_style('whitegrid')

# Same two categorical slots already used for a 2-series split elsewhere in this project (see
# scripts/2_test/plot_pearson_comparison.py) -- kept consistent rather than re-deriving a palette.
COMPLETED_COLOR = "#36669c"
PRUNED_COLOR = "#3ec995"
# The top panel's histogram is the WHOLE population, not the completed subset -- deliberately
# neutral gray rather than reusing COMPLETED_COLOR, so it doesn't visually read as "just the
# completed trials" once the categorical colors appear in the panel below.
HIST_COLOR = "#9a988f"
BASELINE_COLOR = "#52514e"   # secondary-ink gray -- a reference line, not a data series
BEST_COLOR = "#c0392b"       # distinct warm accent so the single best trial doesn't read as a 3rd series


@dataclass
class TrialResult:
    trial_id: str
    val_pearson: float
    stage: str | None
    iteration: int | None
    completed: bool  # True = finished on its own; False = cut off mid-training (pruned by ASHA)


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    return f if not np.isnan(f) else None


def scan_trials(experiment_dir: str) -> list[TrialResult]:
    """Re-read every train_fn_<trial_id>_... /progress.csv under experiment_dir directly (see
    module docstring for why this can't just reuse tune_status_combined.py's CSV output)."""
    results: list[TrialResult] = []
    for trial_dir in sorted(glob.glob(os.path.join(experiment_dir, "train_fn_*"))):
        name = os.path.basename(trial_dir)
        parts = name.split("_")
        if len(parts) < 3:
            continue
        trial_id = parts[2]

        progress_path = os.path.join(trial_dir, "progress.csv")
        if not os.path.exists(progress_path):
            continue
        with open(progress_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        last = rows[-1]
        val_pearson = _parse_float(last.get("val_pearson"))
        if val_pearson is None:
            continue
        completed = last.get("train_pearson") in (None, "")

        iteration = last.get("training_iteration")
        results.append(
            TrialResult(
                trial_id=trial_id,
                val_pearson=val_pearson,
                stage=last.get("stage"),
                iteration=int(float(iteration)) if iteration not in (None, "") else None,
                completed=completed,
            )
        )
    return results


def plot_val_pearson_distribution(
    trials: list[TrialResult],
    baseline_pearson: float,
    baseline_label: str,
    output_path: str,
    n_bins: int = 40,
) -> None:
    values = np.array([t.val_pearson for t in trials])
    best = max(trials, key=lambda t: t.val_pearson)

    n_completed = sum(t.completed for t in trials)
    n_pruned = len(trials) - n_completed

    fig, (ax_hist, ax_strip) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]},
    )

    # --- Panel 1: overall distribution -------------------------------------------------
    bins = np.linspace(values.min(), values.max(), n_bins + 1)
    ax_hist.hist(values, bins=bins, color=HIST_COLOR, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax_hist.set_ylabel(f"Trial Count (n={len(trials)})", fontsize=14, labelpad=10)
    ax_hist.set_title("Validation Pearson's r across Hyperparameter Sweep", fontsize=18, fontweight="bold", pad=16)

    # --- Panel 2: every trial, colored by completed vs. pruned by ASHA -----------------
    rng = np.random.default_rng(0)
    completed_vals = [t.val_pearson for t in trials if t.completed]
    pruned_vals = [t.val_pearson for t in trials if not t.completed]
    ax_strip.scatter(
        pruned_vals, rng.uniform(-1, 1, size=len(pruned_vals)),
        s=14, color=PRUNED_COLOR, alpha=0.6, linewidths=0,
        label=f"Pruned by ASHA (n={n_pruned})",
    )
    ax_strip.scatter(
        completed_vals, rng.uniform(-1, 1, size=len(completed_vals)),
        s=14, color=COMPLETED_COLOR, alpha=0.6, linewidths=0,
        label=f"Ran to completion (n={n_completed})",
    )
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("All Trials (jittered)", fontsize=14, labelpad=10)
    ax_strip.set_xlabel("Validation Pearson's r", fontsize=14, labelpad=10)
    # Points are jittered across the FULL x-range (including a pile-up of diverged trials right
    # at x=0), legend placed outside the axes so it never sits on top of any point regardless of jitter.
    ax_strip.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.15), borderaxespad=0,
        frameon=True, facecolor="white", edgecolor="#e1e0d9", fontsize=9,
    )

    # --- Baseline + best-trial markers, drawn on both panels ---------------------------
    for ax in (ax_hist, ax_strip):
        ax.axvline(baseline_pearson, color=BASELINE_COLOR, linestyle="--", linewidth=1.5, zorder=3)
        ax.axvline(best.val_pearson, color=BEST_COLOR, linestyle="-", linewidth=1.5, zorder=3)

    # Anchored into the near-empty low-val_pearson region (the failed/diverged trials pile up
    # near 0, but the 0.1-0.5 band is sparse) rather than stacked at the top of their own lines,
    # which collide with each other whenever baseline and best are close together in x.
    top = ax_hist.get_ylim()[1]
    ax_hist.annotate(
        f"{baseline_label}\nr={baseline_pearson:.5f}",
        xy=(baseline_pearson, 0.55 * top), xytext=(0.30 * values.max(), 0.92 * top),
        ha="center", va="top", fontsize=13, color=BASELINE_COLOR,
        arrowprops=dict(arrowstyle="-", color=BASELINE_COLOR, linewidth=1),
    )
    ax_hist.annotate(
        f"Best trial\nr={best.val_pearson:.4f}",
        xy=(best.val_pearson, 0.30 * top), xytext=(0.30 * values.max(), 0.55 * top),
        ha="center", va="top", fontsize=13, color=BEST_COLOR,
        arrowprops=dict(arrowstyle="-", color=BEST_COLOR, linewidth=1),
    )

    for ax in (ax_hist, ax_strip):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=12)

    fig.tight_layout(h_pad=2.5)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}")
    print(
        f"n={len(trials)} trials | completed={n_completed} | pruned_by_asha={n_pruned} | "
        f"best={best.trial_id} val_pearson={best.val_pearson:.4f} | baseline={baseline_pearson:.5f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiment_dir", nargs="?",
        default="results/ray_tune/ag_hpsweep_1000",
        help="Ray Tune experiment storage dir, e.g. results/ray_tune/ag_hpsweep_1000",
    )
    parser.add_argument("--baseline_pearson", type=float, default=0.70671)
    parser.add_argument("--baseline_label", type=str, default="Before hyperparameter sweep:")
    parser.add_argument("--output_path", type=str, default="results/plots/ag_hpsweep_1000_val_pearson_distribution.png")
    parser.add_argument("--n_bins", type=int, default=40)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    trials = scan_trials(args.experiment_dir)
    if not trials:
        raise SystemExit(f"No scored trials found under {args.experiment_dir}")
    plot_val_pearson_distribution(
        trials,
        baseline_pearson=args.baseline_pearson,
        baseline_label=args.baseline_label,
        output_path=args.output_path,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
