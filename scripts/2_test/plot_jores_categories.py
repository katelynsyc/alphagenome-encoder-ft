#!/usr/bin/env python
"""Build the 3 category-specific scatterplots from evaluate_jores_categories.py's output.

Consumes evaluation_predictions.csv (index/id/category/target_conditions/total_rounds/
perturbation columns, plus the usual "{condition}_true"/"{condition}_pred" columns for
cold/dark/light/warm/maize) so the model does not need to be re-run to make these plots.
Same 5-panel-per-condition layout as plot_jores_scatter.py (diagonal reference line, a
fit line, n and Pearson r annotated per panel), but each figure scopes to a different
row subset and colors points by a different variable instead of local point density:

  1. "Evolved Condition-Specific Sequences" -- evolution rows, each condition's panel
     restricted to rows whose evolution objective(s) actually targeted that condition
     (see evolution_objectives.py). Colored by total evolution rounds.
  2. "Off-Target Evolved Sequences" -- the complementary subset: evolution rows plotted
     on conditions their objective did NOT target, showing how much (or little) a
     directed-evolution run's gains generalize beyond what it was selected for. Colored
     by total evolution rounds on the same scale as (1) since it's the same population.
  3. "Perturbed Library Sequences" -- TFBS shuffling/insertion rows (same row subset in
     every panel, not condition-scoped), colored categorically by which perturbation was
     applied: shuffling, or insertion of 1/2/3 binding sites.

A panel with a handful of extreme points would otherwise force its view to stretch far
past where the bulk of its distribution sits. When a panel has a small number of such
points (see _crop_range_and_outliers), the view is cropped to a percentile-based range
and each excluded point gets a small triangle at the axis edge, rotated to point toward
where it actually falls -- n and Pearson r annotations always reflect the FULL panel,
cropping is a view choice, not a data filter. Panels are only forced onto independent
(un-shared) x/y limits when at least one panel in that figure actually needs cropping;
otherwise all 5 keep the original shared-scale behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

CONDITION_NAMES = ["cold", "dark", "light", "warm", "maize"]

# Distinct from plot_jores_scatter.py's per-condition CONDITION_COLORS scheme -- these
# plots encode a different variable (evolution rounds / perturbation type) in point
# color, so the fit line stays a single neutral accent instead of one color per
# condition. Both the continuous (rounds) and categorical (perturbation type) colorings
# draw from viridis, matching plot_jores_scatter.py's viridis density-coloring scheme.
FIT_LINE_COLOR = "#444444"
ROUNDS_CMAP = "viridis"

# Same 4 purple/blue shades as before (viridis purple/teal-blue, plus a non-viridis
# orchid/sky-blue pair so nothing reads as neon green/yellow), but assigned lightest ->
# darkest in PERTURBATION_ORDER's draw order (insertion_1 drawn first/bottom, shuffling
# drawn last/on top) so the darkest points end up on top instead of buried underneath.
PERTURBATION_COLORS = {
    "insertion_1": "#5dade2",  # sky blue -- lightest, drawn first (bottom)
    "insertion_2": "#9b59b6",  # orchid purple
    "insertion_3": "#31688e",  # teal-blue (viridis)
    "shuffling": "#440154",    # dark purple (viridis) -- darkest, drawn last (on top); a qualitatively different edit, not a step in the ramp
}
PERTURBATION_LABELS = {
    "insertion_1": "Insertion (1 TFBS)",
    "insertion_2": "Insertion (2 TFBS)",
    "insertion_3": "Insertion (3 TFBS)",
    "shuffling": "TFBS Shuffling",
}
PERTURBATION_ORDER = ["insertion_1", "insertion_2", "insertion_3", "shuffling"]

# Outlier cropping: a point is flagged as an outlier if it's more than
# OUTLIER_MAD_THRESHOLD robust-sigmas (median absolute deviation, scaled to be
# std-equivalent under normality) from the panel's median -- this catches a handful of
# genuinely extreme points regardless of what FRACTION of the panel they make up, unlike
# a fixed percentile cutoff (e.g. 1st/99th), which fails exactly when it matters most:
# a panel with a noisy model's occasional wild misprediction can easily have >1% of its
# points sitting far from the bulk, so a percentile trim would leave them inside the
# "core" range instead of flagging them.
#
# Cropping is skipped only when outliers are BOTH more than MAX_OUTLIER_MARKERS AND more
# than MAX_OUTLIER_FRACTION of the panel -- either cap alone would misfire: a fixed count
# alone rejects e.g. 31 outliers out of 3440 points (0.9%, clearly still "a handful"
# relative to the panel), while a fixed fraction alone can reject a tiny panel with only
# 3 stray points out of 40 (7%, still just 3 points to draw triangles for). Requiring
# BOTH to be exceeded before giving up means only a panel that's actually wide (a lot of
# points relative to its size) skips cropping, instead of one that's merely large.
OUTLIER_MAD_THRESHOLD = 6.0
MAD_TO_SIGMA = 1.4826  # scales MAD to be a consistent estimator of std under a normal distribution
OUTLIER_PAD_FRAC = 0.08
MAX_OUTLIER_MARKERS = 15
MAX_OUTLIER_FRACTION = 0.01
OUTLIER_MARKER_COLOR = "#B22222"
OUTLIER_MARKER_INSET_FRAC = 0.07


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 3 Jores design-category scatterplots")
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help='Prepended to each figure title, e.g. --model_name plantGREP -> '
             '"plantGREP -- AlphaGenome: Evolved Condition-Specific Sequences" '
             '(same convention as plot_jores_scatter.py).',
    )
    return parser


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    denom = np.std(y_true) * np.std(y_pred)
    if denom == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _new_figure(condition_names: list[str], *, share: bool):
    fig, axes = plt.subplots(
        1, len(condition_names), figsize=(4 * len(condition_names), 4), sharex=share, sharey=share
    )
    # Extra horizontal gutter between panels -- unshared axes (see `share` above) can
    # each have an independently-sized tick-label range (e.g. "-150" vs "0.0"), and the
    # default spacing is only wide enough when every panel's tick labels are identical
    # widths, which shared axes used to guarantee but independent ones no longer do.
    fig.subplots_adjust(wspace=0.35)
    return fig, axes


def _crop_range_and_outliers(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """A (lo, hi) view range sized to the panel's non-outlier ("inlier") points, plus a
    boolean mask of the outliers excluded from it (see OUTLIER_MAD_THRESHOLD above).
    Returns None when there's nothing worth cropping: too few points for a stable
    median/MAD, a degenerate (zero-spread) panel, zero outliers (the full range already
    reads fine), or too many outliers by BOTH the count and fraction caps (see
    MAX_OUTLIER_MARKERS/MAX_OUTLIER_FRACTION above)."""
    if y_true.size < 5:
        return None
    combined = np.concatenate([y_true, y_pred])
    median = float(np.median(combined))
    mad = float(np.median(np.abs(combined - median)))
    if mad == 0:
        return None
    threshold = OUTLIER_MAD_THRESHOLD * MAD_TO_SIGMA * mad

    outlier_mask = (np.abs(y_true - median) > threshold) | (np.abs(y_pred - median) > threshold)
    n_outliers = int(outlier_mask.sum())
    if n_outliers == 0:
        return None
    if n_outliers > MAX_OUTLIER_MARKERS and n_outliers > MAX_OUTLIER_FRACTION * y_true.size:
        return None

    inliers = combined[np.abs(combined - median) <= threshold]
    lo, hi = float(inliers.min()), float(inliers.max())
    span = hi - lo
    if span <= 0:
        return None
    pad = OUTLIER_PAD_FRAC * span
    lo, hi = lo - pad, hi + pad
    return lo, hi, outlier_mask


def _draw_outlier_marker(ax, x: float, y: float, lo: float, hi: float, color) -> None:
    """A small triangle pinned near the edge of the cropped view, rotated to point
    toward an excluded point that actually sits at (x, y), off-screen. `color` is that
    point's own data color (evolution-rounds cmap value, or perturbation-type color) --
    the triangle SHAPE + black edge is what marks it as an outlier indicator rather than
    a regular scatter point; its fill still identifies which category it belongs to."""
    span = hi - lo
    inset = OUTLIER_MARKER_INSET_FRAC * span
    center = (lo + hi) / 2
    clipped_x = min(max(x, lo + inset), hi - inset)
    clipped_y = min(max(y, lo + inset), hi - inset)
    dx, dy = x - center, y - center
    if dx == 0 and dy == 0:
        return
    angle = math.degrees(math.atan2(-dx, dy))  # angle=0 -> default triangle marker points "up" (+y)
    ax.plot(
        clipped_x, clipped_y, marker=(3, 0, angle), markersize=9,
        color=color, markeredgecolor="black", markeredgewidth=0.6,
        linestyle="none", clip_on=True, zorder=6,
    )


def _draw_reference_and_fit(ax, y_true: np.ndarray, y_pred: np.ndarray, lo: float | None, hi: float | None) -> None:
    if lo is not None:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")

    if y_true.size >= 2:
        m, b = np.polyfit(y_true, y_pred, 1)
        x_line = np.array([lo, hi]) if lo is not None else np.array([y_true.min(), y_true.max()])
        ax.plot(x_line, m * x_line + b, color=FIT_LINE_COLOR, linewidth=1.5)


def _finalize_panel(
    ax,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop: tuple[float, float, np.ndarray] | None,
    point_colors=None,
    show_ylabel: bool = True,
) -> None:
    """`point_colors`, when given, is one color per (y_true, y_pred) row -- the same
    color that point was scattered in -- so an outlier's triangle marker still shows
    which category/round it belongs to instead of a single undifferentiated color."""
    if crop is not None:
        lo, hi, outlier_mask = crop
    else:
        finite = np.concatenate([y_true, y_pred])
        finite = finite[np.isfinite(finite)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (None, None)
        outlier_mask = None

    _draw_reference_and_fit(ax, y_true, y_pred, lo, hi)

    if crop is not None:
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        outlier_colors = (
            point_colors[outlier_mask] if point_colors is not None else [OUTLIER_MARKER_COLOR] * int(outlier_mask.sum())
        )
        for out_x, out_y, out_color in zip(y_true[outlier_mask], y_pred[outlier_mask], outlier_colors):
            _draw_outlier_marker(ax, out_x, out_y, lo, hi, out_color)

    r = _pearsonr(y_true, y_pred)
    ax.set_title(f"{name} (r={r:.3f})" if not math.isnan(r) else f"{name} (r=n/a)")
    ax.set_xlabel("Actual log2 enrichment")
    # Only the leftmost panel gets a y-axis label -- with independent (un-shared) axes,
    # every panel can have a differently-sized tick-label range (e.g. "-150" vs "0.0"),
    # and a repeated rotated ylabel on every panel was colliding with its left neighbor's
    # plot area once those tick labels stopped being identical widths across panels.
    if show_ylabel:
        ax.set_ylabel("Predicted log2 enrichment")
    ax.annotate(f"n = {y_true.size}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
    if crop is not None:
        n_outliers = int(outlier_mask.sum())
        noun = "outlier" if n_outliers == 1 else "outliers"
        ax.annotate(
            f"{n_outliers} {noun} cropped", xy=(0.05, 0.85), xycoords="axes fraction",
            fontsize=7, color=OUTLIER_MARKER_COLOR, style="italic",
        )


def _title(base_title: str, model_name: str | None = None) -> str:
    title = f"AlphaGenome: {base_title}"
    return title


def _finalize_figure(
    fig, title: str, *, top: float, bottom: float, right: float | None = None,
    model_name: str | None = None,
) -> None:
    """Suptitle + margins shared by all 3 plots. `top`/`bottom` are subplots_adjust
    fractions -- distinct per plot since the perturbation plot also reserves bottom
    space for its legend, on top of the x-axis label every plot already has. `right`
    reserves room for the evolution-rounds colorbar (see plot_evolution_scatter) --
    call this BEFORE adding that colorbar so its axis is placed against the final
    layout instead of triggering matplotlib's own axes-shrinking, which the auto
    fig.colorbar(ax=axes) placement used to fight with subsequent subplots_adjust
    calls and could overlap the rightmost panel."""
    fig.subplots_adjust(top=top, bottom=bottom, **({"right": right} if right is not None else {}))
    fig.suptitle(_title(title, model_name), fontsize=13, y=0.99)


def plot_evolution_scatter(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    mode: str,
    condition_names: list[str] = CONDITION_NAMES,
    model_name: str | None = None,
) -> None:
    """mode='on_target' -> plot_jores_categories's plot (1); 'off_target' -> plot (2)."""
    evolution_rows = [row for row in rows if row["category"] == "evolution"]

    per_condition: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in condition_names:
        true_vals: list[float] = []
        pred_vals: list[float] = []
        round_vals: list[int] = []
        for row in evolution_rows:
            targets = {token for token in row["target_conditions"].split(",") if token}
            is_target = name in targets
            include = is_target if mode == "on_target" else not is_target
            if not include:
                continue
            true_vals.append(float(row[f"{name}_true"]))
            pred_vals.append(float(row[f"{name}_pred"]))
            round_vals.append(int(row["total_rounds"]))
        per_condition[name] = (
            np.array(true_vals, dtype=np.float64),
            np.array(pred_vals, dtype=np.float64),
            np.array(round_vals, dtype=np.float64),
        )

    all_rounds = np.concatenate([rounds for _, _, rounds in per_condition.values() if rounds.size])
    vmin = float(all_rounds.min()) if all_rounds.size else 0.0
    vmax = float(all_rounds.max()) if all_rounds.size else 1.0

    crops = {name: _crop_range_and_outliers(per_condition[name][0], per_condition[name][1]) for name in condition_names}
    share = not any(crop is not None for crop in crops.values())

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(ROUNDS_CMAP)

    fig, axes = _new_figure(condition_names, share=share)
    scatter = None
    for i, (ax, name) in enumerate(zip(axes, condition_names)):
        y_true, y_pred, rounds = per_condition[name]
        # Draw ascending by round count last-first-drawn-is-bottom, so the lightest
        # (highest-round, most-viridis-yellow) points land on top of the darker,
        # low-round points instead of being buried under them.
        order = np.argsort(rounds)
        scatter = ax.scatter(
            y_true[order], y_pred[order],
            c=rounds[order], cmap=ROUNDS_CMAP, vmin=vmin, vmax=vmax,
            s=10, alpha=0.7, edgecolors="none",
        )
        # Same cmap/norm the scatter above uses, so an outlier's triangle is colored by
        # its own round count instead of a single undifferentiated outlier color.
        point_colors = cmap(norm(rounds))
        _finalize_panel(ax, name, y_true, y_pred, crops[name], point_colors=point_colors, show_ylabel=(i == 0))

    # Reserve the right 8% of the figure for the colorbar BEFORE placing it, then give
    # it an explicit axis spanning the panels' actual vertical extent -- letting
    # fig.colorbar(ax=axes) auto-shrink the panels instead used to leave the colorbar
    # sitting right against (or overlapping) the rightmost ("maize") panel.
    _finalize_figure(fig, title, top=0.83, bottom=0.15, right=0.90, model_name=model_name)
    if scatter is not None:
        panel_position = axes[0].get_position()
        cbar_ax = fig.add_axes([0.92, panel_position.y0, 0.015, panel_position.height])
        fig.colorbar(scatter, cax=cbar_ax, label="Evolution rounds")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_perturbation_scatter(
    rows: list[dict[str, str]],
    output_path: Path,
    title: str,
    condition_names: list[str] = CONDITION_NAMES,
    model_name: str | None = None,
) -> None:
    perturbation_rows = [row for row in rows if row["category"] in ("tfbs_shuffling", "tfbs_insertion")]
    # Same order as perturbation_rows -- every row here has a non-empty "perturbation"
    # label by construction, so an outlier's triangle can be colored by which
    # perturbation it belongs to instead of a single undifferentiated outlier color.
    point_colors = np.array([PERTURBATION_COLORS[row["perturbation"]] for row in perturbation_rows])

    per_condition = {
        name: (
            np.array([float(row[f"{name}_true"]) for row in perturbation_rows], dtype=np.float64),
            np.array([float(row[f"{name}_pred"]) for row in perturbation_rows], dtype=np.float64),
        )
        for name in condition_names
    }
    crops = {name: _crop_range_and_outliers(*per_condition[name]) for name in condition_names}
    share = not any(crop is not None for crop in crops.values())

    fig, axes = _new_figure(condition_names, share=share)
    for i, (ax, name) in enumerate(zip(axes, condition_names)):
        y_true_all, y_pred_all = per_condition[name]

        for label in PERTURBATION_ORDER:
            mask = np.array([row["perturbation"] == label for row in perturbation_rows])
            if not mask.any():
                continue
            ax.scatter(
                y_true_all[mask], y_pred_all[mask],
                c=PERTURBATION_COLORS[label], label=PERTURBATION_LABELS[label],
                s=10, alpha=0.7, edgecolors="none",
            )

        _finalize_panel(ax, name, y_true_all, y_pred_all, crops[name], point_colors=point_colors, show_ylabel=(i == 0))

    # Bottom margin is reserved (via _finalize_figure's subplots_adjust) for BOTH the
    # per-panel "Actual log2 enrichment" x-axis label and this legend below it -- the
    # legend sits at y=0.04 in figure coordinates, safely under the axes region (which
    # subplots_adjust below caps at bottom=0.30), so it can no longer land on top of the
    # x-axis label text the way a negative-y/bbox_inches="tight" anchor could.
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(PERTURBATION_ORDER), bbox_to_anchor=(0.5, 0.04), fontsize=9)

    _finalize_figure(fig, title, top=0.83, bottom=0.30, model_name=model_name)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv).resolve()
    if not predictions_path.exists():
        parser.error(f"Predictions CSV not found: {predictions_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(predictions_path)

    on_target_path = output_dir / "evolved_condition_specific.png"
    plot_evolution_scatter(rows, on_target_path, "Evolved Condition-Specific Sequences", mode="on_target",
                            model_name=args.model_name)
    print(f"Saved plot to {on_target_path}")

    off_target_path = output_dir / "off_target_evolved.png"
    plot_evolution_scatter(rows, off_target_path, "Off-Target Evolved Sequences", mode="off_target",
                            model_name=args.model_name)
    print(f"Saved plot to {off_target_path}")

    perturbed_path = output_dir / "perturbed_library.png"
    plot_perturbation_scatter(rows, perturbed_path, "Perturbed Library Sequences", model_name=args.model_name)
    print(f"Saved plot to {perturbed_path}")


if __name__ == "__main__":
    main()
