from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
import seaborn; seaborn.set_style('white')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from captum.attr import Saliency
from tangermeme.plot import plot_logo

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    load_config_from_checkpoint,
    create_jores_splits,
    summarize_species_masks,
)

# Order must match JoresMPRADataset._targets (see mydata.py).
CONDITION_NAMES = [
    "cold",
    # "dark",
    # "light",
    "warm",
    # "maize",
]
CONDITION = {"cold": 0, "dark": 1, "light": 2, "warm": 3, "maize": 4}

# y-axis quantity for each attr_dict entry, used to label the logo plots.
ATTRIBUTION_YLABELS = {
    "cold": "Corrected saliency\n(d predicted cold enrichment / d input)",
    "dark": "Corrected saliency\n(d predicted dark enrichment / d input)",
    "light": "Corrected saliency\n(d predicted light enrichment / d input)",
    "warm": "Corrected saliency\n(d predicted warm enrichment / d input)",
    "maize": "Corrected saliency\n(d predicted maize enrichment / d input)",
    "cold_not_warm": "Corrected saliency\n(d predicted [cold - warm] enrichment / d input)",
    "warm_not_cold": "Corrected saliency\n(d predicted [warm - cold] enrichment / d input)",
}

# name -> (a, b) meaning derived[name] = attr_dict[a] - attr_dict[b]. Valid for both
# raw and corrected maps: differentiation and apply_correction's mean-subtraction are
# both linear operators, so grad(f_a - f_b) == grad(f_a) - grad(f_b) exactly, and the
# same identity holds after correction -- no extra saliency call is needed.
DIFFERENCE_PAIRS = {
    "cold_not_warm": ("cold", "warm"),
    "warm_not_cold": ("warm", "cold"),
}


def add_condition_differences(attr_dict, pairs=DIFFERENCE_PAIRS):
    """Derives difference conditions (e.g. cold_not_warm = cold - warm) from
    already-computed per-condition maps -- exact, not an approximation (see
    DIFFERENCE_PAIRS docstring above)."""
    out = dict(attr_dict)
    for name, (a, b) in pairs.items():
        out[name] = attr_dict[a] - attr_dict[b]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute gradient-corrected input saliency for an encoder-only AlphaGenome Jores checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache_path", type=str, default=None,
                         help="Where to save/load the raw gradient result (X, organism_idx, y0, "
                              "grad_raw, grad_corrected). Defaults to '<checkpoint>_gradient_cache.pt' "
                              "next to the checkpoint.")
    parser.add_argument("--recompute", action="store_true",
                         help="Ignore an existing cache and re-run saliency.")
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True,
                         help="Include the 15bp Jores adapters on each side of the sequence when "
                              "building the dataset (matches training). Pass --no-use_adapters "
                              "to compute gradients on the raw ~170bp insert only, with no adapters added.")
    parser.add_argument("--gradient_batch_size", type=int, default=32,
                         help="Batch size used when computing saliency (memory only -- does not change values).")
    return parser


def load_model_and_test_data(checkpoint_path, config, use_adapters=True):
    """Cheap, one-time setup: model + test_dataset. In a notebook, call this once
    (after resolving `config`, e.g. via load_config_from_checkpoint) and keep the
    results around -- nothing here depends on gradients having been computed yet.

    use_adapters=False builds the raw ~170bp insert with no adapters concatenated on
    either side (see JoresMPRADataset), instead of the adapter-padded training construct."""
    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=device)  # loads trained weights

    _, _, test_dataset = create_jores_splits(  # held out set for testing
        config.data.input_tsv,
        seed=config.runtime.seed,
        sequence_length=config.data.sequence_length,
        use_adapters=use_adapters,
        reverse_complement=config.data.reverse_complement,
        rc_prob=config.data.rc_prob,
        random_shift=config.data.random_shift,
        shift_prob=config.data.shift_prob,
        max_shift=config.data.max_shift,
    )
    return model, device, test_dataset


def raw_gradients(model, X, organism_idx, device, task_names=CONDITION_NAMES, batch_size=32):
    """Signed input saliency, one map per condition in task_names. AlphaGenomeEncoderModel.forward
    already takes one-hot directly as (N, L, 4) and returns a plain (N, 5) tensor. abs=False is required: captum's default
    abs=True would discard the sign, which the gradient correction (and grad-times-input) both need.

    Returns {condition_name: (N, L, 4)} raw (uncorrected) gradients."""
    model = model.to(device).eval()
    saliency = Saliency(model)

    attrs = {name: [] for name in task_names}
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size].to(device).clone().requires_grad_(True)
        ob = organism_idx[start:start + batch_size].to(device)
        for name in task_names:
            a = saliency.attribute(xb, target=CONDITION[name], additional_forward_args=(ob,), abs=False)
            attrs[name].append(a.detach().cpu())

    return {name: torch.cat(v, dim=0) for name, v in attrs.items()}


def apply_correction(attr, nucleotide_dim=-1):
    """Gradient correction from Majdandzic, Rajesh & Koo (2023), Genome Biology 24:109 --
    projects the raw gradient onto the simplex's tangent hyperplane by subtracting the
    per-position mean across nucleotide channels. One-hot DNA only ever lies on the
    simplex A+C+G+T=1, so the model is free to behave arbitrarily off it; that freedom
    injects a spurious gradient component orthogonal to the simplex. Removing the
    per-position mean removes exactly that component, leaving the tangential
    (data-supported) part of the gradient."""
    return attr - attr.mean(dim=nucleotide_dim, keepdim=True)


def run_gradients(model, test_dataset, device, indices=None, task_names=CONDITION_NAMES, batch_size=128):
    """The expensive step -- run this ONCE per notebook session and hold on to
    X/organism_idx/y0/grad_raw/grad_corrected. Downstream plotting is cheap tensor
    math and can be re-run freely without calling this again."""
    if indices is None:
        indices = range(len(test_dataset))

    # stack the per-item (L, 4) one-hot tensors instead of getting one example at a time
    X = torch.stack([test_dataset[i][0] for i in indices])   # (N, L, 4)
    organism_idx = torch.zeros(X.shape[0], dtype=torch.long)  # confirmed unused -- we only use the encoder

    # Chunked the same way raw_gradients() below chunks its saliency pass -- the full
    # test set (N ~ 33k) in one forward call overflows max_pool1d's int32 indexing
    # (N * 768 channels * 170 positions at the first pool alone is > 2**31).
    y0_batches = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:start + batch_size].to(device)
            ob = organism_idx[start:start + batch_size].to(device)
            y0_batches.append(model(xb, ob).detach().cpu())
    y0 = torch.cat(y0_batches, dim=0)  # (N, 5) reference predictions
    print(f"y0 dims: {tuple(y0.shape)}")  # (N, 5) -- [cold, dark, light, warm, maize] on the reference sequences

    grad_raw = raw_gradients(model, X, organism_idx, device, task_names=task_names, batch_size=batch_size)
    grad_corrected = {name: apply_correction(g) for name, g in grad_raw.items()}

    return X, organism_idx, y0, grad_raw, grad_corrected


def default_cache_path(checkpoint_path, use_adapters=True):
    suffix = "" if use_adapters else "_no_adapters"
    return checkpoint_path.parent / f"{checkpoint_path.stem}_gradient_cache{suffix}.pt"


def save_grad_cache(cache_path, X, organism_idx, y0, grad_raw, grad_corrected):
    torch.save({
        "X": X, "organism_idx": organism_idx, "y0": y0,
        "grad_raw": grad_raw, "grad_corrected": grad_corrected,
    }, cache_path)


def load_grad_cache(cache_path):
    cached = torch.load(cache_path, map_location="cpu")
    return cached["X"], cached["organism_idx"], cached["y0"], cached["grad_raw"], cached["grad_corrected"]


def _mean_logo_matrix(attr, mask=None):
    """(N, 4, L) attribution -> (4, L), averaged across all N sequences (mask=None)
    or just the sequences where mask is True."""
    arr = attr.detach().cpu().numpy() if hasattr(attr, "detach") else np.asarray(attr)
    if mask is not None:
        arr = arr[np.asarray(mask)]
    return arr.mean(axis=0), arr.shape[0]


def _standardized_ylim(logo_matrices, pad_frac=0.05):
    """Shared y-axis limits for a batch of (4, L) logo matrices. Uses the same
    stacked-height logic plot_logo itself uses internally -- the sum of the
    positive-valued channels at each position, and the sum of the negative-valued
    ones -- rather than each entry's raw min/max, since several bases can stack at
    one position and a naive min/max would clip that row's glyphs. Rounded outward
    to the nearest 0.5 so the grey 0.5-interval gridlines land on the frame edges.
    (Mirrors saturation_mutagenesis.py's helper of the same name, for consistent
    formatting between ISM and gradient logo plots.)
    """
    pos_max = max(np.sum(np.where(m > 0, m, 0), axis=0).max() for m in logo_matrices)
    neg_min = min(np.sum(np.where(m < 0, m, 0), axis=0).min() for m in logo_matrices)
    pad = pad_frac * (pos_max - neg_min) if pos_max > neg_min else 0.05
    lo = np.floor((neg_min - pad) * 2) / 2
    hi = np.ceil((pos_max + pad) * 2) / 2
    return float(lo), float(hi)


def _style_logo_ax(ax, ylim):
    """Apply consistent framing on top of plot_logo()'s output. plot_logo hides
    the top/right/bottom spines and autoscales ylim per row, so this re-enables a
    full black box, forces a shared ylim across rows/figures, adds small tick
    marks, and draws faint grey dotted horizontal gridlines every 0.5 units.

    seaborn.set_style('white') (module import time, above) sets the xtick.bottom /
    ytick.left rcParams to False, which suppresses tick marks project-wide -- so
    bottom/left must be forced back on explicitly here, not just styled via length/color.
    (Mirrors saturation_mutagenesis.py's helper of the same name.)
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", which="both", direction="out", length=3, width=0.8, color="black",
                    labelsize=8, bottom=True, left=True, top=False, right=False)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.7, color="grey", alpha=0.6)
    ax.set_axisbelow(True)


def _sanitize_filename(text):
    """e.g. 'At-12807(PP)_fwd' -> 'At-12807_PP__fwd' -- some ids carry characters
    that are awkward in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def find_row_indices_for_ids(input_tsv, query_ids):
    """Resolve arbitrary sequence ids to test-split row indices, aligned to the same
    row order as run_gradients() / summarize_species_masks(). (Mirrors
    saturation_mutagenesis.py's helper of the same name -- see its docstring for the
    exact/prefix matching rule and why an unsuffixed id like "At-12806" expands to
    every strand variant present in the test split.)

    Returns {matched_id: row_idx}. Raises KeyError listing any query_ids with zero
    matches (e.g. because they're only in the train/val split, not test -- gradients
    are only cached for the test split).
    """
    _, split_rows = summarize_species_masks(input_tsv)
    ids = [row["id"] for row in split_rows]

    matches = {}
    unmatched = []
    for query_id in query_ids:
        hits = [i for i, row_id in enumerate(ids) if row_id == query_id or row_id.startswith(f"{query_id}_")]
        if not hits:
            unmatched.append(query_id)
            continue
        for i in hits:
            matches[ids[i]] = i

    if unmatched:
        raise KeyError(
            f"id(s) not found in {input_tsv}'s test split (gradients are only cached for the test split -- "
            f"check modelling_data_tamsACR.tsv's 'set' column): {unmatched}"
        )
    return matches


def plot_gradient_logos_by_id(attr_dict, input_tsv, ids, output_dir=None, X=None, hypothetical=True):
    """Notebook entry point -- plot gradient logos for specific, named sequences
    (rather than a top-n selection; see plot_top_expression_gradient_logos for
    that). ids are looked up by the dataset's 'id' column via
    find_row_indices_for_ids -- an unsuffixed id like "At-12806" expands to every
    strand variant present in the test split.

    Same per-figure layout, styling, and per-sequence y-axis standardization as
    plot_top_expression_gradient_logos.

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)} -- grad_corrected
                     stores (N, L, 4), so transpose first, e.g.
                     {name: g.transpose(1, 2) for name, g in grad_corrected.items()}.
    ids:            sequence id(s) to plot, e.g. ["Sl-sh2115_rev", "At-12806"].
    output_dir:     if given, each figure is saved to output_dir / "gradient_logo_seq_<matched_id>.png".
    X, hypothetical: see plot_top_expression_gradient_logos -- hypothetical=False
                     projects onto the observed base only (gradient x input; requires
                     X, same (N, 4, L) layout as attr_dict).

    Returns {matched_id: fig}.
    """
    if not hypothetical and X is None:
        raise ValueError("hypothetical=False requires passing X (the one-hot sequences, same (N, 4, L) layout as attr_dict)")

    id_to_row = find_row_indices_for_ids(input_tsv, ids)

    _, split_rows = summarize_species_masks(input_tsv)
    cold_expr = {row["id"]: float(row["enrichment_cold"]) for row in split_rows}
    warm_expr = {row["id"]: float(row["enrichment_warm"]) for row in split_rows}

    conditions = list(attr_dict)
    n_seqs = next(iter(attr_dict.values())).shape[0]
    if n_seqs != len(split_rows):
        raise ValueError(
            f"attr_dict has {n_seqs} sequences but {input_tsv}'s test split has {len(split_rows)} rows -- "
            "plot_gradient_logos_by_id only works when the gradient cache covers the FULL test set "
            "(run_gradients(..., indices=None), the main() default), not a stratified subset."
        )

    def _logo_matrix(condition_name, row_idx):
        arr = attr_dict[condition_name][row_idx]
        logo_matrix = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        if not hypothetical:
            x_row = X[row_idx]
            x_row = x_row.detach().cpu().numpy() if hasattr(x_row, "detach") else np.asarray(x_row)
            logo_matrix = logo_matrix * x_row  # project onto the observed base (gradient x input)
        return logo_matrix

    figs = {}
    for matched_id, row_idx in id_to_row.items():
        logo_matrices = [_logo_matrix(condition_name, row_idx) for condition_name in conditions]
        ylim = _standardized_ylim(logo_matrices)  # shared across this sequence's own condition rows only

        fig, axes = plt.subplots(len(conditions), 1, figsize=(9, 1.8 * len(conditions)), sharex=True, squeeze=False)

        for row, (condition_name, logo_matrix) in enumerate(zip(conditions, logo_matrices)):
            ax = axes[row, 0]
            plot_logo(logo_matrix, ax=ax)  # plot_logo's standard A/C/G/T coloring
            _style_logo_ax(ax, ylim)
            ax.set_ylabel(condition_name, fontsize=9, rotation=0, ha="right", va="center")

        title = f"{matched_id}  (cold={cold_expr[matched_id]:.2f}, warm={warm_expr[matched_id]:.2f})"
        axes[0, 0].set_title(title, fontsize=10, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel("Corrected saliency score", fontsize=9)
        fig.tight_layout()

        if output_dir:
            fig.savefig(Path(output_dir) / f"gradient_logo_seq_{_sanitize_filename(matched_id)}.png", dpi=300)

        figs[matched_id] = fig

    return figs


def select_top_expression_indices(input_tsv, species_masks, n=5):
    """Rank test-split sequences by *measured* MPRA expression (enrichment_cold /
    enrichment_warm columns in input_tsv), not model predictions, and take the top n
    per species for each condition. Only needs the TSV -- no model, no dataset
    object, no forward pass. (Mirrors saturation_mutagenesis.py's helper of the
    same name -- see its docstring for why row order lines up with a full-test-set
    cache with no extra bookkeeping.)

    species_masks:  {species_name: bool array}, e.g. build_species_masks(...) output
                     with "Other" excluded, aligned to the test split (summarize_species_masks).

    Returns (selections, cold_expr, warm_expr, ids):
      selections: {"cold": {species: [row_idx, ...]}, "warm": {species: [row_idx, ...]}},
                   row_idx values sorted by descending expression within each species.
      cold_expr, warm_expr, ids: (N,) arrays aligned to the test split's row order, for labeling.
    """
    _, split_rows = summarize_species_masks(input_tsv)
    cold_expr = np.array([float(row["enrichment_cold"]) for row in split_rows])
    warm_expr = np.array([float(row["enrichment_warm"]) for row in split_rows])
    ids = np.array([row["id"] for row in split_rows])

    selections = {}
    for condition_name, expr in (("cold", cold_expr), ("warm", warm_expr)):
        selections[condition_name] = {}
        for species, mask in species_masks.items():
            idxs = np.flatnonzero(mask)
            order = np.argsort(-expr[idxs])  # descending
            selections[condition_name][species] = idxs[order[:n]]

    return selections, cold_expr, warm_expr, ids


def plot_gradient_logos_by_species(attr_dict, species_masks, output_dir=None):
    """One logo-plot figure per condition in attr_dict, saved to its own file. Each
    figure stacks one row per group (Overall, then each species) -- never overlaid --
    with a shared x-axis (position) so groups line up for by-eye comparison.

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)}, same N/order
                     as species_masks.
    species_masks:  {species_name: array-like[bool]}, e.g. build_species_masks(...)
                     with "Other" excluded.
    output_dir:     if given, each figure is saved to output_dir / "gradient_logo_<condition_name>.png".

    Returns {condition_name: fig}.
    """
    groups = ["Overall"] + list(species_masks)
    figs = {}

    for condition_name, attr in attr_dict.items():
        fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.8 * len(groups)), sharex=True, squeeze=False)

        for row, group in enumerate(groups):
            mask = None if group == "Overall" else species_masks[group]
            logo_matrix, n = _mean_logo_matrix(attr, mask)
            ax = axes[row, 0]
            plot_logo(logo_matrix, ax=ax)  # plot_logo's standard A/C/G/T coloring
            ax.set_ylabel(f"{group}\n(n={n})", fontsize=9, rotation=0, ha="right", va="center")

        axes[0, 0].set_title(condition_name, fontsize=11, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel(ATTRIBUTION_YLABELS.get(condition_name, "Corrected saliency score"), fontsize=9)
        fig.tight_layout()

        if output_dir:
            fig.savefig(Path(output_dir) / f"gradient_logo_{condition_name}.png", dpi=300)

        figs[condition_name] = fig

    return figs


def plot_top_expression_gradient_logos(attr_dict, input_tsv, species_masks, n=5, output_dir=None,
                                        X=None, hypothetical=True):
    """Notebook entry point -- per-sequence analog of plot_gradient_logos_by_species():
    instead of averaging gradient-corrected saliency across sequences, plots one
    figure per individually selected sequence. Uses the same _standardized_ylim /
    _style_logo_ax formatting as saturation_mutagenesis.py's ISM per-sequence plots
    (black box, small ticks, faint grey 0.5-interval gridlines, y-axis standardized
    across that one sequence's own condition rows), so gradient and ISM figures read
    as one consistent visual style. Reuses an already-computed, cached gradient
    result -- no model and no re-run of raw_gradients()/apply_correction() needed;
    see load_grad_cache().

    For each species, selects the top-n sequences by measured enrichment_cold and
    enrichment_warm respectively (select_top_expression_indices, ground-truth MPRA
    values -- not model predictions) and makes one figure per selected sequence, one
    row per condition in attr_dict (just cold and warm by default -- pass
    add_condition_differences(grad_corrected) in if you also want the difference
    conditions as extra rows).

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)} -- grad_corrected
                     stores (N, L, 4) (saliency preserves the dataset's native
                     layout), so transpose first, e.g.
                     {name: g.transpose(1, 2) for name, g in grad_corrected.items()}
                     (see main()'s grad_corrected_for_plot).
    species_masks:  {species_name: array-like[bool]}, e.g. build_species_masks(...)
                     with "Other" excluded.
    output_dir:     if given, each figure is saved to
                     output_dir / "gradient_logo_seq_<rank_by>_<species>_rank<rank>_row<row_idx>.png".
    X:              the one-hot sequences, same (N, 4, L) layout as attr_dict -- transpose
                     the gradient cache's X the same way, e.g. X.transpose(1, 2). Only
                     needed when hypothetical=False.
    hypothetical:   attr_dict normally holds the full gradient over all 4 channels per
                     position (every base's saliency, not just the one actually in the
                     sequence -- "hypothetical", in ISM terminology). Passing
                     hypothetical=False here projects onto the observed base only
                     (grad * X, zeroing the other 3 channels per position -- i.e.
                     gradient x input) at plot time -- no need to recompute
                     grad_corrected -- for a single-letter-per-position track. Requires
                     X. Mirrors saturation_mutagenesis.py's
                     plot_top_expression_sequence_logos(hypothetical=...).

    Returns {key: fig}, key = "<rank_by>_<species>_rank<rank>_row<row_idx>".
    """
    if not hypothetical and X is None:
        raise ValueError("hypothetical=False requires passing X (the one-hot sequences, same (N, 4, L) layout as attr_dict)")

    selections, cold_expr, warm_expr, ids = select_top_expression_indices(input_tsv, species_masks, n=n)

    conditions = list(attr_dict)
    n_seqs = next(iter(attr_dict.values())).shape[0]
    if n_seqs != len(cold_expr):
        raise ValueError(
            f"attr_dict has {n_seqs} sequences but {input_tsv}'s test split has {len(cold_expr)} rows -- "
            "plot_top_expression_gradient_logos only works when the gradient cache covers the FULL test "
            "set (run_gradients(..., indices=None), the main() default), not a stratified subset."
        )

    def _logo_matrix(condition_name, row_idx):
        arr = attr_dict[condition_name][row_idx]
        logo_matrix = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        if not hypothetical:
            x_row = X[row_idx]
            x_row = x_row.detach().cpu().numpy() if hasattr(x_row, "detach") else np.asarray(x_row)
            logo_matrix = logo_matrix * x_row  # project onto the observed base (gradient x input)
        return logo_matrix

    plot_items = [
        (rank_by, species, rank, int(row_idx))
        for rank_by, per_species in selections.items()
        for species, row_idxs in per_species.items()
        for rank, row_idx in enumerate(row_idxs, start=1)
    ]

    figs = {}
    for item in plot_items:
        rank_by, species, rank, row_idx = item
        logo_matrices = [_logo_matrix(condition_name, row_idx) for condition_name in conditions]
        ylim = _standardized_ylim(logo_matrices)  # shared across this sequence's own condition rows only

        fig, axes = plt.subplots(len(conditions), 1, figsize=(9, 1.8 * len(conditions)), sharex=True, squeeze=False)

        for row, (condition_name, logo_matrix) in enumerate(zip(conditions, logo_matrices)):
            ax = axes[row, 0]
            plot_logo(logo_matrix, ax=ax)  # plot_logo's standard A/C/G/T coloring
            _style_logo_ax(ax, ylim)
            ax.set_ylabel(condition_name, fontsize=9, rotation=0, ha="right", va="center")

        title = (f"{species} — top-{rank} by {rank_by} expression  "
                 f"(id={ids[row_idx]}, cold={cold_expr[row_idx]:.2f}, warm={warm_expr[row_idx]:.2f})")
        axes[0, 0].set_title(title, fontsize=10, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel("Corrected saliency score", fontsize=9)
        fig.tight_layout()

        key = f"{rank_by}_{species}_rank{rank}_row{row_idx}"
        if output_dir:
            fig.savefig(Path(output_dir) / f"gradient_logo_seq_{key}.png", dpi=300)

        figs[key] = fig

    return figs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    config, checkpoint = load_config_from_checkpoint(checkpoint_path)

    if args.input_tsv is not None:
        config.data.input_tsv = args.input_tsv
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.pin_memory is not None:
        config.data.pin_memory = args.pin_memory
    if args.use_amp is not None:
        config.runtime.use_amp = args.use_amp
    if args.device is not None:
        config.runtime.device = args.device

    if not config.data.input_tsv:
        parser.error("data.input_tsv must be present in the checkpoint config or provided via --input_tsv")

    suffix = "" if args.use_adapters else "_no_adapters"
    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_gradient_logos_by_species{suffix}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (Path(args.cache_path).resolve() if args.cache_path is not None
                    else default_cache_path(checkpoint_path, use_adapters=args.use_adapters))

    if cache_path.exists() and not args.recompute:
        print(f"Loading cached gradient result from {cache_path}")
        X, organism_idx, y0, grad_raw, grad_corrected = load_grad_cache(cache_path)
    else:
        model, device, test_dataset = load_model_and_test_data(checkpoint_path, config, use_adapters=args.use_adapters)
        X, organism_idx, y0, grad_raw, grad_corrected = run_gradients(
            model, test_dataset, device, batch_size=args.gradient_batch_size,
        )
        save_grad_cache(cache_path, X, organism_idx, y0, grad_raw, grad_corrected)
        print(f"Saved gradient result to {cache_path}")

    # --- cheap, freely-rerunnable part ---
    grad_corrected = add_condition_differences(grad_corrected)

    # plot_gradient_logos_by_species expects (N, 4, L); saliency preserves the dataset's
    # (N, L, 4) one-hot layout, so transpose once here.
    grad_corrected_for_plot = {name: g.transpose(1, 2) for name, g in grad_corrected.items()}

    masks, _ = summarize_species_masks(config.data.input_tsv)  # test split, aligned to test_dataset's row order
    species_only = {key: val for key, val in masks.items() if key != "Other"}  # discard control seqs not linked to species

    figs = plot_gradient_logos_by_species(grad_corrected_for_plot, species_only, output_dir=output_dir)
    print(f"Saved {len(figs)} logo plots to {output_dir}")


if __name__ == "__main__":
    main()
