from __future__ import annotations

#from tangermeme
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
import seaborn; seaborn.set_style('white')
from tangermeme.plot import plot_logo
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import argparse
from pathlib import Path

import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    load_config_from_checkpoint,
    create_jores_splits,
    summarize_species_masks
)
from alphagenome_encoder_ft.mydata import _read_custom_tsv
"""
Run in notebook to plot specific ISM values from the cache with both the hypothetical and the existing attribution
import sys, os
os.chdir("/grid/koo/home/kachu/projects/alphagenome-encoder-ft")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts/3_analyze")

from pathlib import Path
from alphagenome_encoder_ft import load_config_from_checkpoint

checkpoint_path = Path("results/e898939e/df4406c4716cd2cf/stage2/best.pt")
config, _ = load_config_from_checkpoint(checkpoint_path)

target_ids = ["Sl-sh2115_rev", "At-12806_fwd"]  
base_output_dir = checkpoint_path.parent / "ism_and_gradient_logos_selected"
all_bases_dir = base_output_dir / "all_bases"          # hypothetical=True: all 4 (mean-centered) bases per position
observed_dir = base_output_dir / "observed_base_only"  # hypothetical=False: only the nucleotide actually in the sequence
all_bases_dir.mkdir(parents=True, exist_ok=True)
observed_dir.mkdir(parents=True, exist_ok=True)

# --- ISM (cached, no rerun) ---
from saturation_mutagenesis import (
    load_ism_cache, default_cache_path as ism_cache_path, compute_attributions, plot_sequence_logos_by_id,
)

X, organism_idx, y0, y_hat = load_ism_cache(ism_cache_path(checkpoint_path))
ism_attr_dict = compute_attributions(y0, y_hat, X)  # cold, warm only (diffs commented out)

ism_figs_all_bases = plot_sequence_logos_by_id(ism_attr_dict, config.data.input_tsv, target_ids, output_dir=all_bases_dir)
ism_figs_observed = plot_sequence_logos_by_id(
    ism_attr_dict, config.data.input_tsv, target_ids, output_dir=observed_dir, X=X, hypothetical=False,
)
print(f"Saved {len(ism_figs_all_bases)} ISM (all bases) + {len(ism_figs_observed)} ISM (observed base only) logo plots")

# --- Gradient / saliency (cached, no rerun) ---
from raw_gradient_correction import (
    load_grad_cache, default_cache_path as grad_cache_path, plot_gradient_logos_by_id,
)

Xg, organism_idx_g, y0_g, grad_raw, grad_corrected = load_grad_cache(grad_cache_path(checkpoint_path))
grad_corrected_for_plot = {name: g.transpose(1, 2) for name, g in grad_corrected.items()}  # (N,L,4) -> (N,4,L)
Xg_for_plot = Xg.transpose(1, 2)  # match attr_dict's (N,4,L) layout

grad_figs_all_bases = plot_gradient_logos_by_id(grad_corrected_for_plot, config.data.input_tsv, target_ids, output_dir=all_bases_dir)
grad_figs_observed = plot_gradient_logos_by_id(
    grad_corrected_for_plot, config.data.input_tsv, target_ids, output_dir=observed_dir, X=Xg_for_plot, hypothetical=False,
)
print(f"Saved {len(grad_figs_all_bases)} gradient (all bases) + {len(grad_figs_observed)} gradient (observed base only) logo plots")
"""

# Order must match JoresMPRADataset._targets (see mydata.py).
CONDITION_NAMES = ["cold", "dark", "light", "warm", "maize"] #want to take cold_idx=0, warm_idx=3
CONDITION = {"cold": 0, "dark": 1, "light": 2, "warm": 3, "maize": 4}

# y-axis quantity for each attr_dict entry produced by compute_attributions(), used to label the logo plots.
ATTRIBUTION_YLABELS = {
    "cold": "ISM attribution\n(Δ predicted cold enrichment)",
    "warm": "ISM attribution\n(Δ predicted warm enrichment)",
    "cold_not_warm": "ISM attribution\n(Δ predicted [cold − warm] enrichment)",
    "warm_not_cold": "ISM attribution\n(Δ predicted [warm − cold] enrichment)",
}

# Short per-row labels for the per-sequence figures (ATTRIBUTION_YLABELS is too long to
# use as a row label when every row is already the same sequence).
ATTRIBUTION_ROW_LABELS = {
    "cold": "cold",
    "warm": "warm",
    "cold_not_warm": "cold − warm",
    "warm_not_cold": "warm − cold",
}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an encoder-only AlphaGenome Jores checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache_path", type=str, default=None,
                         help="Where to save/load the raw ISM result (X, organism_idx, y0, y_hat). "
                              "Defaults to '<checkpoint>_ism_cache.pt' next to the checkpoint.")
    parser.add_argument("--recompute", action="store_true",
                         help="Ignore an existing cache and re-run saturation mutagenesis.")
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=True,
                         help="Include the 15bp Jores adapters on each side of the sequence when "
                              "building the ISM dataset (matches training). Pass --no-use_adapters "
                              "to run ISM on the raw ~170bp insert only, with no adapters added.")
    return parser


class TangermemeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X, organism_idx):
        return self.model(X.transpose(1, 2), organism_idx)  #converts back from (N, 4, L) after tangermeme does edits to (N, L, 4) 

 #process the data by using difference/mean-center/project logic from _attribution_score, then subset to split by species + the combination of high/low warm/cold
 #process has only linear operations of subtraction of mean and multiplying by a fixed mask (if using non-hypothetical) 
def process(y0, y_hat, X, hypothetical=True): #hypothetical keeps all 4 bases not just observed-base prediction
    attr = y_hat - y0[:, None, None]
    attr -= attr.mean(dim=1, keepdim=True)          # center across the 4 substitutions per position
    return attr if hypothetical else X * attr      # project onto observed base unless hypothetical

def _mean_logo_matrix(attr, mask=None): #across the different sequences, what are the consensus motifs, typically you don't want this and you want per-sequence
    """(N, 4, L) ISM attribution -> (4, L), averaged across all N sequences
    (mask=None) or just the sequences where mask is True."""
    arr = attr.detach().cpu().numpy() if hasattr(attr, "detach") else np.asarray(attr)
    if mask is not None:
        arr = arr[np.asarray(mask)]
    return arr.mean(axis=0), arr.shape[0]


def _standardized_ylim(logo_matrices, pad_frac=0.05):
    """Shared y-axis limits for a batch of (4, L) logo matrices. Uses the same
    stacked-height logic plot_logo itself uses internally -- the sum of the
    positive-valued channels at each position, and the sum of the negative-valued
    ones -- rather than each entry's raw min/max, since under hypothetical
    attributions several bases can stack at one position and a naive min/max
    would clip that row's glyphs. Rounded outward to the nearest 0.5 so the grey
    0.5-interval gridlines land on the frame edges.
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


def plot_attribution_logos_by_species(attr_dict, species_masks, output_dir=None): #this plots the average across the sequences, uses _mean_logo_matrix and not per-sequence
    """One logo-plot figure per attribution type in attr_dict, saved to its own
    file. Each figure stacks one row per group (Overall, then each species) --
    never overlaid -- with a shared x-axis (position) so groups line up for
    by-eye comparison.

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)}, same N/order
                     as species_masks. condition_name is used as both the plot
                     title and the output filename, so keep it filename-safe
                     (see compute_attributions).
    species_masks:  {species_name: array-like[bool]}, e.g. build_species_masks(...)
                     with "Other" excluded.
    output_dir:     if given, each figure is saved to output_dir / "ism_logo_<condition_name>.png".

    Returns {condition_name: fig}.
    """
    groups = ["Overall"] + list(species_masks)
    figs = {}

    for condition_name, attr in attr_dict.items():
        logo_matrices, ns = [], []
        for group in groups:
            mask = None if group == "Overall" else species_masks[group]
            logo_matrix, n = _mean_logo_matrix(attr, mask)
            logo_matrices.append(logo_matrix)
            ns.append(n)
        ylim = _standardized_ylim(logo_matrices)  # shared across this condition's Overall + species rows

        fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.8 * len(groups)), sharex=True, squeeze=False)

        for row, (group, logo_matrix, n) in enumerate(zip(groups, logo_matrices, ns)):
            ax = axes[row, 0]
            plot_logo(logo_matrix, ax=ax)  # plot_logo's standard A/C/G/T coloring
            _style_logo_ax(ax, ylim)
            ax.set_ylabel(f"{group}\n(n={n})", fontsize=9, rotation=0, ha="right", va="center")

        axes[0, 0].set_title(condition_name, fontsize=11, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel(ATTRIBUTION_YLABELS.get(condition_name, "ISM attribution score"), fontsize=9)
        fig.tight_layout()

        if output_dir:
            fig.savefig(Path(output_dir) / f"ism_logo_{condition_name}.png", dpi=300)

        figs[condition_name] = fig

    return figs


def _sanitize_filename(text):
    """e.g. 'At-12807(PP)_fwd' -> 'At-12807_PP__fwd' -- some ids carry characters
    that are awkward in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def find_row_indices_for_ids(input_tsv, query_ids):
    """Resolve arbitrary sequence ids to test-split row indices, aligned to the same
    row order as run_ism() / summarize_species_masks().

    Each query id is matched exactly first; if that fails, it's matched as a prefix
    against "<query_id>_..."-style ids (the dataset's id column carries an explicit
    strand suffix, e.g. "_fwd"/"_rev") -- every id in the test split that matches
    either rule is returned, so an unsuffixed query like "At-12806" expands to both
    strand variants if both are present in the test split, rather than silently
    guessing one.

    Returns {matched_id: row_idx}. Raises KeyError listing any query_ids with zero
    matches (e.g. because they're only in the train/val split, not test -- ISM is
    only cached for the test split).
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
            f"id(s) not found in {input_tsv}'s test split (ISM is only cached for the test split -- "
            f"check modelling_data_tamsACR.tsv's 'set' column): {unmatched}"
        )
    return matches


def plot_sequence_logos_by_id(attr_dict, input_tsv, ids, output_dir=None, X=None, hypothetical=True):
    """Notebook entry point -- plot ISM logos for specific, named sequences (rather
    than a top-n selection; see plot_top_expression_sequence_logos for that). ids are
    looked up by the dataset's 'id' column via find_row_indices_for_ids -- an
    unsuffixed id like "At-12806" expands to every strand variant present in the
    test split.

    Same 4-row-per-figure (or however many conditions attr_dict has) layout, styling,
    and per-sequence y-axis standardization as plot_top_expression_sequence_logos.

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)} from
                     compute_attributions() on the FULL test-set ISM cache
                     (run_ism(..., indices=None), the main() default).
    ids:            sequence id(s) to plot, e.g. ["Sl-sh2115_rev", "At-12806"].
    output_dir:     if given, each figure is saved to output_dir / "ism_logo_seq_<matched_id>.png".
    X, hypothetical: see plot_top_expression_sequence_logos -- hypothetical=False
                     projects onto the observed base only (requires X).

    Returns {matched_id: fig}.
    """
    if not hypothetical and X is None:
        raise ValueError("hypothetical=False requires passing X (the one-hot sequences used to build attr_dict)")

    id_to_row = find_row_indices_for_ids(input_tsv, ids)

    _, split_rows = summarize_species_masks(input_tsv)
    cold_expr = {row["id"]: float(row["enrichment_cold"]) for row in split_rows}
    warm_expr = {row["id"]: float(row["enrichment_warm"]) for row in split_rows}

    conditions = list(attr_dict)
    n_seqs = next(iter(attr_dict.values())).shape[0]
    if n_seqs != len(split_rows):
        raise ValueError(
            f"attr_dict has {n_seqs} sequences but {input_tsv}'s test split has {len(split_rows)} rows -- "
            "plot_sequence_logos_by_id only works when the ISM cache covers the FULL test set "
            "(run_ism(..., indices=None), the main() default), not a stratified subset."
        )

    def _logo_matrix(condition_name, row_idx):
        arr = attr_dict[condition_name][row_idx]
        logo_matrix = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        if not hypothetical:
            x_row = X[row_idx]
            x_row = x_row.detach().cpu().numpy() if hasattr(x_row, "detach") else np.asarray(x_row)
            logo_matrix = logo_matrix * x_row  # project onto the observed base
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
            ax.set_ylabel(ATTRIBUTION_ROW_LABELS.get(condition_name, condition_name),
                           fontsize=9, rotation=0, ha="right", va="center")

        title = f"{matched_id}  (cold={cold_expr[matched_id]:.2f}, warm={warm_expr[matched_id]:.2f})"
        axes[0, 0].set_title(title, fontsize=10, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel("ISM attribution score", fontsize=9)
        fig.tight_layout()

        if output_dir:
            fig.savefig(Path(output_dir) / f"ism_logo_seq_{_sanitize_filename(matched_id)}.png", dpi=300)

        figs[matched_id] = fig

    return figs


def select_top_expression_indices(input_tsv, species_masks, n=5):
    """Rank test-split sequences by *measured* MPRA expression (enrichment_cold /
    enrichment_warm columns in input_tsv), not model predictions, and take the top n
    per species for each condition.

    Only needs the TSV -- no model, no dataset object, no forward pass. Relies on
    summarize_species_masks's row order matching create_jores_splits's test split
    (see its docstring), which is also the row order run_ism() produces when called
    with indices=None (the default in main()), so this lines up with a full-test-set
    ISM cache with no extra bookkeeping.

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


def plot_top_expression_sequence_logos(attr_dict, input_tsv, species_masks, n=5, output_dir=None,
                                        X=None, hypothetical=True):
    """Notebook entry point -- reuses an already-computed attr_dict (from
    compute_attributions(y0, y_hat, X) on a cached, full-test-set ISM result) to plot
    individual per-sequence logos, instead of the species-averaged logos from
    plot_attribution_logos_by_species(). No model and no re-run of saturation_mutagenesis
    needed; see load_ism_cache().

    For each species, selects the top-n sequences by measured enrichment_cold and
    enrichment_warm respectively (see select_top_expression_indices) and makes one
    figure per selected sequence: 4 stacked rows (cold, warm, cold_not_warm,
    warm_not_cold), each that single sequence's own ISM logo -- no averaging.

    attr_dict:      {condition_name: torch.Tensor of shape (N, 4, L)}, same N/row order
                     as species_masks -- i.e. compute_attributions() run on the FULL
                     test-set ISM cache (run_ism(..., indices=None), the main() default).
    species_masks:  {species_name: array-like[bool]}, e.g. build_species_masks(...)
                     with "Other" excluded.
    output_dir:     if given, each figure is saved to
                     output_dir / "ism_logo_seq_<rank_by>_<species>_rank<rank>_row<row_idx>.png".
    X:              the one-hot sequences (N, 4, L) used to build attr_dict, e.g. from
                     load_ism_cache(). Only needed when hypothetical=False.
    hypothetical:   attr_dict is normally built with compute_attributions()'s default
                     hypothetical=True, which keeps all 4 (mean-centered) bases per
                     position -- individual sequences render busier than the
                     species-averaged plots because there's no averaging to cancel out
                     each sequence's non-motif "background jitter", and because the
                     reference base's centered value is generally nonzero too (it's
                     -mean of the other three, not 0). Passing hypothetical=False here
                     projects onto the observed base only (X * attr, zeroing the other 3
                     channels per position) at plot time -- no need to recompute
                     attr_dict -- for a single-letter-per-position track. Requires X.

    Each figure's y-axis is standardized across that one sequence's own condition
    rows only, not across different sequences -- sharing one scale across every
    selected sequence would squash the smaller-magnitude ones flat.
    """
    if not hypothetical and X is None:
        raise ValueError("hypothetical=False requires passing X (the one-hot sequences used to build attr_dict)")

    selections, cold_expr, warm_expr, ids = select_top_expression_indices(input_tsv, species_masks, n=n)

    conditions = list(attr_dict)
    n_seqs = next(iter(attr_dict.values())).shape[0]
    if n_seqs != len(cold_expr):
        raise ValueError(
            f"attr_dict has {n_seqs} sequences but {input_tsv}'s test split has {len(cold_expr)} rows -- "
            "plot_top_expression_sequence_logos only works when the ISM cache covers the FULL test set "
            "(run_ism(..., indices=None), the main() default), not a stratified subset."
        )

    def _logo_matrix(condition_name, row_idx):
        arr = attr_dict[condition_name][row_idx]
        logo_matrix = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        if not hypothetical:
            x_row = X[row_idx]
            x_row = x_row.detach().cpu().numpy() if hasattr(x_row, "detach") else np.asarray(x_row)
            logo_matrix = logo_matrix * x_row  # project onto the observed base
        return logo_matrix

    # Flatten the (rank_by, species, rank, row_idx) selection into a plotting order.
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
            ax.set_ylabel(ATTRIBUTION_ROW_LABELS.get(condition_name, condition_name),
                           fontsize=9, rotation=0, ha="right", va="center")

        title = (f"{species} — top-{rank} by {rank_by} expression  "
                 f"(id={ids[row_idx]}, cold={cold_expr[row_idx]:.2f}, warm={warm_expr[row_idx]:.2f})")
        axes[0, 0].set_title(title, fontsize=10, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel("ISM attribution score", fontsize=9)
        fig.tight_layout()

        key = f"{rank_by}_{species}_rank{rank}_row{row_idx}"
        if output_dir:
            fig.savefig(Path(output_dir) / f"ism_logo_seq_{key}.png", dpi=300)

        figs[key] = fig

    return figs


def load_model_and_test_data(checkpoint_path, config, use_adapters=True):
    """Cheap, one-time setup: model + wrapper + test_dataset. In a notebook, call
    this once (after resolving `config`, e.g. via load_config_from_checkpoint)
    and keep the results around -- nothing here depends on ISM having run yet.

    use_adapters=False builds the raw ~170bp insert with no adapters concatenated on
    either side (see JoresMPRADataset), instead of the adapter-padded training construct."""
    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=device) #loads trained weights
    wrapped = TangermemeWrapper(model)

    _, _, test_dataset = create_jores_splits( #held out set for testing
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
    return model, wrapped, test_dataset


def run_ism(wrapped, test_dataset, indices=None):
    """The expensive step -- run this ONCE per notebook session and hold on to
    X/organism_idx/y0/y_hat. Everything downstream (compute_attributions(),
    species masks, plotting) is cheap tensor math and can be re-run freely
    without calling this again."""
    if indices is None:
        indices = range(len(test_dataset))

    #stack the tensors from the dataset instead of getting one example at a time
    X = torch.stack([test_dataset[i][0] for i in indices])   # (N, L, 4)
    X = X.transpose(1, 2) #saturation mutagenesis expects X = (N, 4 nucleotides, L) but my channels come out as (N, L, 4) so i need to transpose first
    organism_idx = torch.zeros(X.shape[0], dtype=torch.long)     # confirmed unused because we only use encoder for this

    #run the expensive peturbation passes once by just taking raw_outputs and then slicing myself
    y0, y_hat = saturation_mutagenesis(wrapped, X, args=(organism_idx,), raw_outputs=True) #returns raw model predictions for ref seqs & peturbed seqs
    print(f"y0 dims: {tuple(y0.shape)}")   # (N, 5)  -- [cold, dark, light, warm, maize] on the reference sequences
    print(f"y_hat dims: {tuple(y_hat.shape)}") # (N, 4, L, 5)  -- same 5 conditions for every single-base substitution

    return X, organism_idx, y0, y_hat


def default_cache_path(checkpoint_path, use_adapters=True):
    suffix = "" if use_adapters else "_no_adapters"
    return checkpoint_path.parent / f"{checkpoint_path.stem}_ism_cache{suffix}.pt"


def save_ism_cache(cache_path, X, organism_idx, y0, y_hat):
    torch.save({"X": X, "organism_idx": organism_idx, "y0": y0, "y_hat": y_hat}, cache_path)


def load_ism_cache(cache_path):
    cached = torch.load(cache_path, map_location="cpu")
    return cached["X"], cached["organism_idx"], cached["y0"], cached["y_hat"]


def combine_ism_caches(cache_paths, output_path):
    """Concatenate several load_ism_cache()-compatible caches along the batch (N)
    dimension into one pooled cache, e.g. to pool multiple individually-cached
    sequences into one batch for tfmodisco_ism.py (which wants seqlets pooled
    across examples, not one sequence at a time)."""
    Xs, organism_idxs, y0s, y_hats = zip(*(load_ism_cache(path) for path in cache_paths))
    save_ism_cache(
        output_path, torch.cat(Xs, dim=0), torch.cat(organism_idxs, dim=0),
        torch.cat(y0s, dim=0), torch.cat(y_hats, dim=0),
    )


def compute_attributions(y0, y_hat, X):
    """Cheap: derives the attribution variants from one run_ism() result.
    Re-run this freely -- it never touches the model or does another forward pass."""
    #take the per-condition attributions, need to apply processing, feed it the slices i want to
    attr_cold = process(y0[..., CONDITION["cold"]], y_hat[..., CONDITION["cold"]], X) #just take that one column
    attr_warm = process(y0[..., CONDITION["warm"]], y_hat[..., CONDITION["warm"]], X)

    # cold/warm-difference conditions -- commented out for now, just plotting cold and warm directly.
    # attr_cold_not_hot = process(y0[..., CONDITION["cold"]] - y0[..., CONDITION["warm"]], y_hat[..., CONDITION["cold"]] - y_hat[..., CONDITION["warm"]], X)  # drives cold-high/warm-low
    # attr_hot_not_cold = -attr_cold_not_hot #high activity in warm but low activity in cold

    return {
        "cold": attr_cold,
        "warm": attr_warm,
        # "cold_not_warm": attr_cold_not_hot,   # drives cold-high/warm-low
        # "warm_not_cold": attr_hot_not_cold,   # drives warm-high/cold-low
    }


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
    default_output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_ism_logos_by_species{suffix}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else default_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (Path(args.cache_path).resolve() if args.cache_path is not None
                  else default_cache_path(checkpoint_path, use_adapters=args.use_adapters))

    if cache_path.exists() and not args.recompute:
        print(f"Loading cached ISM result from {cache_path}")
        X, organism_idx, y0, y_hat = load_ism_cache(cache_path)
    else:
        model, wrapped, test_dataset = load_model_and_test_data(checkpoint_path, config, use_adapters=args.use_adapters)
        X, organism_idx, y0, y_hat = run_ism(wrapped, test_dataset)
        save_ism_cache(cache_path, X, organism_idx, y0, y_hat)
        print(f"Saved ISM result to {cache_path}")

    # --- cheap, freely-rerunnable part ---
    attr_dict = compute_attributions(y0, y_hat, X)

    masks, _ = summarize_species_masks(config.data.input_tsv)  # test split, aligned to test_dataset's row order
    species_only = {key: val for key, val in masks.items() if key != "Other"} #discard control seqs that aren't linked to species

    figs = plot_attribution_logos_by_species(attr_dict, species_only, output_dir=output_dir)
    print(f"Saved {len(figs)} logo plots to {output_dir}")



if __name__ == "__main__":
    main()
