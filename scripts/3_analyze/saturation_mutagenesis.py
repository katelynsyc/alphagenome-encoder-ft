from __future__ import annotations

#from tangermeme
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
import seaborn; seaborn.set_style('white')
from tangermeme.plot import plot_logo
import matplotlib.pyplot as plt
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

def _mean_logo_matrix(attr, mask=None): #across the different sequences, what are the consensus motifs
    """(N, 4, L) ISM attribution -> (4, L), averaged across all N sequences
    (mask=None) or just the sequences where mask is True."""
    arr = attr.detach().cpu().numpy() if hasattr(attr, "detach") else np.asarray(attr)
    if mask is not None:
        arr = arr[np.asarray(mask)]
    return arr.mean(axis=0), arr.shape[0]

def plot_attribution_logos_by_species(attr_dict, species_masks, output_dir=None):
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
        fig, axes = plt.subplots(len(groups), 1, figsize=(9, 1.8 * len(groups)), sharex=True, squeeze=False)

        for row, group in enumerate(groups):
            mask = None if group == "Overall" else species_masks[group]
            logo_matrix, n = _mean_logo_matrix(attr, mask)
            ax = axes[row, 0]
            plot_logo(logo_matrix, ax=ax)  # plot_logo's standard A/C/G/T coloring
            ax.set_ylabel(f"{group}\n(n={n})", fontsize=9, rotation=0, ha="right", va="center")

        axes[0, 0].set_title(condition_name, fontsize=11, fontweight="bold")
        axes[-1, 0].set_xlabel("Sequence Position", fontsize=9)
        fig.supylabel(ATTRIBUTION_YLABELS.get(condition_name, "ISM attribution score"), fontsize=9)
        fig.tight_layout()

        if output_dir:
            fig.savefig(Path(output_dir) / f"ism_logo_{condition_name}.png", dpi=300)

        figs[condition_name] = fig

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


def compute_attributions(y0, y_hat, X):
    """Cheap: derives all 4 attribution variants from one run_ism() result.
    Re-run this freely -- it never touches the model or does another forward pass."""
    #take the per-condition attributions, need to apply processing, feed it the slices i want to
    attr_cold = process(y0[..., CONDITION["cold"]], y_hat[..., CONDITION["cold"]], X) #just take that one column
    attr_warm = process(y0[..., CONDITION["warm"]], y_hat[..., CONDITION["warm"]], X)

    attr_cold_not_hot = process(y0[..., CONDITION["cold"]] - y0[..., CONDITION["warm"]], y_hat[..., CONDITION["cold"]] - y_hat[..., CONDITION["warm"]], X)  # drives cold-high/warm-low
    attr_hot_not_cold = -attr_cold_not_hot #high activity in warm but low activity in cold

    return {
        "cold": attr_cold,
        "warm": attr_warm,
        "cold_not_warm": attr_cold_not_hot,   # drives cold-high/warm-low
        "warm_not_cold": attr_hot_not_cold,   # drives warm-high/cold-low
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
