from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
import seaborn; seaborn.set_style('white')
import matplotlib.pyplot as plt
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

    with torch.no_grad():
        y0 = model(X.to(device), organism_idx.to(device)).detach().cpu()  # (N, 5) reference predictions
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
