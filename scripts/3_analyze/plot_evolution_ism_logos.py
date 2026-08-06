from __future__ import annotations

"""Run warm-condition ISM (saturation_mutagenesis) on the initial (pre-evolution)
and final (best evolved) sequences of one or more ism_greedy_evolution.py runs, and
plot hypothetical (all 4 bases) + observed-base logos for each -- 4 plots per
summary json (start/end x hypothetical/observed) -- plus a 5th, FIMO-annotated
version of the observed-base logo (reusing label_ism_motifs.py's motif-catalog +
FIMO-scan + prominence-filter pipeline), so 6 plots per summary json / 12 total for
the two selected CREs.

Unlike saturation_mutagenesis.py's cached-test-set pipeline, these sequences don't
exist in input_tsv (they're evolved, off-dataset), so ISM is run fresh here via
tangermeme.saturation_mutagenesis, the same call ism_greedy_evolution.py itself uses
each round. No adapters are added -- <id>_summary.json's initial_sequence/
final_sequence are always the bare insert (the adapters, if any were used during
evolution, are stripped back out before saving -- see evolve_sequence()'s
final_construct[start:end]), and start/end default to the whole sequence here too,
so mutation scans every insert position.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from tangermeme.plot import plot_logo
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from alphagenome_encoder_ft import AlphaGenomeEncoderModel, load_config_from_checkpoint
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

# saturation_mutagenesis.py and label_ism_motifs.py live alongside this script -- see
# saturation_mutagenesis.py's own docstring notebook snippet, and ism_greedy_evolution.py,
# for the same sibling-import pattern.
from saturation_mutagenesis import CONDITION, TangermemeWrapper, process, save_ism_cache, combine_ism_caches
from label_ism_motifs import (
    build_motif_catalog,
    build_jaspar_catalog,
    scan_sequence_with_fimo,
    add_attribution_prominence,
    filter_to_prominent_hits,
    to_plot_logo_annotations,
    _shared_ylim_scalar,
)

STAGES = [("start", "initial_sequence"), ("end", "final_sequence")]
KINDS = ["hypothetical", "observed"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--summary_json", required=True, nargs="+",
                         help="One or more <id>_summary.json files written by ism_greedy_evolution.py "
                              "(e.g. best_greedy_evolution/selected_path_example/*_summary.json).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ism_batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patterns_tsv", default=None,
                         help="TF-MoDISco_patterns.tsv.gz for the motif catalog (see tfmodisco_ism.py). "
                              "Default: <checkpoint_dir>/TF-MoDISco_ism_best_ism_cache_no_adapters/"
                              "TF-MoDISco_patterns.tsv.gz -- the no-adapters catalog, matching these "
                              "sequences' own no-adapters ISM.")
    parser.add_argument("--tomtom_tsv", default=None,
                         help="TF-MoDISco_tomtom_matches.tsv for the same patterns_tsv (see find_known_motifs.py). "
                              "Default: alongside --patterns_tsv's default.")
    parser.add_argument("--jaspar_pval_threshold", type=float, default=0.01,
                         help="Only include catalog patterns with a JASPAR match pval0 <= this. Default 0.01, "
                              "matching label_ism_motifs.py/plot_tfmodisco_patterns.py.")
    parser.add_argument("--meme_db", default=None,
                         help="Optional MEME-format motif database (e.g. "
                              "metadata/motif_databases/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt) to scan "
                              "directly, unioned onto the TF-MoDISco+JASPAR catalog -- see label_ism_motifs.py's "
                              "docstring. Evolved sequences are off-dataset, so a motif they contain/introduce that "
                              "TF-MoDISco's original discovery run never saw can't be in --patterns_tsv; this "
                              "catches those too. Default None (TF-MoDISco catalog only, as before).")
    parser.add_argument("--fimo_pval_threshold", type=float, default=1e-2,
                         help="FIMO p-value threshold for calling a candidate sequence hit. Default 1e-2, "
                              "same as label_ism_motifs.py -- see its docstring for why this is looser than "
                              "memelite's own 1e-4 default.")
    parser.add_argument("--min_attribution_frac", type=float, default=0.3,
                         help="Only keep a FIMO hit if its |mean observed-base attribution per position| over "
                              "the hit's span is at least this fraction of THIS sequence's own tallest letter "
                              "(max |observed attribution| at any single position) -- relative per-sequence "
                              "prominence rather than a fixed absolute cutoff, since the two CREs/stages have "
                              "very different attribution scales (see the start-vs-end no-adapters logos). "
                              "Default 0.3.")
    parser.add_argument("--top_n", type=int, default=3,
                         help="Cap each sequence to its top_n hits by |mean attribution|, after overlap "
                              "suppression. Default 3 (a handful of the most prominent motifs, not every "
                              "FIMO sequence-resemblance hit).")
    parser.add_argument("--cache_dir", default=None,
                         help="Directory to write load_ism_cache()-compatible .pt caches (all 5 conditions, "
                              "not just warm) for tfmodisco_ism.py to consume downstream: one per (id, stage), "
                              "plus pooled 'before' (every stage=start sequence) and 'after' (every stage=end "
                              "sequence) caches across all --summary_json inputs. Default: <output_dir>/ism_caches.")
    return parser


def tight_ylim(matrices: list[np.ndarray], pad_frac: float = 0.05) -> tuple[float, float]:
    """Same stacked-height logic as saturation_mutagenesis.py's _standardized_ylim,
    but NOT rounded outward to the nearest 0.5 -- cropped tight to the actual
    attribution extent (plus a small pad) instead of the full gridline interval."""
    pos_max = max(np.sum(np.where(m > 0, m, 0), axis=0).max() for m in matrices)
    neg_min = min(np.sum(np.where(m < 0, m, 0), axis=0).min() for m in matrices)
    pad = pad_frac * (pos_max - neg_min) if pos_max > neg_min else 0.05
    return float(neg_min - pad), float(pos_max + pad)


def style_ax(ax, ylim: tuple[float, float]) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="y", linestyle=":", color="grey", alpha=0.6)
    ax.tick_params(left=True, bottom=True)


def run_ism_warm(wrapped: TangermemeWrapper, sequence: str, batch_size: int):
    """Fresh (uncached) ISM on a single bare-insert sequence. Returns (hyp, obs,
    X_row, X, organism_idx, y0, y_hat): hyp/obs (warm-only, for plotting) each
    (4, L) numpy, X_row the (4, L) one-hot numpy (for the FIMO scan), and
    X/organism_idx/y0/y_hat the raw, ALL-CONDITION tensors as saturation_mutagenesis
    returned them -- kept around so the caller can save_ism_cache() them for
    tfmodisco_ism.py, which (via compute_attributions) needs every condition
    present even if only "warm" is requested downstream."""
    onehot = sequence_to_onehot(sequence).astype("float32")  # (L, 4)
    X = torch.from_numpy(onehot).transpose(0, 1).unsqueeze(0)  # (1, 4, L)
    organism_idx = torch.zeros(1, dtype=torch.long)  # unused by the encoder-only forward pass

    y0, y_hat = saturation_mutagenesis(
        wrapped, X, args=(organism_idx,), raw_outputs=True, batch_size=batch_size,
    )  # y0: (1, 5), y_hat: (1, 4, L, 5)

    warm_idx = CONDITION["warm"]
    y0_warm = y0[:, warm_idx]          # (1,)
    y_hat_warm = y_hat[..., warm_idx]  # (1, 4, L)

    hyp = process(y0_warm, y_hat_warm, X, hypothetical=True)[0]   # (4, L)
    obs = process(y0_warm, y_hat_warm, X, hypothetical=False)[0]  # (4, L)
    return hyp.detach().cpu().numpy(), obs.detach().cpu().numpy(), X[0].numpy(), X, organism_idx, y0, y_hat


def plot_and_save(matrix: np.ndarray, title: str, out_path: Path) -> None:
    ylim = tight_ylim([matrix])
    fig, ax = plt.subplots(figsize=(9, 1.8))
    plot_logo(matrix, ax=ax)
    style_ax(ax, ylim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Sequence Position", fontsize=9)
    ax.set_ylabel("ISM attribution\n(warm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_annotated_and_save(
    obs: np.ndarray, X_row: np.ndarray, motifs: dict[str, np.ndarray], title: str, out_path: Path,
    fimo_pval_threshold: float, min_attribution_frac: float, top_n: int,
) -> None:
    """FIMO-scan X_row against the motif catalog, attach obs's own prominence, and
    plot the observed-base logo with the surviving hits annotated -- same pipeline as
    label_ism_motifs.py's annotate_sequence/plot_annotated_sequence, just for one
    condition (warm) and one externally-supplied (obs, X_row) pair instead of a
    cached test-set row.

    Prominence is thresholded relative to THIS sequence's own tallest letter (max
    |observed attribution| at any single position), not a fixed absolute value --
    the four sequences (2 CREs x start/end) span very different attribution scales
    (e.g. Zm-1631_rev start peaks around +/-0.4, its evolved end around +/-1.2), so a
    single absolute cutoff either misses everything on the small-scale sequences or
    lets through noise on the large-scale ones."""
    hits = scan_sequence_with_fimo(X_row, motifs, fimo_pval_threshold)
    print(f"FIMO: {len(hits)} raw sequence hit(s) (before attribution filtering)")

    observed_1d = obs.sum(axis=0)  # (L,), one nonzero base per position
    tallest_letter = float(np.abs(observed_1d).max())
    min_attribution = min_attribution_frac * tallest_letter

    hits = add_attribution_prominence(hits, observed_1d)
    hits = filter_to_prominent_hits(hits, min_attribution, top_n)
    print(f"{len(hits)} prominent hit(s) kept (|attribution| >= {min_attribution_frac} x tallest letter "
          f"{tallest_letter:.3f} = {min_attribution:.3f}, top {top_n})")
    if len(hits):
        print(hits[['start', 'end', 'strand', 'sign', 'motif_name', 'attribution', 'p-value']].to_string(index=False))

    annotations = to_plot_logo_annotations(hits)
    ylim = _shared_ylim_scalar({"warm": obs})

    fig, ax = plt.subplots(figsize=(max(18, obs.shape[-1] * 0.22), 4.5))
    # annot_cmap: a flat list of "black" so every annotation row (label + bar) is
    # black regardless of which row plot_logo's collision-avoidance packing assigns
    # it to -- the default "Set1" colormap ties color to that packing row, not to
    # the hit's attribution sign, which reads as (and isn't) a pos/neg indicator.
    plot_logo(obs, ax=ax, annotations=annotations, show_score=False, n_tracks=4, ylim=ylim,
              annot_cmap=["black"] * 10)
    ax.set_ylim(-ylim, ylim)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sequence Position", fontsize=13)
    ax.set_ylabel("Δ predicted enrichment", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    args = build_arg_parser().parse_args()

    checkpoint_path = Path(args.checkpoint_path).resolve()
    default_modisco_dir = checkpoint_path.parent / "TF-MoDISco_ism_best_ism_cache_no_adapters"
    patterns_tsv = args.patterns_tsv or str(default_modisco_dir / "TF-MoDISco_patterns.tsv.gz")
    tomtom_tsv = args.tomtom_tsv or str(default_modisco_dir / "TF-MoDISco_tomtom_matches.tsv")

    config, _ = load_config_from_checkpoint(checkpoint_path)
    device = torch.device(args.device or config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = AlphaGenomeEncoderModel.from_checkpoint(checkpoint_path, device=device)
    wrapped = TangermemeWrapper(model)

    motifs = build_motif_catalog(patterns_tsv, tomtom_tsv, args.jaspar_pval_threshold)
    print(f"Motif catalog: {len(motifs)} patterns with JASPAR pval0 <= {args.jaspar_pval_threshold} "
          f"(from {patterns_tsv})")
    if args.meme_db:
        jaspar_motifs = build_jaspar_catalog(args.meme_db)
        print(f"JASPAR catalog: {len(jaspar_motifs)} motifs from {args.meme_db}")
        motifs = {**motifs, **jaspar_motifs}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "ism_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths_by_stage = {"start": [], "end": []}  # for the pooled before/after caches

    for summary_path in args.summary_json:
        summary = json.loads(Path(summary_path).read_text())
        seq_id = summary["id"]

        for stage, seq_key in STAGES:
            sequence = summary[seq_key]
            warm_level = summary[f"{'initial' if stage == 'start' else 'final'}_levels"]["warm"]
            hyp, obs, X_row, X, organism_idx, y0, y_hat = run_ism_warm(wrapped, sequence, args.ism_batch_size)

            for kind, matrix in zip(KINDS, [hyp, obs]):
                out_path = out_dir / f"ism_logo_{seq_id}_{stage}_{kind}_warm.png"
                title = f"{seq_id} ({stage}, warm={warm_level:.2f}, {kind}, no adapters)"
                plot_and_save(matrix, title, out_path)

            annotated_out_path = out_dir / f"ism_motifs_{seq_id}_{stage}_warm.png"
            annotated_title = f"{seq_id} ({stage}, warm={warm_level:.2f}) ISM attribution, no adapters"
            plot_annotated_and_save(
                obs, X_row, motifs, annotated_title, annotated_out_path,
                args.fimo_pval_threshold, args.min_attribution_frac, args.top_n,
            )

            # Single-sequence cache (N=1) -- lets tfmodisco_ism.py run on this exact
            # sequence alone, separated by CRE, in addition to the pooled before/after
            # caches below.
            cache_path = cache_dir / f"ism_cache_{seq_id}_{stage}.pt"
            save_ism_cache(cache_path, X, organism_idx, y0, y_hat)
            print(f"Saved {cache_path}")
            cache_paths_by_stage[stage].append(cache_path)

    # Pooled caches: "before" = every stage=start sequence across all --summary_json
    # inputs batched together (N=len(summary_json)), "after" = every stage=end
    # sequence -- for motifs shared/emerging across BOTH CREs, separate from each
    # individual sequence's own cache above.
    pooled_names = {"start": "before", "end": "after"}
    for stage, pooled_name in pooled_names.items():
        pooled_path = cache_dir / f"ism_cache_{pooled_name}.pt"
        combine_ism_caches(cache_paths_by_stage[stage], pooled_path)
        print(f"Saved {pooled_path} (pooled {stage}, N={len(cache_paths_by_stage[stage])})")


if __name__ == "__main__":
    main()
