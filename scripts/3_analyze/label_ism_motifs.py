"""Label TF-MoDISco-discovered motifs onto a single sequence's own ISM attribution
logo (the per-sequence, hypothetical=False view plot_sequence_logos_by_id draws), so
you can see WHERE in that one sequence each known motif sits, rather than just the
pooled/averaged patterns TF-MoDISco itself outputs.

Pipeline, per condition (cold/warm):
  1. Build a motif catalog from TF-MoDISco_patterns.tsv.gz, keeping only patterns
     with a confident JASPAR match (see find_known_motifs.py's tomtom_tsv), same
     convention plot_tfmodisco_patterns.py's --pval_threshold uses.
  2. Scan the motif catalog against this ONE sequence's DNA (not its attribution)
     with FIMO (memelite.fimo.fimo) -- this finds every place in the sequence that
     resembles a catalog motif, independent of whether the model actually cares
     about that occurrence.
  3. Attach attribution "prominence" to each FIMO hit: the MEAN (not sum -- see
     below) of the sequence's own observed-base ISM attribution (X * hypothetical_
     attr, i.e. hypothetical=False) over the hit's span.
  4. Greedily suppress overlapping hits, keeping only the highest-prominence hit
     per overlapping cluster, then keep hits whose |prominence| clears
     --min_attribution, capped at the --top_n strongest per condition. This is
     what keeps the plot clean: the catalog is redundant (BZIP60/TGA4/CDF5/DOF3.6
     etc. all recognize similar ACGT-rich cores), so FIMO alone reports many
     different motif names all hitting the same underlying span.

Mean, not sum, because FIMO hits vary widely in width (8bp core motifs vs. 30-50bp
ones) -- summing would make a long, mildly-important motif look "stronger" than a
short, sharply-important one purely because it spans more positions.

Per tangermeme's own guidance (FIMO isn't in tangermeme, it's in memelite): FIMO
doesn't handle overhangs well, so it's for scanning whole sequences, not seqlets.
The scan itself uses a loose --fimo_pval_threshold (sequence resemblance only,
not "is this prominent") since a strict FIMO p-value can reject sequence hits
that still carry strong model attribution -- e.g. a BZIP60 hit here scores only
p=5e-4 by FIMO but carries 4.7 units of attribution, well above weaker hits at
p=1e-5. --min_attribution/--top_n do the actual "is this prominent" filtering.

Optional --meme_db extends the catalog with every motif straight from a MEME-format
database (e.g. JASPAR), bypassing TF-MoDISco/TOMTOM entirely. The TF-MoDISco catalog
above is scoped to patterns MoDISco discovered on ITS OWN discovery sequences --
a motif this one sequence contains but MoDISco never saw (or saw too rarely to
cluster into a pattern) can't appear in it. Scanning --meme_db directly against
this sequence's own DNA has no such blind spot; --min_attribution/--top_n still do
the prominence filtering, exactly as for the TF-MoDISco catalog.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from memelite.fimo import fimo
from tangermeme.annotate import read_meme
from tangermeme.plot import plot_logo

from alphagenome_encoder_ft import load_config_from_checkpoint
from saturation_mutagenesis import (
    compute_attributions,
    default_cache_path,
    find_row_indices_for_ids,
    load_ism_cache,
)

bases = list('ACGT')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--seq_id", required=True, help="e.g. Sb-26402_fwd")
    parser.add_argument("--patterns_tsv", required=True, help="TF-MoDISco_patterns.tsv.gz for the motif catalog.")
    parser.add_argument("--tomtom_tsv", required=True,
                         help="TF-MoDISco_tomtom_matches.tsv (vs. a known-motif database) for the same patterns_tsv.")
    parser.add_argument("--use_adapters", action=argparse.BooleanOptionalAction, default=False,
                         help="Use the ISM cache/sequence built WITH the 15bp Jores adapters (200bp construct). "
                              "Default False: the raw ~170bp insert only, no adapters -- pass --use_adapters to "
                              "switch to the adapter-padded cache/sequence instead.")
    parser.add_argument("--jaspar_pval_threshold", type=float, default=0.01,
                         help="Only include catalog patterns with a JASPAR match pval0 <= this. Default 0.01.")
    parser.add_argument("--meme_db", default=None,
                         help="Optional MEME-format motif database (e.g. "
                              "metadata/motif_databases/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt) to scan "
                              "directly, unioned onto the TF-MoDISco+JASPAR catalog. Catches motifs relevant to this "
                              "sequence that TF-MoDISco's own discovery run never saw. Default None (TF-MoDISco "
                              "catalog only, as before).")
    parser.add_argument("--fimo_pval_threshold", type=float, default=1e-2,
                         help="FIMO p-value threshold for calling a candidate sequence hit. Default 1e-2, looser "
                              "than memelite's own 1e-4 default -- this only screens for sequence resemblance, not "
                              "model relevance, and a strict threshold rejects real high-attribution hits that are "
                              "lower-affinity variants of the same motif (e.g. a second BZIP60 half-site here scores "
                              "only FIMO p=0.007 but still carries 0.48 mean attribution, on par with the strongest "
                              "hits). --min_attribution/--top_n do the real prominence filtering.")
    parser.add_argument("--min_attribution", type=float, default=0.1,
                         help="Only keep a FIMO hit if its |mean observed-base attribution per position| over the "
                              "hit's span is at least this. Mean, not sum, so short/sharp and long/diffuse hits are "
                              "compared fairly. Default 0.1.")
    parser.add_argument("--top_n", type=int, default=6,
                         help="Cap each condition to its top_n hits by |mean attribution|, after overlap "
                              "suppression, so the plot only shows the most prominent, non-redundant motifs. Default 6.")
    parser.add_argument("--output_path", default=None, help="Where to save the annotated figure (PNG).")
    return parser


def trim_by_cwm(ppm: np.ndarray, cwm: np.ndarray, trim_threshold: float = 0.3) -> np.ndarray:
    score = np.abs(cwm).sum(axis=1)
    pass_inds = np.where(score >= score.max() * trim_threshold)[0]
    return ppm[pass_inds.min(): pass_inds.max() + 1]


def build_motif_catalog(patterns_tsv: str, tomtom_tsv: str, jaspar_pval_threshold: float) -> dict[str, np.ndarray]:
    """{label: (4, len) trimmed PPM} for every pattern with a confident JASPAR match."""
    df = pd.read_csv(patterns_tsv, sep='\t')
    matches = pd.read_csv(tomtom_tsv, sep='\t').set_index(['condition', 'pattern'])

    motifs = {}
    for (condition, pattern), group in df.groupby(['condition', 'pattern']):
        match, pval = matches.loc[(condition, pattern), ['match0', 'pval0']]
        if pval > jaspar_pval_threshold:
            continue
        group = group.sort_values('pos')
        ppm = group[[f'ppm_{b}' for b in bases]].to_numpy()
        cwm = group[[f'cwm_{b}' for b in bases]].to_numpy()
        trimmed = trim_by_cwm(ppm, cwm)
        label = f'{condition}_{pattern} ({match})'
        motifs[label] = trimmed.T.astype('float32')  # (4, len)
    return motifs


def build_jaspar_catalog(meme_db: str) -> dict[str, np.ndarray]:
    """{label: (4, len) PWM} for every motif in a MEME-format database (e.g. a JASPAR
    CORE download), scanned as-is -- no TF-MoDISco/TOMTOM step. Labels are wrapped as
    'jaspar (<meme id>)' so _short_tf_name's '(...)' parsing still finds the TF name."""
    db = read_meme(meme_db)
    return {f'jaspar ({name})': pwm.numpy().astype('float32') for name, pwm in db.items()}


def scan_sequence_with_fimo(X_row: np.ndarray, motifs: dict[str, np.ndarray], pval_threshold: float) -> pd.DataFrame:
    """X_row: (4, L) one-hot. Returns every FIMO hit against the catalog, columns
    include motif_name/start/end/strand/score/p-value (memelite's own schema)."""
    hits = fimo(motifs, X_row[None], threshold=pval_threshold, dim=1)
    return hits[0] if hits else pd.DataFrame(columns=['motif_name', 'start', 'end', 'strand', 'score', 'p-value'])


def add_attribution_prominence(hits: pd.DataFrame, observed_attr_row: np.ndarray) -> pd.DataFrame:
    """observed_attr_row: (L,) this sequence's own hypothetical=False attribution,
    summed over the base axis (only the observed base is nonzero per position).
    Adds a signed `attribution` (mean per position over each hit's span, so hits of
    different widths are compared fairly) and `sign` column."""
    hits = hits.copy()
    hits["attribution"] = [
        observed_attr_row[start:end].mean() for start, end in zip(hits["start"], hits["end"])
    ]
    hits["sign"] = np.where(hits["attribution"] >= 0, "pos", "neg")
    return hits


def suppress_overlapping_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """The motif catalog is redundant (BZIP60/TGA4/CDF5/DOF3.6/... all recognize
    similar ACGT-rich cores), so FIMO commonly reports several different motif
    names all hitting the same span. Greedily keep only the highest-|attribution|
    hit within each cluster of overlapping spans."""
    ordered = hits.sort_values("attribution", key=np.abs, ascending=False)
    kept = []
    for _, hit in ordered.iterrows():
        if not any(hit["start"] < k["end"] and hit["end"] > k["start"] for k in kept):
            kept.append(hit)
    return pd.DataFrame(kept)


def filter_to_prominent_hits(hits: pd.DataFrame, min_attribution: float, top_n: int) -> pd.DataFrame:
    hits = suppress_overlapping_hits(hits)
    if hits.empty:
        return hits
    hits = hits[hits["attribution"].abs() >= min_attribution]
    hits = hits.sort_values("attribution", key=np.abs, ascending=False).head(top_n)
    return hits.sort_values("start").reset_index(drop=True)


def annotate_sequence(args) -> tuple[int, dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Returns (row_idx, {condition: (4, L) observed-base (hypothetical=False)
    attribution for row_idx -- only the observed base is nonzero per position},
    {condition: prominent FIMO hits DataFrame})."""
    config, _ = load_config_from_checkpoint(args.checkpoint_path)
    cache_path = default_cache_path(Path(args.checkpoint_path), use_adapters=args.use_adapters)
    print(f"Using ISM cache: {cache_path}")
    X, organism_idx, y0, y_hat = load_ism_cache(cache_path)

    id_to_row = find_row_indices_for_ids(config.data.input_tsv, [args.seq_id])
    row_idx = id_to_row[args.seq_id]
    X_row = X[row_idx].numpy()  # (4, L) one-hot, this sequence only

    attr_dict = compute_attributions(y0, y_hat, X)  # hypothetical=True, {cold, warm}: (N, 4, L)
    motifs = build_motif_catalog(args.patterns_tsv, args.tomtom_tsv, args.jaspar_pval_threshold)
    print(f"Motif catalog: {len(motifs)} patterns with JASPAR pval0 <= {args.jaspar_pval_threshold}")
    if args.meme_db:
        jaspar_motifs = build_jaspar_catalog(args.meme_db)
        print(f"JASPAR catalog: {len(jaspar_motifs)} motifs from {args.meme_db}")
        motifs = {**motifs, **jaspar_motifs}

    all_hits = scan_sequence_with_fimo(X_row, motifs, args.fimo_pval_threshold)
    print(f"FIMO: {len(all_hits)} raw sequence hit(s) in {args.seq_id} (before attribution filtering)")

    row_observed_attr, row_hits = {}, {}
    for condition, attr in attr_dict.items():
        observed_4L = (X[row_idx] * attr[row_idx]).numpy()  # (4, L), hypothetical=False -- what plot_logo draws
        row_observed_attr[condition] = observed_4L

        observed_1d = observed_4L.sum(axis=0)  # (L,), collapsed for span-sum scoring only (one nonzero base/position)
        hits = add_attribution_prominence(all_hits, observed_1d)
        hits = filter_to_prominent_hits(hits, args.min_attribution, args.top_n)
        row_hits[condition] = hits
        print(f"{condition}: {len(hits)} prominent hit(s) kept "
              f"(|attribution| >= {args.min_attribution}, top {args.top_n})")

    return row_idx, row_observed_attr, row_hits


def _short_tf_name(motif_name: str) -> str:
    """'cold_pos_pattern_14 (MA0967.1 BZIP60)' -> 'BZIP60' -- plot_logo's annotation
    rows are packed tightly, so the full pattern id + JASPAR accession (still
    available in the printed hit table) would be unreadable here."""
    inside = motif_name.rsplit('(', 1)[-1].rstrip(')')  # 'MA0967.1 BZIP60'
    return inside.split(' ', 1)[-1] if ' ' in inside else inside


def to_plot_logo_annotations(hits: pd.DataFrame) -> pd.DataFrame | None:
    if hits.empty:
        return None
    return pd.DataFrame({
        "motif_name": [_short_tf_name(m) for m in hits["motif_name"]],
        "start": hits["start"],
        "end": hits["end"],
        "score": hits["attribution"].abs(),
    })


def _shared_ylim_scalar(row_observed_attr: dict[str, np.ndarray], pad_frac: float = 0.3) -> float:
    """A single symmetric bound (not a (lo, hi) tuple) shared across both rows.

    plot_logo's own `ylim` param (despite its tuple-typed docstring) is only used
    as a symmetric scalar -- `ax.set_ylim(-ylim, ylim)` -- and, critically, its
    annotation bar/text stacking is positioned from `ax.get_ylim()` AT THE MOMENT
    plot_logo draws them. Setting ylim *after* calling plot_logo (the previous
    approach) left the bars positioned for a smaller, single-row range and then
    stretched the axis underneath them, which is what caused bars/text to overlap
    the sequence letters. Passing this scalar straight into plot_logo instead
    means the bar spacing is computed for the final, correct range from the start.

    pad_frac is deliberately generous (0.3, vs. a typical ~0.05 for a plain plot)
    because the annotation bars/labels stack downward from y=0 into the same
    space negative-attribution letters occupy -- the extra headroom is what
    keeps them from colliding with real letters instead of empty margin.
    """
    all_vals = np.concatenate([attr.sum(axis=0) for attr in row_observed_attr.values()])
    bound = max(abs(all_vals.min()), abs(all_vals.max()))
    return bound * (1 + pad_frac)


def plot_annotated_sequence(seq_id: str, row_observed_attr: dict[str, np.ndarray], row_hits: dict[str, pd.DataFrame],
                             output_path: str | None) -> "plt.Figure":
    import matplotlib.pyplot as plt

    conditions = list(row_observed_attr)
    seq_len = next(iter(row_observed_attr.values())).shape[-1]
    width = max(18, seq_len * 0.22)  # wider per-base spacing so letters aren't squished
    fig, axes = plt.subplots(len(conditions), 1, figsize=(width, 4.5 * len(conditions)), squeeze=False)

    ylim = _shared_ylim_scalar(row_observed_attr)

    for row, condition in enumerate(conditions):
        ax = axes[row, 0]
        annotations = to_plot_logo_annotations(row_hits[condition])
        plot_logo(row_observed_attr[condition], ax=ax, annotations=annotations, show_score=False, n_tracks=4, ylim=ylim)
        ax.set_ylim(-ylim, ylim)  # matches what plot_logo already set when annotations is non-empty;
                                  # only load-bearing for the edge case where a row has zero annotations
        ax.set_title(condition.capitalize(), fontsize=16, fontweight="bold")
        ax.set_ylabel("Δ predicted enrichment", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)

    axes[-1, 0].set_xlabel("Sequence Position", fontsize=15)
    fig.suptitle(f"{seq_id} ISM attribution", fontsize=18, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
        print(f"Saved {output_path}")
    return fig


def main() -> None:
    args = build_arg_parser().parse_args()
    row_idx, row_observed_attr, row_hits = annotate_sequence(args)

    for condition, hits in row_hits.items():
        if len(hits):
            print(f"\n{condition} prominent hits for {args.seq_id} (row {row_idx}):")
            print(hits[['start', 'end', 'strand', 'sign', 'motif_name', 'attribution', 'p-value']].to_string(index=False))

    output_path = args.output_path or f"ism_motifs_{args.seq_id}.png"
    plot_annotated_sequence(args.seq_id, row_observed_attr, row_hits, output_path)


if __name__ == "__main__":
    main()
