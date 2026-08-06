from __future__ import annotations

### Identify motifs in cold/warm ISM attributions using TF-MoDISco ###
# Load the required modules:
import argparse
from pathlib import Path

import h5py
import modiscolite
import numpy as np
import pandas as pd

from saturation_mutagenesis import load_ism_cache, compute_attributions

DEFAULT_CONDITIONS = ['cold', 'warm']


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TF-MoDISco on cached ISM attributions for cold/warm.")
    parser.add_argument("--cache_path", required=True,
                         help="ISM cache .pt written by saturation_mutagenesis.py (save_ism_cache).")
    parser.add_argument("--output_dir", default="TF-MoDISco_ism",
                         help="Directory to write per-condition .h5 files and the combined .tsv.gz.")
    parser.add_argument("--max_seqlets", type=int, default=20000,
                         help="max_seqlets_per_metacluster passed to TFMoDISco.")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                         help="Which of compute_attributions()'s conditions to run TF-MoDISco on. "
                              f"Default: {DEFAULT_CONDITIONS} (both). Pass e.g. --conditions warm to "
                              "run only the warm condition.")
    return parser


def to_length_major(attr):
    # ISM cache tensors are (N, 4, L); modiscolite indexes one_hot as
    # one_hot[example][position][base] (see core.TrackSet: `len(one_hot[0])`
    # is used as the sequence length), so this must be (N, L, 4), not (N, 4, L).
    return attr.detach().cpu().numpy().transpose(0, 2, 1).astype("float32")


def main() -> None:
    args = build_arg_parser().parse_args()

    # Set up:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # parents=True since, unlike the original, out_dir isn't nested under an existing data_dir

    # Load one-hot encoded sequences + raw ISM predictions from the cache
    # (X, organism_idx, y0, y_hat), instead of loading DeepLIFT's ohe_seqs.npz:
    X, organism_idx, y0, y_hat = load_ism_cache(args.cache_path)  # X: (N, 4, L) one-hot, shared by both conditions -- same test-split sequences

    # Derive the per-condition attribution tracks from the cache, compute_attributions()'s default hypothetical=True
    # already returns the mean-centered ISM delta at all 4 bases per position, i.e.
    # the "hypothetical contribution" convention TF-MoDISco expects -- no further transform needed.
    attr_dict = compute_attributions(y0, y_hat, X)  # {"cold": (N,4,L), "warm": (N,4,L)} torch tensors

    one_hot = to_length_major(X)  # computed once -- identical for both conditions, like ohe_seqs was

    # Run TF-MoDISco for the requested condition(s):
    for condition in args.conditions:
        print(f'Running TF-MoDISo for condition "{condition}":')

        hyp_contribs = to_length_major(attr_dict[condition])  # this condition's ISM attribution, in place of DeepLIFT_data

        pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
            hypothetical_contribs=hyp_contribs,  # (n_examples, seq_len, 4) -- length-major, see to_length_major
            one_hot=one_hot,  # (n_examples, seq_len, 4)
            verbose=True,  # detailed logging
            sliding_window_size=15,  # seqlet core width in bp
            flank_size=10,  # extra bp of context around each seqlet
            target_seqlet_fdr=0.01,  # statistical threshold
            max_seqlets_per_metacluster=args.max_seqlets,
        )

        modiscolite.io.save_hdf5(  # serializes both pattern lists to one HDF5 file per condition
            filename=str(out_dir / f'TF-MoDISco_patterns_{condition}.h5'),
            pos_patterns=pos_patterns,
            neg_patterns=neg_patterns,
            window_size=15,  # must match sliding_window_size
        )

    # Save pattern data as .tsv files:
    all_patterns = []

    for condition in args.conditions:  # open each condition's h5 file and check if it contains pos or neg patterns
        with h5py.File(out_dir / f'TF-MoDISco_patterns_{condition}.h5') as modisco_file:
            for pattern in ['pos', 'neg']:
                if pattern + '_patterns' not in modisco_file:
                    continue
                for name, datasets in modisco_file[pattern + '_patterns'].items():  # for each motif, named like pattern_1
                    pattern_df = pd.DataFrame({
                        'condition': condition,
                        'pattern': pattern + '_' + name,
                        'pos': np.arange(1, len(datasets['sequence']) + 1),  # datasets['sequence'] = position probability matrix (motif_len, 4)
                        **{f'ppm_{b}': datasets['sequence'][:, i] for i, b in enumerate('ACGT')},
                        **{f'cwm_{b}': datasets['contrib_scores'][:, i] for i, b in enumerate('ACGT')},  # datasets['contrib_scores'] = effect size (motif_len, 4)
                    })
                    all_patterns.append(pattern_df)

    # No patterns found for any condition -- write an empty but correctly-columned output.
    combined_columns = ['condition', 'pattern', 'pos'] + [f'ppm_{b}' for b in 'ACGT'] + [f'cwm_{b}' for b in 'ACGT']
    combined = pd.concat(all_patterns) if all_patterns else pd.DataFrame(columns=combined_columns)
    if not all_patterns:
        print(f"No patterns found for any of {args.conditions} -- writing an empty TF-MoDISco_patterns.tsv.gz")
    combined.to_csv(out_dir / 'TF-MoDISco_patterns.tsv.gz', sep='\t', na_rep='NA', index=False)  # combined table of every motif from every condition to gzipped tsv


if __name__ == "__main__":
    main()
