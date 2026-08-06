from __future__ import annotations

### Compare TF-MoDISco patterns to a known-motif database (TOMTOM, no MEME-suite install required) ###
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tangermeme.annotate import read_meme, tomtom

"""
python3 scripts/3_analyze/find_known_motifs.py \
  --patterns_tsv results/e898939e/df4406c4716cd2cf/stage2/TF-MoDISco_ism/TF-MoDISco_patterns.tsv.gz \
  --meme_db metadata/motif_databases/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt \
  --output_tsv results/e898939e/df4406c4716cd2cf/stage2/TF-MoDISco_ism/TF-MoDISco_tomtom_matches.tsv
"""

bases = list('ACGT')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TOMTOM-match TF-MoDISco patterns against a MEME-format motif database.")
    parser.add_argument("--patterns_tsv", required=True,
                         help="TF-MoDISco_patterns.tsv.gz written by tfmodisco_ism.py.")
    parser.add_argument("--meme_db", required=True,
                         help="Known-motif database in MEME format, e.g. a JASPAR CORE non-redundant download.")
    parser.add_argument("--output_tsv", default="TF-MoDISco_tomtom_matches.tsv",
                         help="Where to write the match table.")
    parser.add_argument("--top_n_matches", type=int, default=3,
                         help="Number of nearest database motifs to report per pattern.")
    parser.add_argument("--trim_threshold", type=float, default=0.3,
                         help="Positions are trimmed to where |cwm| >= this fraction of the pattern's max "
                              "|cwm| (same convention modiscolite's own tomtomlite_dataframe uses), so flanking "
                              "near-zero-contribution positions don't dilute the match to the database motif.")
    return parser


def trim_by_cwm(ppm: np.ndarray, cwm: np.ndarray, trim_threshold: float) -> np.ndarray:
    score = np.abs(cwm).sum(axis=1)  # (pos,) total |contribution| per position, across bases
    pass_inds = np.where(score >= score.max() * trim_threshold)[0] #trims flanking positions where |cwm| < 30% pattern's max from each pattern
    return ppm[pass_inds.min(): pass_inds.max() + 1]


def main() -> None:
    args = build_arg_parser().parse_args()

    df = pd.read_csv(args.patterns_tsv, sep='\t')

    keys, queries = [], []
    for (condition, pattern), group in df.groupby(['condition', 'pattern']):
        group = group.sort_values('pos')
        ppm = group[[f'ppm_{b}' for b in bases]].to_numpy()  # (pos, 4)
        cwm = group[[f'cwm_{b}' for b in bases]].to_numpy()  # (pos, 4)
        trimmed = trim_by_cwm(ppm, cwm, args.trim_threshold)
        keys.append((condition, pattern))
        queries.append(trimmed.T)  # tomtom expects (alphabet, length)

    if not keys:
        # No patterns to match -- write an empty but correctly-columned output.
        columns = ['condition', 'pattern'] + [c for j in range(args.top_n_matches) for c in (f'match{j}', f'pval{j}')]
        pd.DataFrame(columns=columns).to_csv(args.output_tsv, sep='\t', index=False)
        print(f'{args.patterns_tsv} has no patterns -- wrote an empty {args.output_tsv}')
        return

    target_db = read_meme(args.meme_db)
    target_names = list(target_db.keys())
    target_pwms = list(target_db.values())

    p_values, _, _, _, _, idxs = tomtom(queries, target_pwms, n_nearest=args.top_n_matches) #runs TOMTOM against the database, reporting top N-nearest known motifs and p-values

    results = {'condition': [k[0] for k in keys], 'pattern': [k[1] for k in keys]}
    for j in range(args.top_n_matches):
        results[f'match{j}'] = [target_names[int(idxs[i, j])].strip() for i in range(len(keys))]
        results[f'pval{j}'] = p_values[:, j]

    out_df = pd.DataFrame(results).sort_values('pval0')
    out_df.to_csv(args.output_tsv, sep='\t', index=False)
    print(f'Wrote {len(out_df)} pattern matches to {args.output_tsv}')


if __name__ == "__main__":
    main()
