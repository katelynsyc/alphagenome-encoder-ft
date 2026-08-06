"""Cross-compare TF-MoDISco pattern sets from two separate runs (e.g. with vs.
without the 15bp Jores adapters) using TOMTOM motif-to-motif comparison, and
plot the patterns that have no significant match in the other run.

Unlike find_known_motifs.py (which TOMTOM-matches patterns against a known-motif
database), this matches one run's patterns directly against the other run's
patterns -- there's no database, the "targets" are just the other run's motifs.

Each run's full pattern set (pooled across condition and pos/neg sign) is
compared against the other's in both directions, so every pattern gets its own
single best cross-run match + p-value. A pattern with no match at or below
--pval_threshold in the OTHER run is "unique" to its own run; anything else is
"overlapping" (found under both conditions).

Patterns are trimmed by |cwm| before comparison, the same convention
find_known_motifs.py uses (see trim_by_cwm), so flanking near-zero-contribution
positions don't dilute the match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tangermeme.annotate import tomtom

bases = list('ACGT')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns_a", required=True,
                         help="TF-MoDISco_patterns.tsv.gz for run A, e.g. TF-MoDISco_ism_adapters/.")
    parser.add_argument("--patterns_b", required=True,
                         help="TF-MoDISco_patterns.tsv.gz for run B, e.g. TF-MoDISco_ism_best_ism_cache_no_adapters/.")
    parser.add_argument("--label_a", default="adapters", help="Short name for run A, used in output labels/filenames.")
    parser.add_argument("--label_b", default="no_adapters", help="Short name for run B, used in output labels/filenames.")
    parser.add_argument("--pval_threshold", type=float, default=0.01,
                         help="TOMTOM p-value at/below which a cross-run match counts as 'overlapping'. "
                              "Default 0.01, uncorrected (matches find_known_motifs.py's convention). Note this is "
                              "the single BEST of many target comparisons per query, so it isn't corrected for "
                              "multiple testing -- treat it as a ranking cutoff, not a rigorous significance test.")
    parser.add_argument("--trim_threshold", type=float, default=0.3,
                         help="Positions are trimmed to where |cwm| >= this fraction of the pattern's max |cwm|.")
    parser.add_argument("--output_tsv", required=True,
                         help="Where to write the per-pattern cross-run match table.")
    parser.add_argument("--output_dir", required=True,
                         help="Directory to write logo plots for the patterns unique to each run.")
    return parser


def trim_by_cwm(ppm: np.ndarray, cwm: np.ndarray, trim_threshold: float) -> np.ndarray:
    score = np.abs(cwm).sum(axis=1)
    pass_inds = np.where(score >= score.max() * trim_threshold)[0]
    return ppm[pass_inds.min(): pass_inds.max() + 1]


def load_patterns(patterns_tsv: str, trim_threshold: float):
    """Returns (keys, pwms): keys[i] = (condition, pattern) tuple, pwms[i] = (4, len) trimmed PPM."""
    df = pd.read_csv(patterns_tsv, sep='\t')
    keys, pwms, raw = [], [], {}
    for (condition, pattern), group in df.groupby(['condition', 'pattern']):
        group = group.sort_values('pos')
        ppm = group[[f'ppm_{b}' for b in bases]].to_numpy()
        cwm = group[[f'cwm_{b}' for b in bases]].to_numpy()
        trimmed_ppm = trim_by_cwm(ppm, cwm, trim_threshold)
        keys.append((condition, pattern))
        pwms.append(trimmed_ppm.T)  # tomtom expects (alphabet, length)
        raw[(condition, pattern)] = group
    return keys, pwms, raw


def cross_match(keys_q, pwms_q, keys_t, pwms_t):
    """For each query, its single best-matching target + p-value."""
    p_values, _, _, _, _, idxs = tomtom(pwms_q, pwms_t, n_nearest=1)
    best_idx = idxs[:, 0].astype(int)
    best_pval = p_values[:, 0]
    best_target = [keys_t[i] for i in best_idx]
    return best_target, best_pval


def plot_unique_patterns(raw: dict, keys: list, title: str, out_path: Path) -> None:
    if not keys:
        print(f"No unique patterns for {title!r} -- skipping {out_path}")
        return

    grid_height = 1.4 * len(keys)
    title_pad = 0.4
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, grid_height + title_pad), squeeze=False)

    for row, key in enumerate(keys):
        condition, pattern = key
        group = raw[key]
        ax_cwm, ax_ppm = axes[row]

        cwm = group[[f'cwm_{b}' for b in bases]].rename(columns=lambda c: c.removeprefix('cwm_'))
        cwm.index = group['pos'].to_numpy()
        logomaker.Logo(cwm, ax=ax_cwm)
        ax_cwm.axhline(0, color='black', linewidth=0.5)
        ax_cwm.set_ylabel('contribution', fontsize=8)

        ppm = group[[f'ppm_{b}' for b in bases]].rename(columns=lambda c: c.removeprefix('ppm_'))
        ppm.index = group['pos'].to_numpy()
        info = logomaker.transform_matrix(ppm, from_type='probability', to_type='information')
        logomaker.Logo(info, ax=ax_ppm)
        ax_ppm.set_ylabel('bits', fontsize=8)

        ax_cwm.annotate(f'{condition}\n{pattern}', xy=(-0.25, 0.5), xycoords='axes fraction',
                         ha='right', va='center', fontsize=8)

    axes[0, 0].set_title('CWM (contribution weights)')
    axes[0, 1].set_title('PPM (information content)')
    axes[-1, 0].set_xlabel('position')
    axes[-1, 1].set_xlabel('position')
    total_height = grid_height + title_pad
    fig.suptitle(title, y=1 - 0.5 * title_pad / total_height)

    fig.tight_layout(rect=(0, 0, 1, grid_height / total_height))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    keys_a, pwms_a, raw_a = load_patterns(args.patterns_a, args.trim_threshold)
    keys_b, pwms_b, raw_b = load_patterns(args.patterns_b, args.trim_threshold)
    print(f"{args.label_a}: {len(keys_a)} patterns; {args.label_b}: {len(keys_b)} patterns")

    match_a_to_b, pval_a_to_b = cross_match(keys_a, pwms_a, keys_b, pwms_b)
    match_b_to_a, pval_b_to_a = cross_match(keys_b, pwms_b, keys_a, pwms_a)

    threshold_a_to_b = threshold_b_to_a = args.pval_threshold
    print(f"Overlap threshold: p<={args.pval_threshold} (flat, uncorrected)")

    rows = []
    for (condition, pattern), (m_cond, m_pattern), pval in zip(keys_a, match_a_to_b, pval_a_to_b):
        rows.append({
            "source": args.label_a, "condition": condition, "pattern": pattern,
            "best_match_source": args.label_b, "best_match_condition": m_cond, "best_match_pattern": m_pattern,
            "pval": pval, "overlapping": pval <= threshold_a_to_b,
        })
    for (condition, pattern), (m_cond, m_pattern), pval in zip(keys_b, match_b_to_a, pval_b_to_a):
        rows.append({
            "source": args.label_b, "condition": condition, "pattern": pattern,
            "best_match_source": args.label_a, "best_match_condition": m_cond, "best_match_pattern": m_pattern,
            "pval": pval, "overlapping": pval <= threshold_b_to_a,
        })

    out_df = pd.DataFrame(rows).sort_values(["source", "pval"])
    out_df.to_csv(args.output_tsv, sep='\t', index=False)

    n_overlap_a = sum(1 for r in rows if r["source"] == args.label_a and r["overlapping"])
    n_overlap_b = sum(1 for r in rows if r["source"] == args.label_b and r["overlapping"])
    print(f"{args.label_a}: {n_overlap_a}/{len(keys_a)} patterns overlap with {args.label_b}")
    print(f"{args.label_b}: {n_overlap_b}/{len(keys_b)} patterns overlap with {args.label_a}")
    print(f"Wrote {len(out_df)} cross-run match rows to {args.output_tsv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_a = [k for k, m, p in zip(keys_a, match_a_to_b, pval_a_to_b) if p > threshold_a_to_b]
    unique_b = [k for k, m, p in zip(keys_b, match_b_to_a, pval_b_to_a) if p > threshold_b_to_a]

    plot_unique_patterns(raw_a, unique_a, f"Unique to {args.label_a} (no match in {args.label_b})",
                          out_dir / f"unique_to_{args.label_a}.png")
    plot_unique_patterns(raw_b, unique_b, f"Unique to {args.label_b} (no match in {args.label_a})",
                          out_dir / f"unique_to_{args.label_b}.png")


if __name__ == "__main__":
    main()
