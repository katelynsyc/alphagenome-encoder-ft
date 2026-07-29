from __future__ import annotations

### Render sequence logos for TF-MoDISco patterns, grouped into cold/warm x pos/neg figures ###
import argparse
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

"""
python3 scripts/3_analyze/plot_tfmodisco_patterns.py \
  --patterns_tsv path/to/TF-MoDISco_patterns.tsv.gz \
  --tomtom_tsv path/to/TF-MoDISco_tomtom_matches.tsv \
  --output_dir TF-MoDISco_logos
"""

bases = list('ACGT')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot CWM/PPM logos for TF-MoDISco patterns, one figure per condition x pos/neg group.")
    parser.add_argument("--patterns_tsv", required=True,
                         help="TF-MoDISco_patterns.tsv.gz written by tfmodisco_ism.py.")
    parser.add_argument("--tomtom_tsv", required=True,
                         help="TF-MoDISco_tomtom_matches.tsv written by find_known_motifs.py, joined in to label each "
                              "pattern's row with its best known-motif match and p-value.")
    parser.add_argument("--output_dir", default="TF-MoDISco_logos",
                         help="Directory to write one PNG per condition x pos/neg group.")
    return parser


def pattern_sort_key(pattern: str) -> int:
    return int(pattern.rsplit('_', 1)[1])


def plot_group(df: pd.DataFrame, matches: pd.DataFrame, title: str, out_path: Path) -> None:
    patterns = sorted(df['pattern'].unique(), key=pattern_sort_key)

    grid_height = 1.4 * len(patterns)
    title_pad = 0.4  # inches reserved above the grid for fig.suptitle, constant regardless of pattern count so it
                     # doesn't collide with row 0 (a fixed *fraction* would shrink in absolute terms as groups grow)
    fig, axes = plt.subplots(len(patterns), 2, figsize=(10, grid_height + title_pad), squeeze=False)

    for row, pattern in enumerate(patterns):
        group = df[df['pattern'] == pattern].sort_values('pos')
        ax_cwm, ax_ppm = axes[row]

        cwm = group[[f'cwm_{b}' for b in bases]].rename(columns=lambda c: c.removeprefix('cwm_'))
        cwm.index = group['pos'].to_numpy()
        logomaker.Logo(cwm, ax=ax_cwm)  # cwm values are signed contributions, plotted directly (no bits transform)
        ax_cwm.axhline(0, color='black', linewidth=0.5)
        ax_cwm.set_ylabel('contribution', fontsize=8)

        ppm = group[[f'ppm_{b}' for b in bases]].rename(columns=lambda c: c.removeprefix('ppm_'))
        ppm.index = group['pos'].to_numpy()
        info = logomaker.transform_matrix(ppm, from_type='probability', to_type='information')  # bits-scaled, standard sequence-logo view
        logomaker.Logo(info, ax=ax_ppm)
        ax_ppm.set_ylabel('bits', fontsize=8)

        match, pval = matches.loc[pattern, ['match0', 'pval0']]
        label = f'{pattern}\nmatch: {match}\np={pval:.2e}'
        ax_cwm.annotate(label, xy=(-0.25, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=8)

    axes[0, 0].set_title('CWM (contribution weights)')
    axes[0, 1].set_title('PPM (information content)')
    axes[-1, 0].set_xlabel('position')
    axes[-1, 1].set_xlabel('position')
    total_height = grid_height + title_pad
    fig.suptitle(title, y=1 - 0.5 * title_pad / total_height)  # center in the reserved band; default y=0.98 doesn't scale with figure height and drifts into the grid

    fig.tight_layout(rect=(0, 0, 1, grid_height / total_height))  # cap the subplot grid below the reserved title band
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.patterns_tsv, sep='\t')
    df['sign'] = df['pattern'].str.split('_', n=1).str[0]  # 'pos' or 'neg', from names like 'pos_pattern_0'

    matches = pd.read_csv(args.tomtom_tsv, sep='\t').set_index(['condition', 'pattern'])

    for (condition, sign), group in df.groupby(['condition', 'sign']):
        plot_group(group, matches.loc[condition], title=f'{condition} {sign}', out_path=out_dir / f'{condition}_{sign}.png')

    print(f'Wrote {df.groupby(["condition", "sign"]).ngroups} grouped figures to {out_dir}')


if __name__ == "__main__":
    main()
