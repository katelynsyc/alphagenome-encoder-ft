from __future__ import annotations

### Render ISM-derived PPM logos next to their matched JASPAR PPM, grouped into cold/warm x pos/neg figures ###
import argparse
import textwrap
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tangermeme.annotate import read_meme, tomtom

"""
python3 scripts/3_analyze/plot_tfmodisco_pwm_vs_jaspar.py \
  --patterns_tsv path/to/TF-MoDISco_patterns.tsv.gz \
  --tomtom_tsv path/to/TF-MoDISco_tomtom_matches.tsv \
  --meme_db metadata/motif_databases/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt \
  --output_dir TF-MoDISco_pwm_jaspar_logos
"""

bases = list('ACGT')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot ISM-derived PPM logos beside their matched JASPAR PPM, one "
                                                   "figure per condition x pos/neg group.")
    parser.add_argument("--patterns_tsv", required=True,
                         help="TF-MoDISco_patterns.tsv.gz written by tfmodisco_ism.py.")
    parser.add_argument("--tomtom_tsv", required=True,
                         help="TF-MoDISco_tomtom_matches.tsv written by find_known_motifs.py, joined in to label each "
                              "pattern's row with its best known-motif match and p-value.")
    parser.add_argument("--meme_db", required=True,
                         help="Known-motif database in MEME format (the same one used by find_known_motifs.py), "
                              "used to pull each pattern's matched JASPAR PPM.")
    parser.add_argument("--output_dir", default="TF-MoDISco_pwm_jaspar_logos",
                         help="Directory to write one PNG per condition x pos/neg group. Defaults to a new "
                              "directory name so existing TF-MoDISco_logos/ outputs are never overwritten.")
    parser.add_argument("--pval_threshold", type=float, default=0.01,
                         help="Only plot patterns whose best JASPAR database match (pval0) is at or below this "
                              "value -- drops patterns with no confident known-motif annotation. Default 0.01.")
    parser.add_argument("--trim_threshold", type=float, default=0.3,
                         help="Coarse pre-filter: positions are trimmed to where |cwm| >= this fraction of the "
                              "pattern's max |cwm| (same convention find_known_motifs.py uses for the TOMTOM "
                              "query) before the exact TOMTOM alignment window (see below) tightens it further. "
                              "Default 0.3.")
    parser.add_argument("--flank", type=int, default=4,
                         help="Extra bases of real flanking DNA to keep on each side of the exact TOMTOM alignment "
                              "span, on the ISM side only (pulled from the full 50bp window). The JASPAR panel "
                              "always shows exactly the aligned span of that motif, with no padding. Default 4.")
    parser.add_argument("--only_groups", nargs='*', default=None,
                         help="Restrict output to these condition_sign groups (e.g. warm_neg cold_pos). The global "
                              "y-axis scale is still computed from every group so a subset preview matches the "
                              "full run. Default: all groups.")
    parser.add_argument("--filename_suffix", default="",
                         help="Appended before .png on each output filename (e.g. '_v2'), so re-running with "
                              "different settings doesn't overwrite a previous run in the same --output_dir. "
                              "Default: none.")
    parser.add_argument("--layout", choices=["page", "rows", "uniform_width"], default="page",
                         help="'page' (default): tile every pattern in a group into a grid of small cells "
                              "(label + ISM logo + JASPAR logo, stacked) sized to fit on one US-Letter-landscape "
                              "PDF page -- groups with few patterns render smaller than a full page rather than "
                              "being stretched to fill it. 'rows': one pattern per row, two columns (ISM, "
                              "JASPAR), figure grows tall with pattern count -- writes PNG. 'uniform_width': like "
                              "'rows' but every base position has the same physical width in both columns -- "
                              "writes PNG.")
    parser.add_argument("--match_box_size", action=argparse.BooleanOptionalAction, default=False,
                         help="--layout page only: normally each condition x sign group picks its own motif-box "
                              "cell size independently (smaller for groups with more patterns, since --layout "
                              "page shrinks cells just enough to fit everything on one page -- see _grid_dims). "
                              "With this flag, every group instead reuses the cell size that "
                              "--match_box_size_reference's own group needs, so all four figures' motif boxes are "
                              "the same physical size; groups smaller than the reference just use fewer "
                              "rows/columns rather than bigger cells. Default: off (each group sized independently, "
                              "the original behavior).")
    parser.add_argument("--match_box_size_reference", default="warm_pos",
                         help="Group (as '<condition>_<sign>', e.g. 'warm_pos') whose --layout page cell size is "
                              "reused by every group when --match_box_size is set. Default: warm_pos.")
    return parser


def pattern_sort_key(pattern: str) -> int:
    return int(pattern.rsplit('_', 1)[1])


def trim_bounds(cwm: np.ndarray, trim_threshold: float) -> tuple[int, int]:
    """Half-open [start, end) slice bounds trimming flanking low-contribution positions, mirroring
    find_known_motifs.py's trim_by_cwm so the PPM shown here matches what TOMTOM actually matched on."""
    score = np.abs(cwm).sum(axis=1)  # (pos,) total |contribution| per position, across bases
    pass_inds = np.where(score >= score.max() * trim_threshold)[0]
    return int(pass_inds.min()), int(pass_inds.max()) + 1


def to_info(ppm: np.ndarray) -> np.ndarray:
    """Per-position information content (bits) from a probability matrix -- the same transform used by the
    existing PPM/bits panel, so heights are properly bounded at 2 bits/position and comparable to what you're
    already used to. Handles exact 0/1 entries (as JASPAR PFMs have) with no NaNs/warnings."""
    df = pd.DataFrame(ppm, columns=bases)
    return logomaker.transform_matrix(df, from_type='probability', to_type='information').to_numpy()


def refine_alignment_window(query_ppm: np.ndarray, target_ppm: np.ndarray
                             ) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, bool]:
    """Re-run TOMTOM on this single (query, target) pair to recover the exact span TOMTOM's own alignment used --
    a tighter, data-driven crop than a fixed CWM-fraction threshold, since it's literally the region that produced
    the reported match. query_ppm/target_ppm are (length, 4). Returns half-open (query_start, query_end) and
    (target_start, target_end) slice bounds (with no context padding -- callers add that against the full,
    untrimmed arrays), the target PPM in the orientation TOMTOM actually aligned to (reverse-complemented if that
    won), and whether that flip happened.

    TOMTOM reports offset/overlap such that target[j] aligns to query[j + offset] over `overlap` columns
    (verified empirically: offset is negative when the target aligns to a later region of the query, positive
    when the query aligns to a later region of the target), so the aligned span of each is:
        query[max(0, -offset) : max(0, -offset) + overlap], target[max(0, offset) : max(0, offset) + overlap]
    TOMTOM's `reverse_complement=True` default also checks the target's reverse complement and reports whichever
    orientation scored better via `strand` (1 == reverse complement won) -- that must be applied to target_ppm
    *before* the offset/overlap slice makes sense, otherwise the two aligned columns show different strands and
    the letters won't visually correspond even though the match is real.
    """
    _, _, offsets, overlaps, strands = tomtom([query_ppm.T], [target_ppm.T])
    offset, overlap, is_rc = int(offsets[0, 0]), int(overlaps[0, 0]), bool(strands[0, 0])

    oriented_target = target_ppm[::-1, ::-1] if is_rc else target_ppm  # A<->T, C<->G, position-reversed

    q_start = max(0, -offset)
    q_end = min(len(query_ppm), q_start + overlap)
    t_start = max(0, offset)
    t_end = min(len(oriented_target), t_start + overlap)
    return (q_start, q_end), (t_start, t_end), oriented_target, is_rc


def collect_group_data(df: pd.DataFrame, matches: pd.DataFrame, target_db: dict, pval_threshold: float,
                        trim_threshold: float, flank: int) -> dict:
    """One pass over every condition x sign group, computing the trimmed ISM PPM and matched JASPAR PPM for each
    pattern that passes pval_threshold. Collected up front (rather than per-figure) so the y-axis scale below can
    be computed globally, across every group, before anything is drawn."""
    groups = {}
    for (condition, sign), group_df in df.groupby(['condition', 'sign']):
        cond_matches = matches.loc[condition]
        patterns = sorted(group_df['pattern'].unique(), key=pattern_sort_key)
        patterns = [p for p in patterns if cond_matches.loc[p, 'pval0'] <= pval_threshold]

        rows = []
        for pattern in patterns:
            match_name, pval = cond_matches.loc[pattern, ['match0', 'pval0']]
            if match_name not in target_db:
                print(f"Warning: match {match_name!r} for {condition}/{pattern} not found in --meme_db -- skipping")
                continue

            pattern_group = group_df[group_df['pattern'] == pattern].sort_values('pos')
            ppm = pattern_group[[f'ppm_{b}' for b in bases]].to_numpy()
            cwm = pattern_group[[f'cwm_{b}' for b in bases]].to_numpy()

            # coarse CWM-fraction trim first (cheap, drops most of the 50bp window; matches what
            # find_known_motifs.py originally fed TOMTOM, so re-aligning against it below reproduces that match)...
            start, end = trim_bounds(cwm, trim_threshold)
            coarse_ppm = ppm[start:end]

            ppm_match_full = target_db[match_name].numpy().T  # (4, L) -> (L, 4)

            # ...then tighten to the exact TOMTOM alignment span, oriented to the strand that actually matched...
            (q_start, q_end), (t_start, t_end), oriented_target, is_rc = refine_alignment_window(coarse_ppm, ppm_match_full)

            # ...then pad the ISM side back out by `flank` bases of real flanking DNA, against the FULL
            # (untrimmed) array -- the coarse-trimmed query has nothing beyond its own edges to pad with. The
            # JASPAR side gets no padding: it's shown as exactly the aligned span, nothing more.
            abs_q_start, abs_q_end = start + q_start, start + q_end
            final_q_start = max(0, abs_q_start - flank)
            final_q_end = min(len(ppm), abs_q_end + flank)

            info_pattern = to_info(ppm[final_q_start:final_q_end])
            info_match = to_info(oriented_target[t_start:t_end])  # exactly the aligned span, no padding

            rows.append({
                'pattern': pattern, 'info_pattern': info_pattern, 'match_name': match_name, 'pval': pval,
                'info_match': info_match, 'is_rc': is_rc,
            })

        if rows:
            groups[(condition, sign)] = rows
    return groups


def plot_logo(ax: plt.Axes, info: np.ndarray, ylim: tuple[float, float]) -> None:
    logo_df = pd.DataFrame(info, columns=bases, index=np.arange(len(info)))
    logomaker.Logo(logo_df, ax=ax)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(info) - 0.5)


def plot_group(rows: list[dict], title: str, out_path: Path, ylim: tuple[float, float]) -> None:
    """Default layout: a plain equal-width-column grid (both panels in a row share the figure's two column
    widths, regardless of how many bases each actually has -- a short JASPAR match gets its letters stretched to
    fill the column, same as a long one)."""
    grid_height = 1.4 * len(rows)
    title_pad = 0.4  # inches reserved above the grid for fig.suptitle, constant regardless of pattern count so it
                     # doesn't collide with row 0 (a fixed *fraction* would shrink in absolute terms as groups grow)
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, grid_height + title_pad), squeeze=False)

    for row_idx, row in enumerate(rows):
        ax_pattern, ax_match = axes[row_idx]

        plot_logo(ax_pattern, row['info_pattern'], ylim)
        ax_pattern.set_ylabel('bits', fontsize=8)

        plot_logo(ax_match, row['info_match'], ylim)

        match_label = row['match_name'] + (' (reverse complement)' if row['is_rc'] else '')
        label = f"{row['pattern']}\nmatch: {match_label}\np={row['pval']:.2e}"
        ax_pattern.annotate(label, xy=(-0.25, 0.5), xycoords='axes fraction', ha='right', va='center', fontsize=8)

    axes[0, 0].set_title('ISM-derived motif')
    axes[0, 1].set_title('JASPAR match')
    total_height = grid_height + title_pad
    fig.suptitle(title, y=1 - 0.5 * title_pad / total_height)  # center in the reserved band; default y=0.98 doesn't scale with figure height and drifts into the grid

    fig.tight_layout(rect=(0, 0, 1, grid_height / total_height))  # cap the subplot grid below the reserved title band
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# constants for plot_group_uniform_width only (--uniform_char_width)
UNIFORM_CHAR_WIDTH = 0.3  # inches per base position -- the SAME value for both columns, so a letter is the same
                          # physical width whether it's in the (longer, flanked) ISM panel or the (short,
                          # exact-span) JASPAR panel, rather than each column squeezed to an equal fixed width
UNIFORM_COLUMN_GAP = 0.6  # inches of horizontal space between the ISM and JASPAR panels
UNIFORM_LABEL_GAP = 0.7   # inches between the row label text and the ISM axis edge -- has to clear both the
                          # y-tick numbers and the rotated 'bits' ylabel, which otherwise collide with the label
UNIFORM_LEFT_MARGIN = 4.2  # inches reserved on the left for the per-row text label (+ UNIFORM_LABEL_GAP above)
UNIFORM_RIGHT_MARGIN = 0.3
UNIFORM_ROW_HEIGHT = 1.5       # inches allotted per pattern row (axis + a little breathing room)
UNIFORM_ROW_AXIS_FRAC = 0.7   # fraction of UNIFORM_ROW_HEIGHT actually used by the axis, leaving spacing between rows
UNIFORM_TOP_MARGIN = 0.9       # inches reserved for fig.suptitle + column headers


def plot_group_uniform_width(rows: list[dict], title: str, out_path: Path, ylim: tuple[float, float]) -> None:
    """--uniform_char_width layout: every base position is the same physical width in both columns. Panels are
    placed with manually computed axes (not plt.subplots) since each row's ISM/JASPAR panel widths differ."""
    max_n_ism = max(len(row['info_pattern']) for row in rows)
    max_n_match = max(len(row['info_match']) for row in rows)
    ism_col_width = max_n_ism * UNIFORM_CHAR_WIDTH
    match_col_width = max_n_match * UNIFORM_CHAR_WIDTH

    fig_width = UNIFORM_LEFT_MARGIN + ism_col_width + UNIFORM_COLUMN_GAP + match_col_width + UNIFORM_RIGHT_MARGIN
    fig_height = UNIFORM_ROW_HEIGHT * len(rows) + UNIFORM_TOP_MARGIN
    fig = plt.figure(figsize=(fig_width, fig_height))

    # both columns' left edges are fixed (based on the group's longest sequence) so every row's panels line up
    # vertically; a row with a shorter sequence just gets a narrower axis anchored at the same left edge, rather
    # than being stretched to fill the column -- that's what keeps the per-base character width constant.
    ism_left = UNIFORM_LEFT_MARGIN / fig_width
    match_left = (UNIFORM_LEFT_MARGIN + ism_col_width + UNIFORM_COLUMN_GAP) / fig_width

    for row_idx, row in enumerate(rows):
        row_bottom_in = ((len(rows) - 1 - row_idx) * UNIFORM_ROW_HEIGHT
                         + UNIFORM_ROW_HEIGHT * (1 - UNIFORM_ROW_AXIS_FRAC) / 2)
        bottom = row_bottom_in / fig_height
        height = (UNIFORM_ROW_HEIGHT * UNIFORM_ROW_AXIS_FRAC) / fig_height

        n_ism = len(row['info_pattern'])
        n_match = len(row['info_match'])
        ax_pattern = fig.add_axes((ism_left, bottom, n_ism * UNIFORM_CHAR_WIDTH / fig_width, height))
        ax_match = fig.add_axes((match_left, bottom, n_match * UNIFORM_CHAR_WIDTH / fig_width, height))

        plot_logo(ax_pattern, row['info_pattern'], ylim)
        ax_pattern.set_ylabel('bits', fontsize=8)

        plot_logo(ax_match, row['info_match'], ylim)

        match_label = row['match_name'] + (' (reverse complement)' if row['is_rc'] else '')
        label = f"{row['pattern']}\nmatch: {match_label}\np={row['pval']:.2e}"
        fig.text(ism_left - UNIFORM_LABEL_GAP / fig_width, bottom + height / 2, label,
                  ha='right', va='center', fontsize=8)

    header_y = 1 - 0.55 * UNIFORM_TOP_MARGIN / fig_height
    fig.text(ism_left + ism_col_width / fig_width / 2, header_y, 'ISM-derived motif', ha='center', fontsize=11)
    fig.text(match_left + match_col_width / fig_width / 2, header_y, 'JASPAR match', ha='center', fontsize=11)
    fig.suptitle(title, y=1 - 0.15 * UNIFORM_TOP_MARGIN / fig_height)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# constants for plot_group_page only (--layout page, the default)
PAGE_WIDTH_IN = 11.0    # US Letter, landscape
PAGE_HEIGHT_IN = 8.5
PAGE_MARGIN_IN = 0.15   # outer margin on every side of the page
TITLE_BAND_IN = 0.9     # reserved at the top for suptitle + one-line legend, above the grid
PREF_CELL_W_IN = 2.3    # "comfortable" cell size, used as-is whenever the group is small enough to fit at this
PREF_CELL_H_IN = 1.55   # size without exceeding the page -- small groups render smaller than a full page rather
                        # than being stretched to fill it; only groups too large at this size get shrunk (see below)
CELL_PAD_IN = 0.05      # inner padding between a cell's border and its label/logo axes
LABEL_FRAC = 0.26       # fraction of cell height given to the label text; the rest splits evenly between the two
                        # logos (stacked, not side-by-side, so each logo keeps the cell's full width -- width is
                        # what matters for reading base identity along a several-bases-long motif)
LABEL_GAP_IN = 0.04     # extra breathing room between the bottom of the label text and the top of the ISM logo
                        # axes beneath it, on top of CELL_PAD_IN -- keeps descenders/long labels from crowding
                        # the logo even when a cell is small


def _grid_dims(n: int, avail_w: float, avail_h: float,
               cell_w: float | None = None, cell_h: float | None = None) -> tuple[int, int, float, float]:
    """Pick (cols, rows, cell_w, cell_h) for n cells inside an avail_w x avail_h in² area. Uses the given cell_w/
    cell_h (PREF_CELL_*  if not given) as-is if n cells fit at that size (small groups end up smaller than the
    full page, or than whatever this cell size fills for a larger reference group); otherwise shrinks the cell
    size just enough to pack everything into the available area (large groups fill the whole page). A caller
    passing a fixed cell_w/cell_h -- see plot_group_page's --match_box_size use -- only hits that shrink branch if
    n itself doesn't even fit at that size, which by construction it always will for the reference group."""
    cell_w = PREF_CELL_W_IN if cell_w is None else cell_w
    cell_h = PREF_CELL_H_IN if cell_h is None else cell_h
    max_cols_at_pref = max(1, int(avail_w // cell_w))
    max_rows_at_pref = max(1, int(avail_h // cell_h))
    if n <= max_cols_at_pref * max_rows_at_pref:
        cols = min(max_cols_at_pref, n)
        rows = -(-n // cols)  # ceil div
        return cols, rows, cell_w, cell_h

    aspect = avail_w / avail_h
    cols = int(np.ceil(np.sqrt(n * aspect)))
    rows = -(-n // cols)
    return cols, rows, avail_w / cols, avail_h / rows


def plot_group_page(rows: list[dict], title: str, out_path: Path, ylim: tuple[float, float],
                     cell_w: float | None = None, cell_h: float | None = None) -> None:
    """--layout page (default): tile every pattern into a grid of small cells, each a stacked (label, ISM logo,
    JASPAR logo) block, sized so the whole group fits on one US-Letter-landscape page. Cell size is fixed at
    PREF_CELL_*  (or at the given cell_w/cell_h, when the caller wants every group's boxes to match a shared
    reference group's size -- see --match_box_size) and only the page (figure) shrinks for small groups; for
    groups too large to fit at that size, the page is held at full size and the cell size shrinks instead --
    either way nothing exceeds one page."""
    n = len(rows)
    avail_w = PAGE_WIDTH_IN - 2 * PAGE_MARGIN_IN
    avail_h = PAGE_HEIGHT_IN - TITLE_BAND_IN - PAGE_MARGIN_IN
    cols, grid_rows, cell_w, cell_h = _grid_dims(n, avail_w, avail_h, cell_w, cell_h)

    fig_w = 2 * PAGE_MARGIN_IN + cols * cell_w
    fig_h = TITLE_BAND_IN + PAGE_MARGIN_IN + grid_rows * cell_h
    fig = plt.figure(figsize=(fig_w, fig_h))

    grid_top_in = fig_h - TITLE_BAND_IN
    label_fontsize = max(4.5, min(8, cell_w * 3.2))
    wrap_width = max(10, int(cell_w * 13))

    for idx, row in enumerate(rows):
        col, r = idx % cols, idx // cols
        cell_x0 = PAGE_MARGIN_IN + col * cell_w
        cell_y1 = grid_top_in - r * cell_h  # top edge of this cell
        cell_y0 = cell_y1 - cell_h

        logos_h = (cell_h * (1 - LABEL_FRAC) - LABEL_GAP_IN - 2 * CELL_PAD_IN) / 2
        logo_x0 = cell_x0 + CELL_PAD_IN
        logo_w = cell_w - 2 * CELL_PAD_IN
        bottom_slot_y0 = cell_y0 + CELL_PAD_IN  # lower of the two stacked logo slots -> JASPAR match
        top_slot_y0 = bottom_slot_y0 + logos_h + CELL_PAD_IN  # upper slot, just below the label -> ISM-derived

        ax_match = fig.add_axes((logo_x0 / fig_w, bottom_slot_y0 / fig_h, logo_w / fig_w, logos_h / fig_h))
        ax_ism = fig.add_axes((logo_x0 / fig_w, top_slot_y0 / fig_h, logo_w / fig_w, logos_h / fig_h))
        plot_logo(ax_ism, row['info_pattern'], ylim)
        plot_logo(ax_match, row['info_match'], ylim)
        ax_ism.set_yticks([])
        ax_match.set_yticks([])

        match_label = row['match_name'] + (' rc' if row['is_rc'] else '')
        # exactly 2 lines total (pattern name, then match+pval) -- LABEL_FRAC only reserves room for 2, so a
        # long match name is truncated with an ellipsis rather than wrapping to a 3rd line that would collide
        # with the ISM logo axes below it
        match_line = textwrap.shorten(f"{match_label}  p={row['pval']:.1e}", wrap_width, placeholder=' …')
        label = f"{row['pattern']}\n{match_line}"
        fig.text(cell_x0 / fig_w + (cell_w / 2) / fig_w, (cell_y1 - CELL_PAD_IN) / fig_h, label,
                  ha='center', va='top', fontsize=label_fontsize, linespacing=1.15)

    fig.suptitle(title, y=1 - 0.4 * TITLE_BAND_IN / fig_h, fontsize=14)
    fig.text(0.5, 1 - 0.85 * TITLE_BAND_IN / fig_h,
              f'n={n} patterns  ·  each cell, top to bottom: ISM-derived motif, JASPAR match  ·  shared bits scale '
              f'(max={ylim[1]:.2f})', ha='center', fontsize=8, color='#555555')

    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.patterns_tsv, sep='\t')
    df['sign'] = df['pattern'].str.split('_', n=1).str[0]  # 'pos' or 'neg', from names like 'pos_pattern_0'

    matches = pd.read_csv(args.tomtom_tsv, sep='\t').set_index(['condition', 'pattern'])
    target_db = read_meme(args.meme_db)

    groups = collect_group_data(df, matches, target_db, args.pval_threshold, args.trim_threshold, args.flank)
    if not groups:
        print(f'No patterns pass pval0 <= {args.pval_threshold} -- nothing to plot')
        return

    all_values = np.concatenate([
        np.concatenate([row['info_pattern'].ravel() for row in rows] + [row['info_match'].ravel() for row in rows])
        for rows in groups.values()
    ])
    ylim = (0, 1.05 * all_values.max())  # shared across every group/pattern so figures are directly comparable

    plot_fn, ext = {'page': (plot_group_page, 'pdf'), 'rows': (plot_group, 'png'),
                    'uniform_width': (plot_group_uniform_width, 'png')}[args.layout]

    page_kwargs = {}
    if args.match_box_size:
        if args.layout != 'page':
            raise SystemExit("--match_box_size only applies to --layout page")
        ref_condition, _, ref_sign = args.match_box_size_reference.partition('_')
        if (ref_condition, ref_sign) not in groups:
            raise SystemExit(f"--match_box_size_reference {args.match_box_size_reference!r} has no patterns "
                              f"passing --pval_threshold -- available groups: "
                              f"{sorted(f'{c}_{s}' for c, s in groups)}")
        avail_w = PAGE_WIDTH_IN - 2 * PAGE_MARGIN_IN
        avail_h = PAGE_HEIGHT_IN - TITLE_BAND_IN - PAGE_MARGIN_IN
        _, _, ref_cell_w, ref_cell_h = _grid_dims(len(groups[(ref_condition, ref_sign)]), avail_w, avail_h)
        page_kwargs = {'cell_w': ref_cell_w, 'cell_h': ref_cell_h}
        print(f'--match_box_size: every group will use {args.match_box_size_reference}\'s '
              f'{ref_cell_w:.2f}in x {ref_cell_h:.2f}in cell size')

    sign_label = {'pos': 'Positive', 'neg': 'Negative'}
    for (condition, sign), rows in groups.items():
        group_name = f'{condition}_{sign}'
        if args.only_groups is not None and group_name not in args.only_groups:
            continue
        title = f'{condition.capitalize()} {sign_label[sign]} Contribution Motifs'
        plot_fn(rows, title=title, out_path=out_dir / f'{group_name}{args.filename_suffix}.{ext}', ylim=ylim,
                **page_kwargs)

    print(f'Wrote grouped PPM-vs-JASPAR figures to {out_dir} '
          f'(patterns filtered to JASPAR match pval0 <= {args.pval_threshold}, shared ylim={ylim})')


if __name__ == "__main__":
    main()
