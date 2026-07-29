"""Find the test-split sequences with the lowest MEASURED warm-condition expression
(enrichment_warm) in modelling_data_tamsACR.tsv, and write them to a table for manual review.

Ranking metric: enrichment_warm, ascending -- the top of the table is the most
warm-repressed sequence in the test split. Species is inferred from the id prefix
using the same JORES_SPECIES_PREFIXES mapping as species_masks.py.

Reuses cold_warm_extremes.py's genomic-coordinate and nearest-gene annotation
joins (media-3.xlsx + plantGREP's tamsACR_annotation.tsv.gz + per-species GFF3s)
-- see that module's docstring for where they come from. Column order matches
actual_cold_minus_warm_top100_genes.tsv (TSS_dist/gene_function pulled up front,
right after the ranking columns) so gene_function is easy to scan.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # cold_warm_extremes.py lives alongside this file
from cold_warm_extremes import (
    add_gene_annotation,
    load_genomic_coords,
    write_species_bed_files,
    DEFAULT_MEDIA3,
    DEFAULT_GENE_ANNOTATION,
    DEFAULT_INPUT,
)

from alphagenome_encoder_ft import build_species_masks

METADATA_DIR = "/grid/koo/home/kachu/projects/alphagenome-encoder-ft/metadata"
DEFAULT_OUTPUT = f"{METADATA_DIR}/lowest_warm_expression_top100.tsv"
DEFAULT_BED_DIR = f"{METADATA_DIR}/acr_bed_lowest_warm"

# Matches actual_cold_minus_warm_top100_genes.tsv's column order (id, TSS_dist,
# gene_function, chromosome, start, end, region, closest_TSS, sequence), with
# rank/species/enrichment_warm prepended since those drive this table's ranking.
OUTPUT_COLUMNS = [
    "rank",
    "id",
    "species",
    "enrichment_warm",
    "TSS_dist",
    "gene_function",
    "chromosome",
    "start",
    "end",
    "region",
    "closest_TSS",
    "sequence",
]


def find_lowest_warm_expression(input_tsv: str, media3_xlsx: str, top_n: int = 100) -> pd.DataFrame:
    """Return the top_n test-split rows with the lowest enrichment_warm."""
    df = pd.read_csv(input_tsv, sep="\t")
    df = df[df["set"] == "test"].reset_index(drop=True)

    masks = build_species_masks(df["id"].tolist())
    species = pd.Series("Other", index=df.index)
    for name, mask in masks.items():
        if name == "Other":
            continue
        species[mask] = name
    df["species"] = species

    top = df.sort_values("enrichment_warm", ascending=True).head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    top = top.merge(load_genomic_coords(media3_xlsx), on="id", how="left")
    n_unmatched = top["chromosome"].isna().sum()
    if n_unmatched:
        unmatched_ids = top.loc[top["chromosome"].isna(), "id"].tolist()
        print(f"{n_unmatched} id(s) had no genomic coordinates in media-3.xlsx "
              f"(likely non-genomic controls): {unmatched_ids}")

    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--media3", default=DEFAULT_MEDIA3)
    parser.add_argument("--gene-annotation", default=DEFAULT_GENE_ANNOTATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bed-dir", default=DEFAULT_BED_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    top = find_lowest_warm_expression(args.input, args.media3, top_n=args.top_n)
    top = add_gene_annotation(top, args.gene_annotation)
    top = top[OUTPUT_COLUMNS]

    top.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(top)} lowest-warm-expression test sequences to {args.output}")
    print(top[["rank", "id", "species", "enrichment_warm", "closest_TSS", "gene_function"]].head(10).to_string(index=False))

    write_species_bed_files(top.rename(columns={"enrichment_warm": "cold_minus_warm"}), args.bed_dir)


if __name__ == "__main__":
    main()
