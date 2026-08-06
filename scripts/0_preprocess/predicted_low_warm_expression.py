"""Find test-split sequences that are BOTH measured and model-predicted to have low
warm-condition expression, and write them to a table for manual review.

Like lowest_warm_expression.py, this is restricted to the test split -- that's the
only split evaluate_jores.py produces predictions for (best_test_eval/test_predictions.csv).

Ranking metric: combined_score = max(enrichment_warm, predicted_warm), ascending.
Taking the max (not e.g. the mean) means a row can only rank near the top if BOTH
the measured and the predicted warm enrichment are low -- a row that's very low in
one but only middling in the other is capped by its higher (less-low) side, so it
drops out of this table. This mirrors predicted_cold_warm_extremes.py's use of
min() for "both high" -- here we want "both low", so the bottleneck is the max.

test_predictions.csv's row order matches modelling_data_tamsACR.tsv's rows filtered
to set == "test" in original file order (no resort) -- same alignment invariant
find_row_indices_for_ids in saturation_mutagenesis.py relies on.
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
DEFAULT_PREDICTIONS = (
    f"{METADATA_DIR}/../results/e898939e/df4406c4716cd2cf/"
    "stage2/best_test_eval/test_predictions.csv"
)
DEFAULT_OUTPUT = f"{METADATA_DIR}/actual_and_predicted_low_warm_top100.tsv"
DEFAULT_BED_DIR = f"{METADATA_DIR}/acr_bed_predicted_low_warm"

# Matches lowest_warm_expression_top100.tsv's column order, with predicted_warm/
# combined_score inserted right after enrichment_warm.
OUTPUT_COLUMNS = [
    "rank",
    "id",
    "species",
    "enrichment_warm",
    "predicted_warm",
    "combined_score",
    "TSS_dist",
    "gene_function",
    "chromosome",
    "start",
    "end",
    "region",
    "closest_TSS",
    "sequence",
]


def find_low_warm_actual_and_predicted(input_tsv: str, predictions_csv: str, top_n: int = 100) -> pd.DataFrame:
    """Return the top_n test-split rows by max(enrichment_warm, predicted_warm), ascending."""
    df = pd.read_csv(input_tsv, sep="\t")
    df = df[df["set"] == "test"].reset_index(drop=True)  # order preserved -- matches test_predictions.csv rows

    preds = pd.read_csv(predictions_csv)
    if len(preds) != len(df):
        raise ValueError(
            f"{predictions_csv} has {len(preds)} rows but {input_tsv}'s test split has {len(df)} -- "
            "these must come from the same checkpoint/input_tsv pair to align row-for-row"
        )

    df["predicted_warm"] = preds["warm_pred"].to_numpy()
    df["combined_score"] = df[["enrichment_warm", "predicted_warm"]].max(axis=1)

    masks = build_species_masks(df["id"].tolist())
    species = pd.Series("Other", index=df.index)
    for name, mask in masks.items():
        if name == "Other":
            continue
        species[mask] = name
    df["species"] = species

    top = df.sort_values("combined_score", ascending=True).head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    top = top.merge(load_genomic_coords(DEFAULT_MEDIA3), on="id", how="left")
    n_unmatched = top["chromosome"].isna().sum()
    if n_unmatched:
        unmatched_ids = top.loc[top["chromosome"].isna(), "id"].tolist()
        print(f"{n_unmatched} id(s) had no genomic coordinates in media-3.xlsx "
              f"(likely non-genomic controls): {unmatched_ids}")

    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gene-annotation", default=DEFAULT_GENE_ANNOTATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bed-dir", default=DEFAULT_BED_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    top = find_low_warm_actual_and_predicted(args.input, args.predictions, top_n=args.top_n)
    top = add_gene_annotation(top, args.gene_annotation)
    top = top[OUTPUT_COLUMNS]

    top.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote top {len(top)} actual-and-predicted low-warm sequences to {args.output}")
    print(top[["rank", "id", "species", "enrichment_warm", "predicted_warm",
               "combined_score", "closest_TSS", "gene_function"]].head(10).to_string(index=False))

    write_species_bed_files(top.rename(columns={"combined_score": "cold_minus_warm"}), args.bed_dir)


if __name__ == "__main__":
    main()
