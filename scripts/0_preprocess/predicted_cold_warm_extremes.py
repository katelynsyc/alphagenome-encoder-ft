"""Find test-split sequences that are BOTH measured and model-predicted to be
strongly cold-induced / warm-repressed, and write them to a table for manual review.

Unlike cold_warm_extremes.py (which ranks by the measured cold_minus_warm across
the full train+val+test dataset), this only covers the test split -- that's the
only split evaluate_jores.py produces predictions for (best_test_eval/test_predictions.csv).

Ranking metric: combined_score = min(actual_cold_minus_warm, predicted_cold_minus_warm).
Taking the min (not e.g. the mean or product) means a row can only score highly if
BOTH the measured and the predicted cold-warm gap are large -- a row that's a big
gap in the data but a model miss (like Sl-sh2115_rev, see actual_cold_minus_warm_top100.tsv's
#1 hit) scores as low as its worse-performing side, and drops out of this table.

test_predictions.csv's row order matches summarize_species_masks(input_tsv, "test")'s
split_rows order (both filter modelling_data_tamsACR.tsv's rows by set == "test" in
original file order, with no resort) -- same alignment invariant find_row_indices_for_ids
in saturation_mutagenesis.py relies on.

Genomic coordinates and nearest-gene annotation are joined in the same way as
cold_warm_extremes.py -- see that module's docstring for where they come from.
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
DEFAULT_OUTPUT = f"{METADATA_DIR}/actual_and_predicted_cold_minus_warm_top100.tsv"
DEFAULT_BED_DIR = f"{METADATA_DIR}/acr_bed_predicted"

OUTPUT_COLUMNS = [
    "rank",
    "id",
    "species",
    "combined_score",
    "actual_cold_minus_warm",
    "predicted_cold_minus_warm",
    "enrichment_cold",
    "predicted_cold",
    "enrichment_warm",
    "predicted_warm",
    "enrichment_light",
    "enrichment_dark",
    "enrichment_maize",
    "chromosome",
    "start",
    "end",
    "region",
    "closest_TSS",
    "TSS_dist",
    "gene_function",
    "sequence",
]


def find_actual_and_predicted_extremes(input_tsv: str, predictions_csv: str, top_n: int = 100) -> pd.DataFrame:
    """Return the top_n test-split rows by min(actual_cold_minus_warm, predicted_cold_minus_warm)."""
    df = pd.read_csv(input_tsv, sep="\t")
    df = df[df["set"] == "test"].reset_index(drop=True)  # order preserved -- matches test_predictions.csv rows

    preds = pd.read_csv(predictions_csv)
    if len(preds) != len(df):
        raise ValueError(
            f"{predictions_csv} has {len(preds)} rows but {input_tsv}'s test split has {len(df)} -- "
            "these must come from the same checkpoint/input_tsv pair to align row-for-row"
        )

    df["predicted_cold"] = preds["cold_pred"].to_numpy()
    df["predicted_warm"] = preds["warm_pred"].to_numpy()
    df["actual_cold_minus_warm"] = df["enrichment_cold"] - df["enrichment_warm"]
    df["predicted_cold_minus_warm"] = df["predicted_cold"] - df["predicted_warm"]
    df["combined_score"] = df[["actual_cold_minus_warm", "predicted_cold_minus_warm"]].min(axis=1)

    masks = build_species_masks(df["id"].tolist())
    species = pd.Series("Other", index=df.index)
    for name, mask in masks.items():
        if name == "Other":
            continue
        species[mask] = name
    df["species"] = species

    top = df.sort_values("combined_score", ascending=False).head(top_n).copy()
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

    top = find_actual_and_predicted_extremes(args.input, args.predictions, top_n=args.top_n)
    top = add_gene_annotation(top, args.gene_annotation)
    top = top[OUTPUT_COLUMNS]

    top.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote top {len(top)} actual-and-predicted cold-vs-warm sequences to {args.output}")
    print(top[["rank", "id", "species", "combined_score", "actual_cold_minus_warm",
               "predicted_cold_minus_warm", "closest_TSS", "gene_function"]].head(10).to_string(index=False))

    write_species_bed_files(top.rename(columns={"combined_score": "cold_minus_warm"}), args.bed_dir)


if __name__ == "__main__":
    main()
