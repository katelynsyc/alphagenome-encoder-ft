"""Count how many test-set sequences fall into each species mask, and flag
any left over in an "Other" bucket.

Thin CLI wrapper -- the actual logic (build_species_masks/summarize_species_masks)
lives in alphagenome_encoder_ft.mydata, next to create_jores_splits, so this script
and scripts/3_analyze/saturation_mutagenesis.py share one implementation instead of
two copies that can drift out of sync.
"""

from alphagenome_encoder_ft import summarize_species_masks


def main():
    metadata_path = "/grid/koo/home/kachu/projects/alphagenome-encoder-ft/metadata"
    input_tsv = metadata_path + "/modelling_data_tamsACR.tsv"
    summarize_species_masks(input_tsv, split="test")


if __name__ == "__main__":
    main()
