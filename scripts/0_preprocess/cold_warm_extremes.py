"""Find sequences that are most cold-induced and most warm-repressed in
modelling_data_tamsACR.tsv, and write them to a table for manual review.

Ranking metric: enrichment_cold - enrichment_warm ("cold_minus_warm"), so the
top of the table is high cold enrichment paired with low warm enrichment.
Species is inferred from the id prefix using the same JORES_SPECIES_PREFIXES
mapping as species_masks.py, so labels stay consistent across scripts.

Genomic coordinates (chromosome, start, end) are pulled in from media-3.xlsx's
'ACR sequence library' sheet, which stores id/orientation as separate columns
(e.g. id="At-18204(PP)", orientation="rev") -- we rebuild the fused
"<id>_<orientation>" key used by modelling_data_tamsACR.tsv to join the two.
Non-genomic control sequences (e.g. "35S", "AB80") have no match and are left
with empty coordinate columns.

Nearest-gene info (region, closest_TSS, TSS_dist) is joined in from plantGREP's
own precomputed data/annotation/ACRs/tamsACR_annotation.tsv.gz
(https://github.com/tobjores/plantGREP) rather than re-derived with a fresh
GFF + `bedtools closest` -- that file already has each ACR's nearest gene TSS
computed from the real gene models used for the paper (a private Lippman-lab
annotation for Tomato, TAIR/Ensembl Plants for the others).

gene_function then looks up that gene's functional description from a local
GFF3 per species (see SPECIES_FUNCTION_SOURCES): for Arabidopsis/Maize/Sorghum
the description sits on the `gene` feature itself; for Tomato
(Tomato_annotation.gff3.gz, a Liftoff transfer) it instead sits on the `mRNA`
feature's Note= attribute, keyed back to the gene via Parent=gene:<id>. That
file isn't literally the paper's private Lippman-lab annotation, but its ids
are the same canonical SolycXXgXXXXXX namespace -- 99.93% of Tomato
closest_TSS ids resolved against it in a spot check -- so its descriptions
apply.

Finally, one BED6 file per species is written from the located hits (e.g. for
loading into a genome browser).
"""

import argparse
import gzip
import os
import re

import pandas as pd

from alphagenome_encoder_ft import build_species_masks

METADATA_DIR = "/grid/koo/home/kachu/projects/alphagenome-encoder-ft/metadata"
GENOMES_DIR = f"{METADATA_DIR}/genomes"
DEFAULT_INPUT = f"{METADATA_DIR}/modelling_data_tamsACR.tsv"
DEFAULT_MEDIA3 = f"{METADATA_DIR}/media-3.xlsx"
DEFAULT_GENE_ANNOTATION = f"{METADATA_DIR}/tamsACR_annotation.tsv.gz"
DEFAULT_OUTPUT = f"{METADATA_DIR}/cold_warm_extremes_top100.tsv"
DEFAULT_BED_DIR = f"{METADATA_DIR}/acr_bed"

# species (as labeled by build_species_masks) -> (GFF3 path, kind, attribute holding a
# human-readable function description).
#   "gene_attr": description sits on the `gene` feature's own attributes.
#   "mrna_note": description sits on the `mRNA` feature's Note=, keyed via Parent=gene:<id>
#                (Liftoff-style GFF3s, e.g. Tomato).
# None means no local source is configured for that species.
SPECIES_FUNCTION_SOURCES = {
    "Arabidopsis": (f"{GENOMES_DIR}/Araport11_GFF3_genes_transposons.20250813.gff", "gene_attr", "computational_description"),
    "Maize": (f"{GENOMES_DIR}/Maize_annotation.gff3.gz", "gene_attr", "description"),
    "Sorghum": (f"{GENOMES_DIR}/Sorghum_annotation.gff3.gz", "gene_attr", "description"),
    "Tomato": (f"{GENOMES_DIR}/Tomato_annotation.gff3.gz", "mrna_note", None),
}

ID_ATTR_RE = re.compile(r"ID=([^;]+)")
PARENT_GENE_ATTR_RE = re.compile(r"Parent=gene:([^;]+)")
NOTE_ATTR_RE = re.compile(r"Note=([^;]+)")
VERSION_SUFFIX_RE = re.compile(r"\.\d+$")
TSS_STRAND_RE = re.compile(r"\([+-]\)$")

OUTPUT_COLUMNS = [
    "rank",
    "id",
    "species",
    "cold_minus_warm",
    "enrichment_cold",
    "enrichment_warm",
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
    "set",
    "sequence",
]


def load_genomic_coords(media3_xlsx: str) -> pd.DataFrame:
    """id (fused with orientation, e.g. "At-18204(PP)_rev") -> chromosome/start/end."""
    coords = pd.read_excel(media3_xlsx, sheet_name="ACR sequence library", header=4)
    coords = coords[["id", "orientation", "chromosome", "start", "end"]].copy()
    coords["id"] = coords["id"].astype(str) + "_" + coords["orientation"].astype(str)
    return coords.drop(columns="orientation").drop_duplicates(subset="id")


def find_cold_warm_extremes(input_tsv: str, media3_xlsx: str, top_n: int = 100) -> pd.DataFrame:
    """Return the top_n rows with the highest (enrichment_cold - enrichment_warm)."""
    df = pd.read_csv(input_tsv, sep="\t")

    df["cold_minus_warm"] = df["enrichment_cold"] - df["enrichment_warm"]

    masks = build_species_masks(df["id"].tolist())
    species = pd.Series("Other", index=df.index)
    for name, mask in masks.items():
        if name == "Other":
            continue
        species[mask] = name
    df["species"] = species

    top = df.sort_values("cold_minus_warm", ascending=False).head(top_n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    top = top.merge(load_genomic_coords(media3_xlsx), on="id", how="left")
    n_unmatched = top["chromosome"].isna().sum()
    if n_unmatched:
        unmatched_ids = top.loc[top["chromosome"].isna(), "id"].tolist()
        print(f"{n_unmatched} id(s) had no genomic coordinates in media-3.xlsx "
              f"(likely non-genomic controls): {unmatched_ids}")

    return top


def load_gene_annotation(gene_annotation_tsv: str) -> pd.DataFrame:
    """base id (no orientation suffix, e.g. "At-18204(PP)") -> region/closest_TSS/TSS_dist,
    from plantGREP's precomputed per-ACR annotation table."""
    annot = pd.read_csv(gene_annotation_tsv, sep="\t")
    return annot[["id", "region", "closest_TSS", "TSS_dist"]].drop_duplicates(subset="id")


def _open_gff(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _load_gene_attr_functions(path: str, description_key: str) -> dict[str, str]:
    """Description sits directly on the `gene` feature's own attributes."""
    description_re = re.compile(rf"{description_key}=([^;]+)")
    functions = {}
    with _open_gff(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            id_match = ID_ATTR_RE.search(fields[8])
            if not id_match:
                continue
            gene_id = id_match.group(1).removeprefix("gene:")
            description_match = description_re.search(fields[8])
            if description_match:
                functions[gene_id] = description_match.group(1)
    return functions


def _load_mrna_note_functions(path: str) -> dict[str, str]:
    """Description sits on the `mRNA` feature's Note=, keyed back to its parent gene via
    Parent=gene:<id>. Gene/mRNA ids here carry a version suffix (e.g. "Solyc01g005000.3")
    that closest_TSS's bare ids don't, so it's stripped before keying the dict."""
    functions = {}
    with _open_gff(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "mRNA":
                continue
            parent_match = PARENT_GENE_ATTR_RE.search(fields[8])
            note_match = NOTE_ATTR_RE.search(fields[8])
            if not parent_match or not note_match:
                continue
            gene_id = VERSION_SUFFIX_RE.sub("", parent_match.group(1))
            functions.setdefault(gene_id, note_match.group(1))
    return functions


def load_gene_functions(species: str) -> dict[str, str]:
    """gene id (e.g. "AT1G01010", "Zm00001d027230", "Solyc01g005000") -> function
    description, for one species. Returns {} if no source is configured for the
    species, or its GFF3 isn't present locally (this is optional enrichment on top
    of tamsACR_annotation.tsv.gz's closest_TSS/TSS_dist/region, not required for
    those to work)."""
    source = SPECIES_FUNCTION_SOURCES.get(species)
    if source is None:
        return {}
    path, kind, description_key = source
    if not os.path.exists(path):
        print(f"{species}: {path} not found locally, skipping gene_function lookup for it")
        return {}
    if kind == "mrna_note":
        return _load_mrna_note_functions(path)
    return _load_gene_attr_functions(path, description_key)


def add_gene_annotation(top: pd.DataFrame, gene_annotation_tsv: str) -> pd.DataFrame:
    """Join in region/closest_TSS/TSS_dist, then look up a gene_function description
    for closest_TSS from each row's species-specific GFF3."""
    top = top.copy()
    top["_base_id"] = top["id"].str.replace(r"_(fwd|rev)$", "", regex=True)

    annotation = load_gene_annotation(gene_annotation_tsv)
    top = top.merge(annotation, left_on="_base_id", right_on="id", how="left", suffixes=("", "_annot"))
    n_unmatched = top["closest_TSS"].isna().sum()
    if n_unmatched:
        print(f"{n_unmatched} id(s) had no match in {gene_annotation_tsv} (likely non-genomic controls)")

    gene_functions = {sp: load_gene_functions(sp) for sp in top["species"].unique() if sp != "Other"}
    bare_gene_id = top["closest_TSS"].str.replace(TSS_STRAND_RE, "", regex=True)
    top["gene_function"] = [
        gene_functions.get(sp, {}).get(gene_id)
        for sp, gene_id in zip(top["species"], bare_gene_id)
    ]
    n_no_function = top["closest_TSS"].notna().sum() - top["gene_function"].notna().sum()
    if n_no_function:
        print(f"{n_no_function} row(s) have a closest_TSS gene but no function text available "
              f"locally -- check SPECIES_FUNCTION_SOURCES has a working path for that species'"
              f" GFF3, or look the gene up directly on TAIR/MaizeGDB/Phytozome/solgenomics.net")

    return top.drop(columns=["_base_id", "id_annot"], errors="ignore")


def write_species_bed_files(top: pd.DataFrame, bed_dir: str) -> None:
    """Write one BED6 file per species from rows with known coordinates.

    media-3.xlsx's start/end are 1-based closed coordinates (end - start + 1
    equals the 170 bp sequence length), so start is shifted down by 1 here to
    match BED's 0-based half-open convention. Chromosome names are written as
    bare numbers (e.g. "1") -- rename them to match whatever convention a
    downstream tool expects (e.g. "Chr1", "chr1"). Strand is left as "." since
    the "orientation" (fwd/rev) column reflects
    the reporter cloning direction, not a confirmed genomic strand.
    """
    os.makedirs(bed_dir, exist_ok=True)
    located = top.dropna(subset=["chromosome", "start", "end"])
    for species, group in located.groupby("species"):
        bed = pd.DataFrame({
            "chrom": group["chromosome"].astype(int).astype(str),
            "start": group["start"].astype(int) - 1,
            "end": group["end"].astype(int),
            "name": group["id"],
            "score": group["cold_minus_warm"].round(3),
            "strand": ".",
        }).sort_values(["chrom", "start"])
        path = f"{bed_dir}/{species.lower()}_acrs.bed"
        bed.to_csv(path, sep="\t", header=False, index=False)
        print(f"Wrote {len(bed)} intervals to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--media3", default=DEFAULT_MEDIA3)
    parser.add_argument("--gene-annotation", default=DEFAULT_GENE_ANNOTATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bed-dir", default=DEFAULT_BED_DIR)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    top = find_cold_warm_extremes(args.input, args.media3, top_n=args.top_n)
    top = add_gene_annotation(top, args.gene_annotation)
    top = top[OUTPUT_COLUMNS]

    top.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote top {len(top)} cold-vs-warm sequences to {args.output}")
    print(top[["rank", "id", "species", "cold_minus_warm", "closest_TSS", "gene_function"]].head(10).to_string(index=False))

    write_species_bed_files(top, args.bed_dir)


if __name__ == "__main__":
    main()
