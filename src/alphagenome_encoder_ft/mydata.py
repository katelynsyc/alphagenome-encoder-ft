
from __future__ import annotations #so that you can reference class before its defined
import csv
from pathlib import Path #to handle pathfiles

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import stats
import seaborn as sns

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

#MPRA library adapters from Deng et al.https://doi.org/10.1093/plcell/koaf236
TOMATO_ADAPTER_UP = "ACTCACTATAGGGCGAATTG" #5' adapter seq for fwd
TOMATO_ADAPTER_DOWN = "GAAGTTCATTTCATTTGGAG" #3' adapter seq

def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]] #reverses sequence, then finds complements

def _to_float(value: str) -> float:
    return float(value) if value != "" else np.nan #tsv writes missing values as an empty cell, not the text 'NaN'

def _compute_barcode_weights(barcodes: np.ndarray, scheme: str ="log", cap: float = 20.0, split: str = "") ->  np.ndarray:
    """Per-fragment sample weight, monotonic in barcode count, mean-1 normalised.
    NOTE: the table gives ONE global barcode count per fragment (not per stage),
    so the same weight applies to all four stage targets -- which is exactly what
    a one-row-per-fragment multitask model with sample_weight expects.

    scheme : 'log' -> log1p(min(barcodes, cap))
    'sqrt' -> sqrt(min(barcodes, cap))
    'lin' -> min(barcodes, cap)
    """
    if scheme == "none": #you can remove weighting of scheme
        return np.ones(len(barcodes), dtype=np.float32)
    b = np.minimum(barcodes.astype(float), cap) #make the max #barcodes to be 20, store in column of barcodes
    if scheme == "log":
        w = np.log1p(b) #log(1 + b), compresses large values more
    elif scheme == "sqrt":
        w = np.sqrt(b) #square root transformation
    elif scheme == "lin":
        w = b #just used capped values
    else:
        raise ValueError(scheme)
    w = w / w.mean() #divides the proportion by the mean of these transformed weights,  ensures weighted loss has similar magnitude to unweighted loss
    
    # plot_scatterplot(barcodes, w, split)
    # plot_heatmap_weights(barcodes, w, split)
    #plot_kde_weights(barcodes, w, split)
    return w.astype(np.float32)

# def plot_scatterplot(barcode_counts, weights, split):
#     print(barcode_counts.shape)
#     print(weights.shape)
#     plt.scatter(barcode_counts,weights)
#     #m, b, r_value, _, _ = stats.linregress(barcode_counts, weights)
#     #plt.plot(barcode_counts, m * barcode_counts + b, color="black", linewidth=1)

#     plt.title(f"Barcode Counts vs. Barcode Weights ({split})")
#     plt.xlabel("Unique Barcodes Count")
#     plt.ylabel("Barcode Weights")
#     plt.savefig(f'results/plots/barcodecountvsweight_{split}.png', dpi=300)
#     plt.close()

# def plot_heatmap_weights(barcode_counts, weights, split):
#     plt.figure(figsize=(10, 8))
#     plt.hist2d(barcode_counts, weights, bins=[50, 50], cmap='viridis', cmin=1)
#     plt.colorbar(label='Density (count)')
#     plt.xlabel('Barcode Count')
#     plt.ylabel('Weight')
#     plt.title(f'2D Density {split}: Barcode Count vs Weight')
#     plt.savefig(f'results/plots/barcodeweight_{split}heatmap.png', dpi=300)

# def plot_kde_weights(barcode_counts, weights, split):
#     plt.figure(figsize=(10, 8))
#     sns.kdeplot(x=barcode_counts, y=weights, cmap='rocket', fill=True, levels=20)
#     plt.xlabel('Barcode Count')
#     plt.ylabel('Weight')
#     plt.title(f'2D {split} Kernel Density: Barcode Count vs Weight')
#     plt.savefig(f'results/plots/barcodeweight_{split}kde.png', dpi=300)

class PlantMPRADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for tomatoMPRA TSV files."""

    CHROM_PREFIX = "SL4.0"  # stripped when comparing against short names like 'ch01' for ease of understanding

    DEFAULT_FOLD_SPLITS = {
        "train": ['ch01', 'ch02', 'ch04', 'ch05', 'ch07', 'ch08', 'ch09', 'ch10', 'ch11', 'ch12'],
        "val":   ['ch03'],
        "test":  ['ch06'],
    }
    ALL_TOMATO_CHROMOSOMES = ['ch01', 'ch02', 'ch03', 'ch04', 'ch05', 'ch06', 'ch07', 'ch08', 'ch09', 'ch10', 'ch11', 'ch12']

    def __init__(
            self,
            input_tsv: str | Path,
            split: str = "train", #default to use training dataset as the split if not specified
            train_chroms: list[str] | None = None,
            val_chroms: list[str] | None = None,
            test_chroms: list[str] | None = None,
            
            use_adapters: bool = True,
            left_adapter: str = TOMATO_ADAPTER_UP,
            right_adapter: str = TOMATO_ADAPTER_DOWN,
            sequence_length: int = 160, #or should this be the value with adapter length

            reverse_complement: bool = False, #for the training set we do want this
            rc_prob: float = 0.5, #probability of applying reverse complement augmentation
            random_shift: bool = False, #shift sequence positions
            shift_prob: float = 0.5, #prob applying random shift augmentation
            max_shift: int = 20, #min length of the adapter
            subset_frac: float = 1.0, #for debugging, uses only fraction of data
            seed: int = 42,
            barcode_min: int = 10,       # threshold for train split
            barcode_min_eval: int = 10,  # quality-control threshold for val/test splits
            weight_scheme = "log", #for weighted loss based on barcode
    ) -> None: #catch errors that may arise from inputs
        if split not in (*self.DEFAULT_FOLD_SPLITS, "all"):
            raise ValueError(f"Unknown split: {split!r}")
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not 0 < subset_frac <= 1:
            raise ValueError("subset_frac must be in (0, 1]")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")

        self.input_tsv = Path(input_tsv)
        self.split = split 
        self.use_adapters = bool(use_adapters)
        self.left_adapter = left_adapter if self.use_adapters else ""
        self.right_adapter = right_adapter if self.use_adapters else ""
        self.sequence_length = sequence_length
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)
        self.barcode_min = barcode_min if split == "train" else barcode_min_eval #can change threshold for training set
        self.barcode_min_eval = barcode_min_eval #for test and val sets, this is always the same 

        self.val_chroms = (
            list(val_chroms) if val_chroms is not None else list(self.DEFAULT_FOLD_SPLITS["val"])
        )
        self.test_chroms = (
            list(test_chroms) if test_chroms is not None else list(self.DEFAULT_FOLD_SPLITS["test"])
        )
        eval_chroms = set(self.val_chroms) | set(self.test_chroms)
        self.train_chroms = ( #store the folds in lists
            # list(train_chroms) if train_chroms is not None else list(self.DEFAULT_FOLD_SPLITS["train"])
            list(train_chroms) if train_chroms is not None else [chrom for chrom in self.ALL_TOMATO_CHROMOSOMES if chrom not in eval_chroms] #infers the rest from total chromosomes
        )

        if not self.input_tsv.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.input_tsv}")

        rows = self._read_tsv()

        if subset_frac < 1.0 and rows: #rows are not empty, then pick the subset % you want
            sample_size = max(1, int(round(len(rows) * subset_frac)))
            sample_indices = self._rng.choice(len(rows), size=sample_size, replace=False) #randomly select this # of samples
            rows = [rows[int(idx)] for idx in sorted(sample_indices.tolist())]
        
        self._payloads = [str(row["Sequence"]) for row in rows] #this makes a compiled list of all the candidate sequences tested in MPRA (variable insert seqs)
        self._chroms = [row["Chr"].removeprefix(self.CHROM_PREFIX) for row in rows] #all built from same rows list

        #input_tsv can either be the raw 4-condition table (Leaf, MG, Br, RR) or an
        #already-imputed table with a Fruit column (see data_prep.py:write_imputed_activity_tsv)
        if rows and "Fruit" in rows[0]:
            self._targets = np.asarray(
                [[_to_float(row["Leaf"]), _to_float(row["Fruit"])] for row in rows],
                dtype=np.float32,
            )  # shape (N, 2): [Leaf, Fruit]
        else:
            self._targets = np.asarray(
                [[_to_float(row["Leaf"]), np.mean([_to_float(row["MG"]), _to_float(row["Br"]), _to_float(row["RR"])])] for row in rows],
                dtype=np.float32,
            )  # shape (N, 2): [Leaf, mean(MG,Br,RR)]

        nan_rows = np.isnan(self._targets).any(axis=1)
        if nan_rows.any():
            example_fragments = [rows[i].get("Fragment", "?") for i in np.flatnonzero(nan_rows)[:5]]
            raise ValueError(
                f"{self.input_tsv} ({self.split} split) has {int(nan_rows.sum())} rows with NaN targets "
                f"(e.g. {example_fragments}); drop or impute them before using this dataset"
            )

        barcodes = np.array([float(row["Unique Barcodes"]) for row in rows], dtype=np.float32)
        self._weights = (
            _compute_barcode_weights(barcodes, scheme=weight_scheme, split=self.split)
            if weight_scheme and self.split == "train" #only compute weights for the training set
            else np.ones(len(rows), dtype=np.float32)
        )

    def _read_tsv(self) -> list[dict[str, str]]:
        if self.split == "all":
            split_folds = self.ALL_TOMATO_CHROMOSOMES
        else:
            split_folds = {
                "train": self.train_chroms,
                "val": self.val_chroms,
                "test": self.test_chroms,
            }[self.split]


        rows: list[dict[str, str]] = [] #list of dictionaries with keys/values as string
        with open(self.input_tsv, newline="") as handle: #so this expects a tsv separated by tabs that has rows like seq, mean_value, fold change, fold, rev
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row["Chr"].removeprefix(self.CHROM_PREFIX) not in split_folds:
                    continue 
                if int(float(row["Unique Barcodes"])) < self.barcode_min: #after you check if this
                    continue
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._payloads)
    
    def _augment(self, onehot: np.ndarray) -> np.ndarray:
        out = onehot
        if self.reverse_complement and self._rng.random() < self.rc_prob: #decide to reverse complement
            out = _reverse_complement_onehot(out)
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob: #random shift augmentation
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1)) #pick a shift based on max val
            out = np.roll(out, shift, axis=0) #shifts elements in array by specified shift
        return out
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: #return one sample
        construct = f"{self.left_adapter}{self._payloads[index]}{self.right_adapter}" #adds the adapters
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot) 
        target = self._targets[index]  # shape (4,): [Leaf, MG, Br, RR] or shape (2,): [Leaf, Fruit] depending on file loaded in from
        #leaf_fruit = np.array([target[0], target[1:4].mean()], dtype=np.float32)  # shape (2,): [Leaf, mean(MG,Br,RR)]


        weight = self._weights[index]
        #return torch.from_numpy(onehot), torch.from_numpy(target), torch.tensor(weight) #add third tensor for the weights
        return torch.from_numpy(onehot), torch.from_numpy(target), torch.tensor(weight)

    def chrom_stats(self, total: int | None = None) -> None: #prints the chromosome stats (which chroms included in each split, # seqs and % of dataset contained)
        counts = Counter(self._chroms)
        split_total = len(self._chroms)
        denom = total if total is not None else split_total
        print(f"\n{self.split} split ({split_total} sequences) — chromosomes: {sorted(set(self._chroms))}")
        for chrom in sorted(counts):
            n = counts[chrom]
            print(f"  {chrom}: {n:5d} ({100 * n / denom:.1f}% of all)")


class DengMPRADataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for Deng et al.'s pre-split train.txt/test.txt files.

    Those files only have {Name|ID, Sequence, Leaf_activity, Fruit_activity} columns --
    no Chr or Unique Barcodes -- so there's no chromosome filtering and no
    barcode-based sample weighting (weights are always 1).
    """

    def __init__(
            self,
            rows: list[dict[str, str]], #already-read rows, e.g. from csv.DictReader
            use_adapters: bool = True,
            left_adapter: str = TOMATO_ADAPTER_UP,
            right_adapter: str = TOMATO_ADAPTER_DOWN,
            sequence_length: int | None = 160,

            reverse_complement: bool = False,
            rc_prob: float = 0.5,
            random_shift: bool = False,
            shift_prob: float = 0.5,
            max_shift: int = 20,
            seed: int = 42,
    ) -> None:
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if not 0 <= rc_prob <= 1:
            raise ValueError("rc_prob must be in [0, 1]")
        if not 0 <= shift_prob <= 1:
            raise ValueError("shift_prob must be in [0, 1]")
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")

        self.use_adapters = bool(use_adapters)
        self.left_adapter = left_adapter if self.use_adapters else ""
        self.right_adapter = right_adapter if self.use_adapters else ""
        self.sequence_length = sequence_length
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self._rng = np.random.default_rng(seed)

        self._payloads = [str(row["Sequence"]) for row in rows]
        self._targets = np.asarray(
            [[_to_float(row["Leaf_activity"]), _to_float(row["Fruit_activity"])] for row in rows],
            dtype=np.float32,
        )  # shape (N, 2): [Leaf, Fruit]

        nan_rows = np.isnan(self._targets).any(axis=1)
        if nan_rows.any():
            raise ValueError(f"{int(nan_rows.sum())} rows have NaN Leaf_activity/Fruit_activity")

        self._weights = np.ones(len(rows), dtype=np.float32) #no barcode counts in this file format

    def __len__(self) -> int:
        return len(self._payloads)

    def _augment(self, onehot: np.ndarray) -> np.ndarray:
        out = onehot
        if self.reverse_complement and self._rng.random() < self.rc_prob: #decide to reverse complement
            out = _reverse_complement_onehot(out)
        if self.random_shift and self.max_shift > 0 and self._rng.random() < self.shift_prob: #random shift augmentation
            shift = int(self._rng.integers(-self.max_shift, self.max_shift + 1)) #pick a shift based on max val
            out = np.roll(out, shift, axis=0) #shifts elements in array by specified shift
        return out

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        construct = f"{self.left_adapter}{self._payloads[index]}{self.right_adapter}" #adds the adapters
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        target = self._targets[index]  # shape (2,): [Leaf, Fruit]
        weight = self._weights[index]
        return torch.from_numpy(onehot), torch.from_numpy(target), torch.tensor(weight)


def _read_deng_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def create_deng_splits(
    train_txt: str | Path,
    test_txt: str | Path,
    val_frac: float = 0.1,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple[Subset, Subset, DengMPRADataset]:
    """Build train/val/test splits directly from Deng et al.'s train.txt/test.txt.

    val_frac is a fraction of the combined train.txt + test.txt row count (so val
    ends up roughly the same size as test, since test.txt is itself ~10% of the
    total), and is carved out of train.txt at random; test.txt is used as-is for
    the test split. Augmentation is applied only to the training subset.
    """
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")

    train_rows = _read_deng_tsv(train_txt)
    test_rows = _read_deng_tsv(test_txt)

    n_val = int(round((len(train_rows) + len(test_rows)) * val_frac))
    if n_val >= len(train_rows):
        raise ValueError(
            f"val_frac={val_frac} implies {n_val} val rows, but train.txt only has {len(train_rows)} rows"
        )

    noaug_kwargs = {**dataset_kwargs, "reverse_complement": False, "random_shift": False}

    full_aug = DengMPRADataset(train_rows, seed=seed, **dataset_kwargs)
    full_noaug = DengMPRADataset(train_rows, seed=seed, **noaug_kwargs)
    test_dataset = DengMPRADataset(test_rows, seed=seed, **noaug_kwargs)

    n = len(full_aug)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    val_dataset = Subset(full_noaug, indices[:n_val])
    train_dataset = Subset(full_aug, indices[n_val:])

    return train_dataset, val_dataset, test_dataset


def create_random_splits(
    input_tsv: str | Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    **dataset_kwargs,
) -> tuple[Subset, Subset, Subset]:
    """Load the full TSV and randomly split into train/val/test Subsets.

    Augmentation is applied only to the training subset.
    Barcode weights are disabled for all splits.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be positive and sum to less than 1")

    full_aug = PlantMPRADataset(input_tsv, split="all", seed=seed, weight_scheme=None, **dataset_kwargs)

    eval_kwargs = {**dataset_kwargs, "reverse_complement": False, "random_shift": False, "weight_scheme": None}
    full_noaug = PlantMPRADataset(input_tsv, split="all", seed=seed, **eval_kwargs)

    n = len(full_aug)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    return (
        Subset(full_aug,   indices[:n_train]),
        Subset(full_noaug, indices[n_train : n_train + n_val]),
        Subset(full_noaug, indices[n_train + n_val :]),
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a standard PyTorch DataLoader."""

    return DataLoader( #iterable allows easier access to samples
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )