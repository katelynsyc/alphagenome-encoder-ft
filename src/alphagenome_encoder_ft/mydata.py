
from __future__ import annotations #so that you can reference class before its defined
import csv
from pathlib import Path #to handle pathfiles

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

#MPRA library adapters from Deng et al.https://doi.org/10.1093/plcell/koaf236
TOMATO_ADAPTER_UP = "ACTCACTATAGGGCGAATTG" #5' adapter seq for fwd
TOMATO_ADAPTER_DOWN = "GAAGTTCATTTCATTTGGAG" #3' adapter seq

def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]] #reverses sequence, then finds complements

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
    ) -> None: #catch errors that may arise from inputs
        if split not in self.DEFAULT_FOLD_SPLITS:
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
            list(train_chroms) if train_chroms is not None else [chrom for chrom in ALL_TOMATO_CHROMOSOMES if chrom not in eval_chroms] #infers the rest from total chromosomes
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
        self._targets = np.asarray(
            [[float(row["Leaf"]), float(row["MG"]), float(row["Br"]), float(row["RR"])] for row in rows],
            dtype=np.float32,
        )  # shape (N, 4): gene expression per tissue [Leaf, MG, Br, RR]


    def _read_tsv(self) -> list[dict[str, str]]:
        split_folds = { #assigns split_folds to whichever of these you selected with split var
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
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: #return one sample
        construct = f"{self.left_adapter}{self._payloads[index]}{self.right_adapter}" #adds the adapters
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        target = self._targets[index] 
        return torch.from_numpy(onehot), torch.from_numpy(np.asarray(target, dtype=np.float32)) #convert seq & expression level to pytorch tensor

    def chrom_stats(self) -> None: #prints the chromosome stats (which chroms included in each split, # seqs and % of dataset contained)
        counts = Counter(self._chroms)
        total = len(self._chroms)
        print(f"\n{self.split} split with {total} sequence across chromosomes: {self._chroms}")
        for chrom in sorted(counts):
            n = counts[chrom]
            print(f"  {chrom}: {n:5d} ({100 * n / total:.1f}%)")

        
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