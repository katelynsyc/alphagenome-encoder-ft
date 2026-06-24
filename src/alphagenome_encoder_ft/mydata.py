
from __future__ import annotations #so that you can reference class before its defined
import csv
from pathlib import Path #to handle pathfiles

import numpy as np
from alphagenome_encoder_ft.constructs import ConstructSpec
import torch
from torch.utils.data import DataLoader, Dataset

from alphagenome_pytorch.utils.sequence import sequence_to_onehot

from .constructs import ConstructSpec

#MPRA library adapters from Deng et al.https://doi.org/10.1093/plcell/koaf236
TOMATO_ADAPTER_UP = "ACTCACTATAGGGCGAATTG" #5' adapter seq for fwd
TOMATO_ADAPTER_DOWN = "GAAGTTCATTTCATTTGGAG" #3' adapter seq

def _reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    return onehot[::-1, :][:, [3, 2, 1, 0]] #reverses sequence, then finds complements

class PlantMPRADataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for tomatoMPRA TSV files."""

    DEFAULT_FOLD_SPLITS = { #by chromosome splitting, 
        "train": ['SL4.0ch01', 'SL4.0ch02', 'SL4.0ch04', 'SL4.0ch05', 'SL4.0ch07', 'SL4.0ch08', 'SL4.0ch09', 'SL4.0ch10', 'SL4.0ch11', 'SL4.0ch12'], #80% of data for training
        "val": ['SL4.0ch03'], #10% for validation
        "test": ['SL4.0ch06'], #10% for testing
    }

    def __init__(
            self, 
            input_tsv: str | Path,
            split: str = "train", #default to use training dataset as the split if not specified
            train_folds: list[int] | None = None, 
            valid_folds: list[int] | None = None,
            test_folds: list[int] | None = None,
            
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
            barcode_min: int = 10 #default is the strict >= 10 unique barcodes, this should only apply to train
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

        self.train_folds = ( #store the folds in lists
            list(train_folds) if train_folds is not None else list(self.DEFAULT_FOLD_SPLITS["train"])
        )
        self.valid_folds = (
            list(valid_folds) if valid_folds is not None else list(self.DEFAULT_FOLD_SPLITS["val"])
        )
        self.test_folds = (
            list(test_folds) if test_folds is not None else list(self.DEFAULT_FOLD_SPLITS["test"])
        )

        if not self.input_tsv.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.input_tsv}")

        rows = self._read_tsv()

        if subset_frac < 1.0 and rows: #rows are not empty, then pick the subset % you want
            sample_size = max(1, int(round(len(rows) * subset_frac)))
            sample_indices = self._rng.choice(len(rows), size=sample_size, replace=False) #randomly select this # of samples
            rows = [rows[int(idx)] for idx in sorted(sample_indices.tolist())]
        
        self._payloads = [str(row["seq"]) for row in rows] #this makes a compiled list of all the candidate sequences tested in MPRA (variable insert seqs)
        self._targets = np.asarray([float(row["mean_value"]) for row in rows], dtype=np.float32) #array of corresponding mean values
        self._construct_lengths = [
            len(self.construct_spec.assemble_sequence(payload, mode=self.construct_mode))
            for payload in self._payloads #list of lengths of the variable seqs assembled w/adapters 
        ]


    def _read_tsv(self) -> list[dict[str, str]]:
        split_folds = { #assigns split_folds to whichever of these you selected with split var
            "train": self.train_folds,
            "val": self.valid_folds,
            "test": self.test_folds,
        }[self.split]


        rows: list[dict[str, str]] = [] #list of dictionaries with keys/values as string
        with open(self.input_tsv, newline="") as handle: #so this expects a tsv separated by tabs that has rows like seq, mean_value, fold change, fold, rev
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if int(row["rev"]) != 0: #0 is false so if it's a reverse complement, skip (because it wants to do it's own rev complement)
                    continue #skips
                if int(row["Chr"]) not in split_folds: #only keeps folds of this current split, do the chromosome check before the barcode filter
                    continue 
                if int(row["barcode_count"]) < self.barcode_min: #after you check if this
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
        construct = f"{self.left_adapter}{self._inserts[index]}{self.right_adapter}" #adds the adapters
        onehot = sequence_to_onehot(construct).astype(np.float32, copy=False)
        onehot = self._augment(onehot)
        onehot = self._pad_or_trim(onehot)
        target = self._targets[index]
        return torch.from_numpy(onehot), torch.from_numpy(np.asarray(target, dtype=np.float32)) #convert seq & expression level to pytorch tensor


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