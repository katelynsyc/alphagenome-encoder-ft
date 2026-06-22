
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

class PlantMPRADataset(Dataset[tuple[torch.Tensor, torch.Tensor]])):
    """PyTorch Dataset for tomatoMPRA TSV files."""

    DEFAULT_FOLD_SPLITS = { #uses 10-fold cross-validation
        "train": [2, 3, 4, 5, 6, 7, 8, 9], #80% of data for training
        "val": [1], #10% for validation
        "test": [10], #10% for testing
    }

    def __init__(
            self, 
            tsv_path: str,
            split: "train", #default to use training dataset as the split if not specified
            train_folds: list[int] | None = None, 
            valid_folds: list[int] | None = None,
            test_folds: list[int] | None = None,
            construct_spec: ConstructSpec | None = None, #defines how to build DNA constructs
            construct_mode: str = "all",
            reverse_complement: bool = False, #for the training set we do want this
            rc_prob: float = 0.5, #probability of applying reverse complement augmentation
            random_shift: bool = False, #shift sequence positions
            shift_prob: float = 0.5, #prob applying random shift augmentation
            max_shift: int = 15, #IS THIS A VALUE I need to change from the lentiMPRA?
            sequence_length: int | None = None,
            subset_frac: float = 1.0, #for debugging, uses only fraction of data
            seed: int = 42,
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
        if construct_spec is None:
            raise ValueError("construct_spec must be provided")
        self.construct_spec = construct_spec
        self.construct_mode = self.construct_spec.validate_mode(construct_mode) #defined method to make sure it's one of these
        self.promoter_seq = self.construct_spec.promoter_seq
        self.barcode_seq = self.construct_spec.barcode_seq
        self.left_adapter_seq = self.construct_spec.left_adapter
        self.right_adapter_seq = self.construct_spec.right_adapter
        self.reverse_complement = reverse_complement
        self.rc_prob = rc_prob
        self.random_shift = random_shift
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.sequence_length = sequence_length
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
        if self.sequence_length is not None:
            too_long = [
                (idx, construct_length)
                for idx, construct_length in enumerate(self._construct_lengths)
                if construct_length > self.sequence_length #BUT WONT THIS BE TOO LONG, so seq length is imported as variable length + the two adapter length?
            ]
            if too_long:
                sample_idx, sample_length = too_long[0]
                raise ValueError(
                    "sequence_length is shorter than the assembled construct length "
                    f"for sample {sample_idx}: {self.sequence_length} < {sample_length}"
                )


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
                if int(row["fold"]) not in split_folds: #only keeps folds of this current split
                    continue 
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._payloads)
    
    ## continue going through existing methods of this here as compared to data.py + the class of the DeepSTARR dataset class too (is the deepstarr one the one I want)
    

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