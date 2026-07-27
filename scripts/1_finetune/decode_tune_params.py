#!/usr/bin/env python
"""Decode Ray Tune trial params.json files into the actual hyperparameter values used for
training (see train_fn() in train_ag_tune.py). lr1/lr2/weight_decay are already the real
sampled values (continuous log-uniform floats); batch_size/hidden_sizes are recovered from
their exponents (value = 2**exponent, see BATCH_SIZE_EXP_*/LINEAR_SIZE_EXP_* in
train_ag_tune.py) -- so nothing here depends on a lookup table that could drift.

Usage:
  # single trial dir or params.json path
  python decode_tune_params.py results/ray_tune/ag_hpsweep_1000/train_fn_30aecdfb_7_.../params.json

  # every trial under an experiment dir, written to a CSV
  python decode_tune_params.py results/ray_tune/ag_hpsweep_1000 --csv decoded_params.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def decode_params(tune_config: dict[str, Any]) -> dict[str, Any]:
    """Mirror of the overrides dict built in train_ag_tune.train_fn()."""

    hidden_sizes = [2 ** tune_config["layer1_size_exp"]]
    if tune_config["num_layers"] == 2:
        hidden_sizes.append(2 ** tune_config["layer2_size_exp"])

    return {
        "batch_size": 2 ** tune_config["batch_size_exp"],
        "hidden_sizes": hidden_sizes,
        "s1_dropout": tune_config["s1_dropout"],
        "s2_dropout": tune_config["s2_dropout"],
        "learning_rate": tune_config["lr1"],
        "second_stage_lr": tune_config["lr2"],
        "weight_decay": tune_config["weight_decay"],
    }


def find_params_json(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if (path / "params.json").exists():
        return [path / "params.json"]
    return sorted(path.glob("*/params.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="params.json file, a single trial dir, or an experiment dir containing many trial dirs")
    parser.add_argument("--csv", type=Path, default=None, help="Write decoded params for all matched trials to this CSV instead of printing JSON")
    args = parser.parse_args()

    params_files = find_params_json(args.path)
    if not params_files:
        parser.error(f"no params.json found under {args.path}")

    rows = []
    for params_file in params_files:
        tune_config = json.loads(params_file.read_text())
        decoded = decode_params(tune_config)
        rows.append({"trial_dir": params_file.parent.name, **decoded})

    if args.csv is not None:
        fieldnames = ["trial_dir", "batch_size", "hidden_sizes", "s1_dropout", "s2_dropout", "learning_rate", "second_stage_lr", "weight_decay"]
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "hidden_sizes": ",".join(map(str, row["hidden_sizes"]))})
        print(f"Wrote {len(rows)} decoded trial(s) to {args.csv}")
    else:
        for row in rows:
            print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
