#!/usr/bin/env python
"""List training runs under a checkpoint base directory by their hyperparameters.

train_ag.py routes each run's checkpoint_dir through training_run_id(config) -- a hash
of every training-relevant config field -- so distinct hyperparameterizations each get
their own stage1/stage2/{last,best}.pt instead of overwriting or resuming onto a
previous run's. That hash is unique and collision-safe, but isn't meant to be read by
eye. This script is the lookup: it reads every run's saved config.json (and history.json,
if the run got far enough to write one) and prints only the hyperparameters that differ
across runs, so you can find "the run with batch_size=128, lr=5e-5" without opening each
folder by hand.

Usage:
    python scripts/2_test/list_runs.py
    python scripts/2_test/list_runs.py --checkpoint_base_dir ./results/models/checkpoints --sort_by best_val_pearson
    python scripts/2_test/list_runs.py --all_fields   # show every field, not just the differing ones
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List training runs by hyperparameters")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./results/models/checkpoints")
    parser.add_argument("--sort_by", choices=["mtime", "best_val_pearson"], default="mtime")
    parser.add_argument("--all_fields", action="store_true", help="Show every config field instead of just the ones that differ across runs")
    return parser


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _best_val_pearson(run_dir: Path) -> float | None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    with open(history_path) as handle:
        history = json.load(handle)
    values = [v for v in history.get("val_pearson", []) if v is not None]
    return max(values) if values else None


def find_runs(checkpoint_base_dir: Path) -> list[dict[str, Any]]:
    """One entry per run_id subfolder (anything with a config.json in it) directly under
    checkpoint_base_dir -- i.e. the training_run_id hash folders train_ag.py writes."""

    runs = []
    for config_path in sorted(checkpoint_base_dir.glob("*/config.json")):
        run_dir = config_path.parent
        with open(config_path) as handle:
            config = json.load(handle)
        runs.append(
            {
                "run_id": run_dir.name,
                "run_dir": run_dir,
                "config": _flatten(config),
                "mtime": config_path.stat().st_mtime,
                "best_val_pearson": _best_val_pearson(run_dir),
                "stage1_done": (run_dir / "stage1" / "best.pt").exists(),
                "stage2_done": (run_dir / "stage2" / "best.pt").exists(),
            }
        )
    return runs


def differing_keys(runs: list[dict[str, Any]]) -> list[str]:
    """Config keys whose value isn't identical across every run -- the ones actually worth
    looking at when eyeballing a sweep. All-identical keys (dataset paths, unrelated fixed
    knobs) are dropped."""

    if len(runs) <= 1:
        return sorted(runs[0]["config"].keys()) if runs else []
    all_keys: set[str] = set()
    for run in runs:
        all_keys.update(run["config"].keys())
    return [
        key
        for key in sorted(all_keys)
        if len({json.dumps(run["config"].get(key), default=str) for run in runs}) > 1
    ]


def main() -> None:
    args = build_arg_parser().parse_args()
    base_dir = Path(args.checkpoint_base_dir)
    runs = find_runs(base_dir)
    if not runs:
        print(f"No runs found under {base_dir.resolve()} (looking for */config.json)")
        return

    keys = sorted({k for run in runs for k in run["config"]}) if args.all_fields else differing_keys(runs)

    if args.sort_by == "best_val_pearson":
        runs.sort(key=lambda r: r["best_val_pearson"] if r["best_val_pearson"] is not None else float("-inf"), reverse=True)
    else:
        runs.sort(key=lambda r: r["mtime"])

    print(f"{len(runs)} run(s) under {base_dir.resolve()}\n")
    for run in runs:
        print(f"{run['run_id']}  ({run['run_dir']})")
        status = [name for name, done in (("stage1", run["stage1_done"]), ("stage2", run["stage2_done"])) if done]
        print(f"  status: {', '.join(status) + ' done' if status else 'no best.pt yet'}")
        if run["best_val_pearson"] is not None:
            print(f"  best val_pearson (from history.json): {run['best_val_pearson']:.4f}")
        for key in keys:
            print(f"  {key} = {run['config'].get(key)}")
        print()


if __name__ == "__main__":
    main()
