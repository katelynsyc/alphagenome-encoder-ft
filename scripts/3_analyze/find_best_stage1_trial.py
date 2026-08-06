#!/usr/bin/env python
"""Find the ag_hpsweep_1000 trial with the best val_pearson at the end of stage 1.

"End of stage 1" is taken as the *best* val_pearson observed at any point during stage 1 for a
trial, not just its last epoch's value -- that's what train_ag.py monitors for early stopping and
is exactly what gets saved to checkpoints/<trial_id>/<hash>/stage1/best.pt. So the winning trial
always corresponds to a real checkpoint file on disk, not just a number in progress.csv. Each
trial's last stage-1 epoch value is also kept for comparison (stage1_last_val_pearson).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def scan_stage1_results(experiment_dir: str) -> pd.DataFrame:
    """One row per trial with its stage-1 best/last val_pearson, plus enough to locate the
    checkpoint and hyperparameters (params.json sits alongside progress.csv in the same trial dir)."""
    rows = []
    skipped = []
    for trial_dir in sorted(glob.glob(os.path.join(experiment_dir, "train_fn_*"))):
        name = os.path.basename(trial_dir)
        parts = name.split("_")
        if len(parts) < 3:
            continue
        trial_id = parts[2]

        progress_path = os.path.join(trial_dir, "progress.csv")
        if not os.path.exists(progress_path):
            skipped.append((trial_id, "no progress.csv"))
            continue

        df = pd.read_csv(progress_path)
        stage1 = df[df["stage"] == "stage1"]
        if stage1.empty:
            skipped.append((trial_id, "no stage1 rows"))
            continue

        best_row = stage1.loc[stage1["val_pearson"].idxmax()]
        last_row = stage1.loc[stage1["epoch"].idxmax()]

        params_path = os.path.join(trial_dir, "params.json")
        params = json.load(open(params_path)) if os.path.exists(params_path) else {}

        rows.append({
            "trial_id": trial_id,
            "trial_dir": trial_dir,
            "stage1_best_val_pearson": best_row["val_pearson"],
            "stage1_best_epoch": best_row["epoch"],
            "stage1_last_val_pearson": last_row["val_pearson"],
            "stage1_last_epoch": last_row["epoch"],
            **params,
        })

    if skipped:
        print(f"Skipped {len(skipped)} trial dirs (e.g. {skipped[:3]})")
    return pd.DataFrame(rows)


def report_best_trial(results: pd.DataFrame, experiment_dir: str) -> pd.Series:
    best = results.loc[results["stage1_best_val_pearson"].idxmax()]

    checkpoint_dirs = glob.glob(os.path.join(experiment_dir, "checkpoints", best["trial_id"], "*"))
    assert len(checkpoint_dirs) == 1, checkpoint_dirs
    stage1_best_ckpt = os.path.join(checkpoint_dirs[0], "stage1", "best.pt")

    hyperparam_cols = [c for c in results.columns if c not in (
        "trial_id", "trial_dir", "stage1_best_val_pearson", "stage1_best_epoch",
        "stage1_last_val_pearson", "stage1_last_epoch",
    )]

    print(f"Best stage-1 trial: {best['trial_id']}")
    print(f"  stage1_best_val_pearson = {best['stage1_best_val_pearson']:.4f} (epoch {int(best['stage1_best_epoch'])})")
    print(f"  stage1_last_val_pearson = {best['stage1_last_val_pearson']:.4f} (epoch {int(best['stage1_last_epoch'])})")
    print(f"  checkpoint: {stage1_best_ckpt}")
    print(f"  exists: {os.path.exists(stage1_best_ckpt)}")
    print("  hyperparameters:")
    for k in hyperparam_cols:
        print(f"    {k}: {best[k]}")

    return best


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiment_dir", nargs="?",
        default="results/ray_tune/ag_hpsweep_1000",
        help="Ray Tune experiment storage dir, e.g. results/ray_tune/ag_hpsweep_1000",
    )
    parser.add_argument("--top_n", type=int, default=15, help="How many top trials to print")
    parser.add_argument(
        "--csv_out", type=str, default=None,
        help="Optional path to write the full per-trial stage-1 table as a CSV",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    results = scan_stage1_results(args.experiment_dir)
    if results.empty:
        raise SystemExit(f"No scored trials found under {args.experiment_dir}")
    print(f"Scored {len(results)} trials\n")

    ranked = results.sort_values("stage1_best_val_pearson", ascending=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ranked.head(args.top_n).to_string(index=False))
    print()

    report_best_trial(results, args.experiment_dir)

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        ranked.to_csv(args.csv_out, index=False)
        print(f"\nSaved full table to {args.csv_out}")


if __name__ == "__main__":
    main()
