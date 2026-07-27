#!/usr/bin/env python
"""Report the best val_pearson across ALL trials physically present in an experiment's
storage_path, not just the ones the official Tuner/ExperimentAnalysis knows about.

Background: a Ray head can end up orphaned (e.g. after a NODE_FAIL where the old
head process survives on a node Slurm no longer tracks) and keep scheduling new
trials into the same storage_path/experiment_name, invisible to the official
Tuner's own experiment_state. Those trials still get checkpointed to shared
storage, just never registered in the official run's bookkeeping.

This script scans the storage_path directory itself for every train_fn_* trial dir,
cross-references trial IDs against what ExperimentAnalysis reports as officially
tracked, and reads result.json/progress.csv directly off disk for anything it finds
that ExperimentAnalysis doesn't know about ("rogue" trials). Self-contained -- no
dependency on any previously-exported snapshot, so it picks up new rogue trials
produced after this script was first run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def load_official(experiment_dir: str) -> dict[str, dict]:
    from ray.tune import ExperimentAnalysis

    analysis = ExperimentAnalysis(os.path.abspath(experiment_dir))
    rows = {}
    for t in analysis.trials:
        r = t.last_result or {}
        rows[t.trial_id] = {
            "trial_id": t.trial_id,
            "source": "official",
            "status": str(t.status),
            "stage": r.get("stage"),
            "iteration": r.get("training_iteration"),
            "val_pearson": r.get("val_pearson"),
        }
    return rows


def _read_last_result_json(trial_dir: str) -> dict | None:
    path = os.path.join(trial_dir, "result.json")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    return last


def _read_last_result_csv(trial_dir: str) -> dict | None:
    path = os.path.join(trial_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def scan_disk_trials(experiment_dir: str) -> dict[str, dict]:
    """Every train_fn_<trial_id>_... directory under experiment_dir, keyed by trial_id,
    with val_pearson/stage/iteration read directly from its own result.json/progress.csv."""
    rows = {}
    for trial_dir in glob.glob(os.path.join(experiment_dir, "train_fn_*")):
        name = os.path.basename(trial_dir)
        # train_fn_<8-hex-trial_id>_<n>_<hyperparams>_<timestamp>
        parts = name.split("_")
        if len(parts) < 3:
            continue
        trial_id = parts[2]

        result = _read_last_result_json(trial_dir) or _read_last_result_csv(trial_dir) or {}
        vp = result.get("val_pearson")
        rows[trial_id] = {
            "trial_id": trial_id,
            "trial_dir": trial_dir,
            "stage": result.get("stage"),
            "iteration": result.get("training_iteration"),
            "val_pearson": float(vp) if vp not in (None, "") else None,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", help="e.g. results/ray_tune/ag_hpsweep_1000")
    parser.add_argument("--csv", help="Optional path to write the combined, sorted table as CSV")
    parser.add_argument("--top", type=int, default=20, help="How many rows to print (default 20)")
    args = parser.parse_args()

    official = load_official(args.experiment_dir)
    on_disk = scan_disk_trials(args.experiment_dir)

    combined = []
    for trial_id, disk_row in on_disk.items():
        if trial_id in official:
            row = dict(official[trial_id])
            # Disk read can be more current than the last persisted experiment_state
            # snapshot for genuinely-official trials too (snapshot writes lag live
            # state, per the "slow experiment checkpoint sync" warnings in the head
            # log) -- prefer disk if it has a result and the snapshot doesn't.
            if row.get("val_pearson") is None and disk_row.get("val_pearson") is not None:
                row["val_pearson"] = disk_row["val_pearson"]
                row["stage"] = disk_row["stage"]
                row["iteration"] = disk_row["iteration"]
        else:
            row = {
                "trial_id": trial_id,
                "source": "rogue",
                "status": "TERMINATED" if disk_row["val_pearson"] is not None else "UNKNOWN",
                "stage": disk_row["stage"],
                "iteration": disk_row["iteration"],
                "val_pearson": disk_row["val_pearson"],
            }
        combined.append(row)

    # Official trials ExperimentAnalysis knows about but with no on-disk train_fn_*
    # dir at all (shouldn't normally happen, but don't silently drop them).
    for trial_id, row in official.items():
        if trial_id not in on_disk:
            combined.append(row)

    n_official = sum(1 for c in combined if c["source"] == "official")
    n_rogue = sum(1 for c in combined if c["source"] == "rogue")

    scored = [c for c in combined if c["val_pearson"] is not None]
    unscored = [c for c in combined if c["val_pearson"] is None]
    scored.sort(key=lambda c: -c["val_pearson"])

    print(f"Total combined trials: {len(combined)} (official={n_official}, rogue={n_rogue})\n")
    print(f"{'trial_id':<10} {'source':<9} {'stage':<8} {'iter':<6} val_pearson")
    for c in scored[: args.top]:
        print(f"{c['trial_id']:<10} {c['source']:<9} {str(c['stage']):<8} {str(c['iteration']):<6} {c['val_pearson']:.4f}")

    if scored:
        best = scored[0]
        print(f"\nBEST OVERALL: {best['trial_id']} ({best['source']}) val_pearson={best['val_pearson']:.4f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trial_id", "source", "status", "stage", "iteration", "val_pearson"])
            w.writeheader()
            for c in scored + unscored:
                w.writerow({k: c.get(k) for k in ("trial_id", "source", "status", "stage", "iteration", "val_pearson")})
        print(f"\nWrote {len(combined)} rows to {args.csv}")


if __name__ == "__main__":
    main()
