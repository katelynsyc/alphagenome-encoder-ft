#!/usr/bin/env python
"""Report Ray Tune trial status for an experiment, reading persisted state from disk.

Works whether the sweep is still running or has already finished -- doesn't need
a live Ray cluster/head, just the experiment's storage_path/experiment_name dir
(the same one passed to train_ag_tune.py's --storage_path/--experiment_name).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from ray.tune import ExperimentAnalysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ray Tune trial status from disk")
    parser.add_argument("experiment_dir", help="e.g. results/ray_tune/ag_tune_smoketest_sep")
    args = parser.parse_args()

    analysis = ExperimentAnalysis(os.path.abspath(args.experiment_dir))
    trials = analysis.trials

    counts = Counter(str(t.status) for t in trials)
    print(f"{len(trials)} trials total: " + ", ".join(f"{n} {status}" for status, n in counts.items()))
    print()

    rows = []
    for t in trials:
        r = t.last_result or {}
        rows.append((r.get("val_pearson"), t.trial_id, str(t.status), r.get("stage"), r.get("training_iteration")))
    rows.sort(key=lambda row: (row[0] is None, -(row[0] or 0)))

    print(f"{'trial_id':<10} {'status':<12} {'stage':<8} {'iter':<6} val_pearson")
    for val_pearson, trial_id, status, stage, iteration in rows:
        vp = f"{val_pearson:.4f}" if val_pearson is not None else "n/a"
        print(f"{trial_id:<10} {status:<12} {str(stage):<8} {str(iteration):<6} {vp}")


if __name__ == "__main__":
    main()
