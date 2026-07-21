#!/usr/bin/env python
"""List Ray Tune trials for a sweep, ranked by best validation metric, with each trial's
resolved checkpoint paths -- so you can find the best trial and reload its weights.

Reads each trial's own stage1/stage2/best.pt directly (training_state["best_monitor"]/
["best_epoch"]) rather than Ray's ExperimentAnalysis/trial_dataframes: Trial.local_path and
.logdir both require a live ray.init() session in the installed Ray version, so they can't be
queried post-hoc from a plain script, and progress.csv can get rewritten fresh on some restarts
rather than appended cumulatively across them (see train_ag_tune.py's checkpoint_dir fix). Each
best.pt's own best_monitor is exactly what run_training_stage itself tracked as the best value
for that lineage, so it's the most reliable single source of truth for "how good did this trial
get" -- and it's the exact file you'd load anyway.

Usage:
    python scripts/2_test/list_tune_runs.py --experiment_dir results/ray_tune/ckpt_fix_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List Ray Tune trials ranked by best val_pearson, with resolved checkpoint paths")
    parser.add_argument("--experiment_dir", type=str, required=True, help="e.g. results/ray_tune/ckpt_fix_test")
    return parser


def _load_stage_summary(stage_dir: Path) -> dict[str, Any] | None:
    best_path = stage_dir / "best.pt"
    if not best_path.exists():
        return None
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    # best.pt deliberately omits training_state (see save_checkpoint's docstring) -- it only
    # ever carries the val_pearson it was saved for under "metrics", not a "best_monitor" field.
    training_state = ckpt.get("training_state") or {}
    metrics = ckpt.get("metrics") or {}
    return {
        "best_monitor": training_state.get("best_monitor", metrics.get("pearson")),
        "best_epoch": training_state.get("best_epoch", ckpt.get("epoch")),
    }


def find_trials(experiment_dir: Path) -> list[dict[str, Any]]:
    """One entry per <trial_id>/<training_run_id hash> pair under experiment_dir/checkpoints/."""

    trials = []
    for config_path in sorted(experiment_dir.glob("checkpoints/*/*/config.json")):
        run_dir = config_path.parent
        with open(config_path) as handle:
            config = json.load(handle)
        stage1 = _load_stage_summary(run_dir / "stage1")
        stage2 = _load_stage_summary(run_dir / "stage2")
        best_stage = "stage2" if stage2 is not None else ("stage1" if stage1 is not None else None)
        best = stage2 if best_stage == "stage2" else stage1
        trials.append(
            {
                "trial_id": run_dir.parent.name,
                "run_id": run_dir.name,
                "run_dir": run_dir,
                "config": config,
                "stage1": stage1,
                "stage2": stage2,
                "best_stage": best_stage,
                "best_monitor": best["best_monitor"] if best else None,
                "best_epoch": best["best_epoch"] if best else None,
            }
        )
    return trials


def main() -> None:
    args = build_arg_parser().parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    trials = find_trials(experiment_dir)
    if not trials:
        print(f"No trials found under {experiment_dir}/checkpoints/*/*/config.json")
        return

    trials.sort(key=lambda t: t["best_monitor"] if t["best_monitor"] is not None else float("-inf"), reverse=True)

    print(f"{len(trials)} trial(s) under {experiment_dir}\n")
    for trial in trials:
        print(f"trial {trial['trial_id']}  (run {trial['run_id']})")
        if trial["best_monitor"] is not None:
            print(f"  best val_pearson: {trial['best_monitor']:.4f}  (stage={trial['best_stage']}, epoch={trial['best_epoch']})")
        else:
            print("  best val_pearson: not available yet")
        for section in ("head", "optim", "stage"):
            print(f"  {section}: {trial['config'].get(section, {})}")
        print(f"  checkpoint_dir: {trial['run_dir']}")
        for stage in ("stage1", "stage2"):
            for name in ("best.pt", "last.pt"):
                if (trial["run_dir"] / stage / name).exists():
                    print(f"    {stage}/{name}")
        print()

    best = trials[0]
    if best["best_stage"] is not None:
        print(f"Best trial: {best['trial_id']} -- reload with:")
        print(f"  load_checkpoint('{best['run_dir'] / best['best_stage'] / 'best.pt'}', model)")


if __name__ == "__main__":
    main()
