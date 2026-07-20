#!/usr/bin/env python
"""Ray Tune HPO driver for train_ag.py -- Optuna search + ASHA scheduling over the same
two-stage AlphaGenome-encoder training path used by train_ag.py, with per-trial checkpoint
directories and automatic resume across preemption.

All the actual checkpoint/resume logic lives in alphagenome_encoder_ft.train
(run_training_stage/run_two_stage_training's last.pt handling) -- this file only wires Ray
Tune's trial lifecycle on top of the existing train_ag.run() entrypoint, so there is exactly
one training path to maintain.

How resume works end to end:
  1. Each trial's config.checkpoint.checkpoint_dir is set to that trial's own Ray-managed
     directory (stable across relaunches of the same trial), so its stage1/stage2 best.pt/
     last.pt files always land in the same place.
  2. If the whole job is preempted and killed, everything above is already on disk (written
     incrementally during training, not held in memory).
  3. Resubmitting this exact command re-enters main() below, which calls Tuner.restore(...)
     instead of building a fresh Tuner if the experiment directory already exists -- Ray
     relaunches whichever trials were still running, handing back their original trial IDs
     (so they get their original directories back), and train_ag.run()'s own resume logic
     (via run_training_stage) picks each one up from its last.pt.

num_samples is the total sweep budget (how many hyperparameter combinations to try) -- Optuna
still decides *which* combination to try at each of those samples based on prior results, and
ASHA can still kill an individual trial early if it's clearly underperforming, but nothing
picks the total trial count for you; that has to come from here.

Submit:
  cd ~/projects/alphagenome-encoder-ft && python scripts/1_finetune/train_ag_tune.py \
    --config configs/ag_jores.json \
    --pretrained_weights /grid/koo/home/shared/models/alphagenome/torch/model_all_folds.safetensors \
    --input_tsv metadata/modelling_data_tamsACR.tsv \
    --experiment_name ag-jores-sweep --num_samples 20 --gpus_per_trial 1
Resubmitting the same command (e.g. after a slow_nice preemption + requeue) resumes the same
experiment automatically -- see Tuner.can_restore() in main() below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# train_ag.py lives next to this file, not in an installed package -- make it importable
# regardless of how this script (or a Ray worker re-importing it) was launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ag  # noqa: E402 -- needs the sys.path tweak above first

import optuna  # noqa: E402
import ray  # noqa: E402
from ray import tune  # noqa: E402
from ray.tune import Tuner, TuneConfig  # noqa: E402
from ray.tune.schedulers import ASHAScheduler  # noqa: E402
from ray.tune.search.optuna import OptunaSearch  # noqa: E402

from alphagenome_encoder_ft import load_train_config, merge_train_config  # noqa: E402

# Ordinal grids for the tuned hyperparameters below -- sampled as indices (trial.suggest_int
# over range(len(LIST))) rather than trial.suggest_categorical(LIST) so TPE can exploit the
# ordering between values (index i is "between" i-1 and i+1) instead of treating each choice
# as an unrelated, equally-dissimilar bucket. Requires each list to be sorted ascending.
LR_LIST = sorted([
    mult * (10**exp)
    for exp in range(-7, -2)       # 10^-7 up to 10^-3
    for mult in [1, 3, 5, 8]
])
#make sure the killed version 
#it'll probably take a full week
#do a test run and want it check all of these things while i'm doing this
#set very low number of trials like 10 from the ray tune and make sure checkpoint is working, that i get weights from both phases

BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]
LINEAR_SIZES = [128, 256, 512, 1024, 2048, 2560, 4096]
WEIGHT_DECAYS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ray Tune HPO driver for train_ag.py")
    parser.add_argument("--config", type=str, required=True, help="Base config JSON, same schema as train_ag.py --config")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--storage_path", type=str, default="./results/ray_tune")
    parser.add_argument("--num_samples", type=int, default=20, help="Total number of hyperparameter trials to run (the sweep budget); Optuna samples each one, not a grid")
    parser.add_argument("--max_concurrent_trials", type=int, default=None)
    parser.add_argument("--gpus_per_trial", type=float, default=1.0)
    parser.add_argument("--cpus_per_trial", type=float, default=4.0)
    parser.add_argument("--metric", type=str, default="val_pearson") #optuna uses val pearson to optimize the values & also early stop for ASHA
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--max_epochs_per_trial", type=int, default=None, help="ASHA max_t; defaults to stage.num_epochs + stage.second_stage_epochs from --config")
    return parser


def _define_by_run_func(trial: "optuna.Trial") -> dict[str, Any] | None:
    """Conditional search space for OptunaSearch(space=...). Every parameter is defined via
    trial.suggest_* here -- Ray/Optuna drop each one into this trial's config dict under the
    name given, so train_fn() below reads them back out as tune_config["lr1_idx"], etc.
    Returning None (rather than a dict) means there are no extra constant values to merge in;
    everything static comes from --config via load_train_config in train_fn() instead."""

    # lr2 <= lr1 (not strictly smaller): nothing downstream requires lr2 < lr1 (config.py only
    # checks second_stage_lr > 0), and requiring strictly-smaller made lr1_idx=0 an impossible
    # draw -- lr2_idx would need to range over 0..-1. That used to be handled by raising
    # optuna.exceptions.TrialPruned() here, but Ray's OptunaSearch doesn't catch TrialPruned
    # raised from a define-by-run space function the way Optuna's own study.optimize() loop
    # does -- it propagates all the way up through Tuner.fit() uncaught and crashes the entire
    # multi-trial run, not just this one sample. Allowing lr2_idx == lr1_idx (both stages at
    # the same, possibly-smallest, rate) keeps the full LR_LIST range testable for lr1 without
    # ever hitting that dead end.
    lr1_idx = trial.suggest_int("lr1_idx", 0, len(LR_LIST) - 1)
    trial.suggest_int("lr2_idx", 0, lr1_idx)  # second-stage lr, no larger than lr1

    num_layers = trial.suggest_int("num_layers", 1, 2)
    layer1_idx = trial.suggest_int("layer1_idx", 0, len(LINEAR_SIZES) - 1)
    if num_layers == 2:
        # <= layer1_idx (not layer1_idx - 1): first layer must be larger or the same size
        trial.suggest_int("layer2_idx", 0, layer1_idx)

    trial.suggest_int("batch_size", 0, len(BATCH_SIZES) - 1)
    trial.suggest_float("s1_dropout", 0.0, 0.6, step=0.05)
    trial.suggest_float("s2_dropout", 0.0, 0.6, step=0.05)
    trial.suggest_int("weight_decay", 0, len(WEIGHT_DECAYS) - 1)

    return None


def train_fn(
    tune_config: dict[str, Any],
    *,
    base_config_path: str,
    pretrained_weights: str | None,
    input_tsv: str | None,
) -> None:
    """The per-trial Ray Tune trainable. Builds this trial's TrainConfig from the base config +
    sampled hyperparameters, points checkpointing at this trial's own directory, and delegates
    to train_ag.run() -- the exact same path train_ag.py uses standalone -- for training,
    resumable checkpointing, and final test evaluation.
    """

    hidden_sizes = [LINEAR_SIZES[tune_config["layer1_idx"]]]
    if tune_config["num_layers"] == 2:
        hidden_sizes.append(LINEAR_SIZES[tune_config["layer2_idx"]])

    overrides: dict[str, Any] = {
        "data": {"batch_size": BATCH_SIZES[tune_config["batch_size"]]},
        "head": {
            "hidden_sizes": hidden_sizes,
            "dropout": tune_config["s1_dropout"],
        },
        "optim": {
            "learning_rate": LR_LIST[tune_config["lr1_idx"]],
            "weight_decay": WEIGHT_DECAYS[tune_config["weight_decay"]],
        },
        "stage": {
            "second_stage_lr": LR_LIST[tune_config["lr2_idx"]],
            "second_stage_dropout": tune_config["s2_dropout"],
        },
        "checkpoint": {},
    }
    if pretrained_weights is not None:
        overrides["checkpoint"]["pretrained_weights"] = pretrained_weights
    if input_tsv is not None:
        overrides["data"]["input_tsv"] = input_tsv

    config = merge_train_config(load_train_config(base_config_path), overrides)
    # Stable across relaunches of this same trial (e.g. after preemption) -- this is what makes
    # every run of ray tune have its own folder, and where run_training_stage's own
    # resume logic finds stage1/stage2/last.pt on relaunch.
    config.checkpoint.checkpoint_dir = ray.train.get_context().get_trial_dir()
    config.logging.use_wandb = True  # per-trial metrics go to Ray/Tune and wandb 
    config.validate()

    def report_to_tune(payload: dict[str, Any]) -> None:
        metrics = {
            key: value for key, value in payload.items() if key not in {"stage", "epoch", "event"}
        }
        metrics["stage"] = payload["stage"]
        metrics["epoch"] = payload["epoch"]
        ray.train.report(metrics)

    train_ag.run(config, show_progress=False, epoch_callback=report_to_tune)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Ray Tune trial actors don't run with this process's cwd, so relative paths baked into
    # train_fn via tune.with_parameters below would otherwise resolve against whatever
    # directory the actor happens to start in instead of where this script was invoked from.
    args.config = str(Path(args.config).resolve())
    if args.input_tsv is not None:
        args.input_tsv = str(Path(args.input_tsv).resolve())

    base_config = load_train_config(args.config)
    # Every trial from _define_by_run_func always samples a real lr2 (second_stage_lr), so
    # this sweep always runs both stages regardless of what the base --config's
    # stage.second_stage_lr says -- unlike train_ag.py standalone, which treats a null
    # second_stage_lr as "stage 2 disabled".
    max_epochs_per_trial = args.max_epochs_per_trial or (
        base_config.stage.num_epochs + base_config.stage.second_stage_epochs
    )

    trainable = tune.with_parameters(
        train_fn,
        base_config_path=args.config,
        pretrained_weights=args.pretrained_weights,
        input_tsv=args.input_tsv,
    )
    trainable = tune.with_resources(trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial})

    sampler = optuna.samplers.TPESampler(seed=base_config.runtime.seed, multivariate=True, group=True)
    search_alg = OptunaSearch(space=_define_by_run_func, sampler=sampler, metric=args.metric, mode=args.mode)
    scheduler = ASHAScheduler(
        time_attr="epoch",
        max_t=max_epochs_per_trial,
        grace_period=15,  # min epochs a trial must run before ASHA can kill it
        reduction_factor=4,  # keep top 25% of each bracket
        brackets=1,
    )
    tune_config = TuneConfig(
        metric=args.metric,
        mode=args.mode,
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent_trials,
    )
    run_config = ray.train.RunConfig(name=args.experiment_name, storage_path=args.storage_path, failure_config=ray.train.FailureConfig(max_failures=4))

    experiment_dir = str(Path(args.storage_path).resolve() / args.experiment_name)
    if Tuner.can_restore(experiment_dir):
        print(f"Existing experiment found at {experiment_dir} -- resuming (Tuner.restore)")
        tuner = Tuner.restore(experiment_dir, trainable=trainable, resume_errored=True)
    else:
        print(f"No existing experiment at {experiment_dir} -- starting fresh")
        tuner = Tuner(trainable, tune_config=tune_config, run_config=run_config)

    results = tuner.fit()
    best_result = results.get_best_result(metric=args.metric, mode=args.mode)
    print(f"Best trial: {best_result.path}")
    print(f"Best {args.metric}: {best_result.metrics.get(args.metric)}")


if __name__ == "__main__":
    main()
