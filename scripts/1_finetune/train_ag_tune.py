#!/usr/bin/env python
"""Ray Tune HPO driver for train_ag.py -- Optuna search + ASHA scheduling over the same
two-stage AlphaGenome-encoder training path used by train_ag.py, with per-trial checkpoint
directories and automatic resume across preemption.

All the actual checkpoint/resume logic lives in alphagenome_encoder_ft.train
(run_training_stage/run_two_stage_training's last.pt handling) -- this file only wires Ray
Tune's trial lifecycle on top of the existing train_ag.run() entrypoint, so there is exactly
one training path to maintain.

How resume works end to end:
  1. Each trial's config.checkpoint.checkpoint_dir is set to storage_path/experiment_name/
     checkpoints/<trial_id> -- a path under Tune's persistent storage_path, not the trial's
     ephemeral local Ray-session directory (ray.tune.get_context().get_trial_dir() lives under
     /tmp/ray/session_<session-id>/..., and session-id changes every time the Ray cluster
     itself restarts, e.g. because the whole SLURM job got preempted and requeued -- so that
     path is NOT stable across the failure mode this is meant to survive, only across an
     in-place trial retry within the same still-running cluster). trial_id is stable across a
     Tuner.restore() of the same experiment even when the cluster is torn down and rebuilt, so
     stage1/stage2 best.pt/last.pt always land in, and are found again in, the same place.
  2. If the whole job is preempted and killed, everything above is already on disk (written
     incrementally during training, not held in memory).
  3. Resubmitting this exact command re-enters main() below, which calls Tuner.restore(...)
     instead of building a fresh Tuner if the experiment directory already exists -- Ray
     relaunches whichever trials were still running, handing back their original trial IDs
     (so they get their original checkpoint_dir back), and train_ag.run()'s own resume logic
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
import os
import shutil
import subprocess
import sys
import traceback
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
    storage_path: str,
    experiment_name: str,
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
    # Anchored under Tune's persistent storage_path (not ray.tune.get_context().get_trial_dir(),
    # which lives under the current Ray session's /tmp dir and is abandoned every time the whole
    # cluster restarts -- e.g. a slow_nice preemption) and keyed by trial_id, which Ray keeps
    # stable across a Tuner.restore() of this same trial. That's what makes this path stable
    # across preemption+requeue, so run_training_stage's own resume logic actually finds
    # stage1/stage2/last.pt on relaunch instead of starting every stage over from scratch.
    trial_id = ray.tune.get_context().get_trial_id()
    config.checkpoint.checkpoint_dir = str(Path(storage_path).resolve() / experiment_name / "checkpoints" / trial_id)
    # Every trial otherwise shares the same wandb_name from the base config, so the wandb Runs
    # table would show ~identical names for the whole sweep. Suffixing with trial_id makes each
    # run identifiable at a glance and matches Ray's own trial name (train_fn_<trial_id> in the
    # Tune results table / error paths), so a good-looking wandb curve maps straight back to its
    # checkpoint_dir above. Safe to change freely: stable_run_id()'s digest is keyed off
    # checkpoint_dir (already trial_id-based) too, so this doesn't affect resume identity.
    config.logging.wandb_name = f"{config.logging.wandb_name}-{trial_id}"
    config.logging.use_wandb = True  # per-trial metrics go to Ray/Tune and wandb
    config.validate()

    def report_to_tune(payload: dict[str, Any]) -> None:
        metrics = {
            key: value for key, value in payload.items() if key not in {"stage", "epoch", "event"}
        }
        metrics["stage"] = payload["stage"]
        metrics["epoch"] = payload["epoch"]
        ray.tune.report(metrics)

    train_ag.run(config, show_progress=False, epoch_callback=report_to_tune)


def _backup_ray_bookkeeping(experiment_dir: str) -> None:
    """Copy tuner.pkl and the newest experiment_state-*.json to a sibling <experiment_dir>_backup
    directory. Called right after a successful Tuner.restore(), and right after a fresh Tuner()
    construction (see main()) -- both are moments where whatever exists on disk was just written
    or parsed cleanly, so it's safe to copy: nothing rewrites tuner.pkl again for the life of the
    experiment (it's written exactly once, at construction), and nothing rewrites
    experiment_state.json again until Ray's next periodic sync. Both are written directly by Ray
    Tune with no atomic rename (unlike searcher-state-*.pkl/search_gen_state-*.json, which already
    are), so a preemption landing mid-write can corrupt them with no fallback of their own --
    tuner.pkl in particular has no historical copies at all. This is a plain external copy, not a
    fix to Ray's write path: corruption during the *next* write is still possible, this just gives
    a last-known-good snapshot to manually (or, via _recover_ray_bookkeeping_or_raise,
    automatically) restore from if that happens.
    """
    src_dir = Path(experiment_dir)
    backup_dir = Path(f"{experiment_dir}_backup")
    backup_dir.mkdir(parents=True, exist_ok=True)

    tuner_pkl = src_dir / "tuner.pkl"
    if tuner_pkl.exists():
        shutil.copy2(tuner_pkl, backup_dir / "tuner.pkl")

    state_files = sorted(src_dir.glob("experiment_state-*.json"))
    newest_state = state_files[-1] if state_files else None
    if newest_state is not None:
        shutil.copy2(newest_state, backup_dir / newest_state.name)

    print(
        f"Backed up Ray bookkeeping to {backup_dir}: "
        f"tuner.pkl={'yes' if tuner_pkl.exists() else 'missing'}, "
        f"experiment_state={newest_state.name if newest_state else 'missing'}"
    )


def _validate_experiment_state_or_raise(experiment_dir: str) -> None:
    """Parse the newest experiment_state-*.json the same way Ray Tune's own TuneController.resume()
    does, and raise if it's corrupted -- checked ourselves, deliberately, instead of relying on
    Ray Tune's own handling of this exact failure, which silently swallows it in both places it
    could otherwise surface: (1) inside Tuner.restore(), a parse failure here is caught by a bare
    `except Exception:` around ExperimentAnalysis construction and discarded with zero indication
    anything went wrong; (2) inside tuner.fit() -> TuneController.__init__, the same failure only
    re-raises if `fail_fast=True` is set on FailureConfig -- otherwise Ray just logs "Failed to
    restore the run state" and silently restarts the experiment from scratch, discarding all prior
    trial history with no exception and no warning anywhere our code could catch. fail_fast=True
    is not a viable path to catching it either: Ray requires max_failures=0 whenever fail_fast is
    set, and max_failures is our *per-trial* retry budget -- enabling fail_fast to catch this one
    rare corruption case would mean any single ordinary transient trial error (an OOM, a flaky
    node, anything) kills the entire sweep instead of just that trial. Validating the file
    ourselves, before ever calling Tuner.restore() or .fit(), sidesteps that tradeoff completely.

    A missing experiment_state-*.json is NOT an error here -- e.g. the very first preemption could
    land before .fit() ever reaches its first control-loop step, before this file is ever created
    at all. There's nothing to lose in that case, and Ray Tune correctly treats "no experiment
    state yet" as "nothing to resume," not corruption.
    """
    from ray.tune.utils.serialization import _loads_with_cloudpickle

    state_files = sorted(Path(experiment_dir).glob("experiment_state-*.json"))
    if not state_files:
        return
    _loads_with_cloudpickle(state_files[-1].read_bytes())


def _recover_ray_bookkeeping_or_raise(experiment_dir: str, restore_exc: Exception) -> None:
    """Called when Tuner.restore() raises -- most likely a preemption landed mid-write on Ray's
    own non-atomic tuner.pkl/experiment_state.json (see _backup_ray_bookkeeping above). If a prior
    successful restore left a backup, restore it and ask Slurm to requeue this job so a fresh
    process/Ray session picks up cleanly. Deliberately does NOT retry Tuner.restore() inside this
    same process -- this Ray session may have already partially ingested the failed state, so a
    clean requeue onto a fresh process is safer than trying to recover in place.

    No email/notification here by design: Slurm's --mail-type=REQUEUE would fire on every routine
    slow_nice/cpu_snice preemption requeue too, not just this rare corruption case, and there's no
    way to make it source-aware -- would flood the inbox with unrelated noise. Local mail transport
    was also confirmed down on this cluster (mail/sendmail return exit 0 but don't actually
    deliver), so a custom email isn't reliable either. Instead this writes a clearly-named
    RECOVERY_INCIDENT file that's easy to find/grep for next time the sweep is checked on.

    Re-raises restore_exc unchanged if there's no backup to recover from (nothing to do -- fails
    exactly as it would have without any of this).
    """
    backup_dir = Path(f"{experiment_dir}_backup")
    tuner_backup = backup_dir / "tuner.pkl"

    incident = (
        f"Tuner.restore() failed for {experiment_dir}\n"
        f"{type(restore_exc).__name__}: {restore_exc}\n\n"
        f"{traceback.format_exc()}"
    )
    print("=" * 70)
    print("RESTORE FAILED -- attempting recovery from backup")
    print(incident)
    print("=" * 70)

    if not tuner_backup.exists():
        print(f"No backup found at {backup_dir} -- nothing to recover from, failing normally")
        raise restore_exc

    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    incident_path = backup_dir / f"RECOVERY_INCIDENT_{job_id}.txt"
    incident_path.write_text(incident)

    src_dir = Path(experiment_dir)
    shutil.copy2(tuner_backup, src_dir / "tuner.pkl")
    state_backups = sorted(backup_dir.glob("experiment_state-*.json"))
    if state_backups:
        shutil.copy2(state_backups[-1], src_dir / state_backups[-1].name)
    print(f"Restored tuner.pkl / experiment_state from {backup_dir} (incident logged to {incident_path})")

    if job_id != "unknown":
        print(f"Requeuing job {job_id} to restart clean")
        subprocess.run(["scontrol", "requeue", job_id], check=False)
    else:
        print("SLURM_JOB_ID not set -- cannot requeue automatically, exiting for manual resubmit")
    sys.exit(75)  # distinct from 0 (success) and 1 (real failure) -- marks an auto-recovered restart


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
        storage_path=args.storage_path,
        experiment_name=args.experiment_name,
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
    run_config = ray.tune.RunConfig(name=args.experiment_name, storage_path=args.storage_path, failure_config=ray.tune.FailureConfig(max_failures=10))

    experiment_dir = str(Path(args.storage_path).resolve() / args.experiment_name)
    if Tuner.can_restore(experiment_dir):
        print(f"Existing experiment found at {experiment_dir} -- resuming (Tuner.restore)")
        try:
            _validate_experiment_state_or_raise(experiment_dir)
            tuner = Tuner.restore(experiment_dir, trainable=trainable, resume_errored=True)
        except Exception as exc:
            _recover_ray_bookkeeping_or_raise(experiment_dir, exc)
            raise  # unreachable if recovery above called sys.exit(); satisfies static analysis
        _backup_ray_bookkeeping(experiment_dir)
    else:
        print(f"No existing experiment at {experiment_dir} -- starting fresh")
        tuner = Tuner(trainable, tune_config=tune_config, run_config=run_config)
        # tuner.pkl is written synchronously inside this constructor (before .fit() runs) and,
        # per Ray Tune's source, is never written again for the life of this experiment -- so
        # this is the one and only moment tuner.pkl can ever be corrupted by a preemption. A
        # backup taken right here, right after that write is confirmed complete by the
        # constructor returning, permanently closes tuner.pkl's corruption risk for this
        # experiment. experiment_state.json doesn't exist yet at this point (it's only created
        # once .fit() starts its trial loop), so this call only captures tuner.pkl for now --
        # that's expected, not a bug; a future successful Tuner.restore() will pick up
        # experiment_state.json once it exists.
        _backup_ray_bookkeeping(experiment_dir)

    results = tuner.fit()
    best_result = results.get_best_result(metric=args.metric, mode=args.mode)
    print(f"Best trial: {best_result.path}")
    print(f"Best {args.metric}: {best_result.metrics.get(args.metric)}")


if __name__ == "__main__":
    main()
