#!/usr/bin/env python
"""Train an encoder-only MPRA model on the tomato MPRA, modeled off of code in train_mpra.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    PlantMPRADataset,
    TrainConfig,
    add_metrics_to_history,
    create_dataloader,
    create_deng_splits,
    create_jores_splits,
    create_optimizer,
    create_random_splits,
    create_scheduler,
    evaluate,
    load_checkpoint,
    load_train_config,
    merge_train_config,
    parse_hidden_sizes,
    run_training_stage,
    run_two_stage_training,
    scheduler_stepper,
)



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train encoder-only AlphaGenome MPRA model")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--input_tsv", type=str, default=None)
    parser.add_argument("--train_txt", type=str, default=None)
    parser.add_argument("--test_txt", type=str, default=None)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--save_mode", type=str, default=None, choices=["minimal", "full", "head"])

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--barcode_min", type=int, default=None)
    parser.add_argument("--barcode_min_eval", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_shift", type=int, default=None)
    parser.add_argument("--subset_frac", type=float, default=None)
    parser.add_argument("--rc_prob", type=float, default=None)
    parser.add_argument("--shift_prob", type=float, default=None)
    parser.add_argument("--reverse_complement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--random_shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--val_chroms", type=str, nargs="+", default=None)
    parser.add_argument("--test_chroms", type=str, nargs="+", default=None)
    parser.add_argument("--weight_scheme", type=str, default=None, choices=["log", "sqrt", "fourthroot", "sqrt_log", "lin", "none"])
    parser.add_argument("--split_mode", type=str, default=None, choices=["chrom", "random", "deng", "jores"])
    parser.add_argument("--train_frac", type=float, default=None)
    parser.add_argument("--val_frac", type=float, default=None)

    parser.add_argument("--pooling_type", type=str, default=None, choices=["flatten", "center", "mean", "sum", "max"])
    parser.add_argument("--center_bp", type=int, default=None)
    parser.add_argument("--hidden_sizes", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--head_type", type=str, default=None, choices=["mpra", "deepstarr", "deeptomato"])
    parser.add_argument("--num_outputs", type=int, default=None)

    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["constant", "cosine", "plateau"])
    parser.add_argument("--plateau_factor", type=float, default=None)
    parser.add_argument("--plateau_patience", type=int, default=None)
    parser.add_argument("--plateau_mode", type=str, default=None, choices=["min"])
    parser.add_argument("--plateau_min_lr", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--val_evals_per_epoch", type=int, default=None)
    parser.add_argument("--second_stage_lr", type=float, default=None)
    parser.add_argument("--second_stage_epochs", type=int, default=None)
    parser.add_argument("--resume_from_stage2", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=False)
    return parser

def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data": {},
        "head": {},
        "optim": {},
        "stage": {},
        "checkpoint": {},
        "logging": {},
        "runtime": {},
    }

    data_pairs = {
        "input_tsv": args.input_tsv,
        "train_txt": args.train_txt,
        "test_txt": args.test_txt,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "barcode_min": args.barcode_min,
        "barcode_min_eval": args.barcode_min_eval,
        "reverse_complement": args.reverse_complement,
        "rc_prob": args.rc_prob,
        "random_shift": args.random_shift,
        "shift_prob": args.shift_prob,
        "max_shift": args.max_shift,
        "subset_frac": args.subset_frac,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "val_chroms": args.val_chroms,
        "test_chroms": args.test_chroms,
        "weight_scheme": args.weight_scheme,
        "split_mode": args.split_mode,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
    }
    head_pairs = {
        "pooling_type": args.pooling_type,
        "center_bp": args.center_bp,
        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes) if args.hidden_sizes is not None else None,
        "dropout": args.dropout,
        "activation": args.activation,
        "head_type": args.head_type,
        "num_outputs": args.num_outputs,
    }
    optim_pairs = {
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_mode": args.plateau_mode,
        "plateau_min_lr": args.plateau_min_lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clip": args.gradient_clip,
    }
    stage_pairs = {
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "val_evals_per_epoch": args.val_evals_per_epoch,
        "second_stage_lr": args.second_stage_lr,
        "second_stage_epochs": args.second_stage_epochs,
        "resume_from_stage2": args.resume_from_stage2,
    }
    checkpoint_pairs = {
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_dir": args.checkpoint_dir,
        "save_mode": args.save_mode,
    }
    logging_pairs = {
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
    }
    runtime_pairs = {
        "device": args.device,
        "use_amp": args.use_amp,
        "seed": args.seed,
    }

    for section_name, values in (
        ("data", data_pairs),
        ("head", head_pairs),
        ("optim", optim_pairs),
        ("stage", stage_pairs),
        ("checkpoint", checkpoint_pairs),
        ("logging", logging_pairs),
        ("runtime", runtime_pairs),
    ):
        overrides[section_name] = {key: value for key, value in values.items() if value is not None}
    return overrides


def _make_dataset(config: TrainConfig, split: str) -> PlantMPRADataset:
    use_augment = split == "train"

    return PlantMPRADataset(
        config.data.input_tsv,
        split=split,
        barcode_min=config.data.barcode_min,
        barcode_min_eval=config.data.barcode_min_eval,
        sequence_length=config.data.sequence_length,
        reverse_complement=config.data.reverse_complement if use_augment else False,
        rc_prob=config.data.rc_prob,
        random_shift=config.data.random_shift if use_augment else False,
        shift_prob=config.data.shift_prob,
        max_shift=config.data.max_shift,
        subset_frac=config.data.subset_frac,
        seed=config.runtime.seed,
        val_chroms=config.data.val_chroms,
        test_chroms=config.data.test_chroms,
        weight_scheme=config.data.weight_scheme,
        num_outputs=config.head.num_outputs,
    )


def _track_names_for(config: TrainConfig) -> list[str] | None:
    """Per-output condition names for per-track pearson, keyed by dataset/split_mode.

    Order must match each dataset's target tensor layout exactly. Returns None (falls
    back to a generic "pearson_trackN" naming in train.py) for anything not listed here.
    """
    if config.data.split_mode == "jores":
        return ["cold", "dark", "light", "warm", "maize"]  # matches JoresMPRADataset._targets order
    if config.head.num_outputs == 4:
        return ["leaf", "MG", "Br", "RR"]  # matches PlantMPRADataset's 4-output target order
    if config.head.num_outputs == 2:
        return ["leaf", "fruit"]  # matches PlantMPRADataset/DengMPRADataset's 2-output target order
    return None


def _make_datasets(config: TrainConfig) -> tuple[Any, Any, Any]:
    """Build (train, val, test) datasets, branching on config.data.split_mode."""
    if config.data.split_mode == "random":
        train_dataset, val_dataset, test_dataset = create_random_splits(
            config.data.input_tsv,
            train_frac=config.data.train_frac,
            val_frac=config.data.val_frac,
            seed=config.runtime.seed,
            sequence_length=config.data.sequence_length,
            barcode_min=config.data.barcode_min,
            barcode_min_eval=config.data.barcode_min_eval,
            reverse_complement=config.data.reverse_complement,
            rc_prob=config.data.rc_prob,
            random_shift=config.data.random_shift,
            shift_prob=config.data.shift_prob,
            max_shift=config.data.max_shift,
            subset_frac=config.data.subset_frac,
            num_outputs=config.head.num_outputs,
        )
        print(f"Random split: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset

    if config.data.split_mode == "deng":
        # DengMPRADataset (train.txt/test.txt) only ever has Leaf_activity/Fruit_activity --
        # there's no MG/Br/RR breakdown in that file format, so 4-output targets aren't possible here.
        if config.head.num_outputs == 4:
            raise ValueError(
                "split_mode='deng' only provides Leaf/Fruit targets; set head.num_outputs=2 "
                "or use split_mode='chrom'/'random' with a 4-column (Leaf/MG/Br/RR) input_tsv"
            )
        train_dataset, val_dataset, test_dataset = create_deng_splits(
            config.data.train_txt, #use the explicitly passed in splits
            config.data.test_txt,
            val_frac=config.data.val_frac,
            seed=config.runtime.seed,
            sequence_length=config.data.sequence_length,
            reverse_complement=config.data.reverse_complement,
            rc_prob=config.data.rc_prob,
            random_shift=config.data.random_shift,
            shift_prob=config.data.shift_prob,
            max_shift=config.data.max_shift,
        )
        print(f"Deng split: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset
    
    if config.data.split_mode == "jores":
        train_dataset, val_dataset, test_dataset = create_jores_splits(
            config.data.input_tsv,
            seed=config.runtime.seed,
            sequence_length=config.data.sequence_length,
            reverse_complement=config.data.reverse_complement,
            rc_prob=config.data.rc_prob,
            random_shift=config.data.random_shift,
            shift_prob=config.data.shift_prob,
            max_shift=config.data.max_shift,
        )
        print(f"Jores split: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
        return train_dataset, val_dataset, test_dataset

    train_dataset = _make_dataset(config, "train")
    val_dataset = _make_dataset(config, "val")
    test_dataset = _make_dataset(config, "test")

    all_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.chrom_stats(total=all_total)
    return train_dataset, val_dataset, test_dataset



def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = merge_train_config(load_train_config(args.config), _build_overrides(args))
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(config.runtime.seed)
    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)
    run_dir = Path(config.checkpoint.checkpoint_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    print(f"Saved config to {run_dir / 'config.json'}")

    print(f"Loading pretrained weights from {config.checkpoint.pretrained_weights}...")

    model = AlphaGenomeEncoderModel.from_pretrained(
        config.checkpoint.pretrained_weights, #path to pretrained weights
        config.head, #config for new prediction head you added
        device=device, #CPU or GPU
    )

    model.initialize_head(config.data.sequence_length, device) #new prediction head with correct input dims for linear layers
    model.eval() #sets mode to set initial state

    n_trainable = sum(p.numel() for p in model.head.parameters()) #only new head parameters
    n_total = sum(p.numel() for p in model.parameters()) #entire model parameters (backbone & head)
    print("AlphaGenomeEncoderModel created.")
    print(f"  Trainable (head)   : {n_trainable:,}")
    print(f"  Frozen (backbone)  : {n_total - n_trainable:,}")
    print(f"  Total parameters   : {n_total:,}")
    print(f"  Trainable fraction : {100 * n_trainable / n_total:.4f}%")
    print()
    print("Head architecture:")
    print(model.head)

    train_dataset, val_dataset, test_dataset = _make_datasets(config)
    track_names = _track_names_for(config)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True, #prevents learning order-dependent patterns
        num_workers=config.data.num_workers, #parallel data loading
        pin_memory=config.data.pin_memory, #GPU optimization
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False, #reproducible evaluation
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    print(f"Datasets loaded from {config.data.input_tsv}")
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")
    print(f"  Test batches  : {len(test_loader):,}")

    stage1_optimizer = create_optimizer(config.optim, model.trainable_parameters(include_encoder=False)) #only head is trainable in stage 1
    stage1_scheduler = create_scheduler(config.optim, stage1_optimizer, config.stage.num_epochs) #scheduler adjusts learning rate during training, defines when to step
    stage1_scheduler_step = scheduler_stepper(config.optim.lr_scheduler)

    wandb_epoch_logger = None
    if config.logging.use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_name,
                config=config.to_dict(),
            )

            def wandb_epoch_logger(metrics: dict[str, Any]) -> None:
                # Groups panels into two top-level wandb sections: "test/..." for the
                # final held-out evaluation (test_loss, test_pearson, ...), and
                # "train/..." for everything tracked epoch-by-epoch during training
                # (both train_ and val_ losses/pearsons). Nested by stage (stage1/stage2)
                # so two-stage runs don't mix their curves together.
                stage = str(metrics["stage"])
                epoch = float(metrics["epoch"])
                is_test = metrics.get("event") == "final_test"
                section = "test" if is_test else "train"
                strip_prefix = "test_" if is_test else "train_"
                payload = {"epoch": epoch}
                for key, value in metrics.items():
                    if key in {"stage", "epoch", "event"}:
                        continue
                    clean_key = key[len(strip_prefix):] if key.startswith(strip_prefix) else key
                    payload[f"{section}/{stage}/{clean_key}"] = value
                wandb.log(payload)
        except ImportError:
            print("wandb is not installed; continuing without wandb")
            config.logging.use_wandb = False

    if config.stage.second_stage_lr is not None: #has two stages of training, the first where the encoder is frozen and just training the head, in the second stage you can train the entire thing
        def stage2_optimizer_factory(model_obj):
            return create_optimizer(
                config.optim,
                model_obj.trainable_parameters(include_encoder=True),
                learning_rate=config.stage.second_stage_lr,
            )

        def stage2_scheduler_factory(optimizer):
            return create_scheduler(config.optim, optimizer, config.stage.second_stage_epochs)

        results = run_two_stage_training(
            model,
            train_loader,
            stage1_optimizer=stage1_optimizer,
            stage2_optimizer_factory=stage2_optimizer_factory,
            config=config,
            device=device,
            val_loader=val_loader,
            stage1_scheduler=stage1_scheduler,
            stage2_scheduler_factory=stage2_scheduler_factory,
            stage1_scheduler_step=stage1_scheduler_step,
            stage2_scheduler_step=scheduler_stepper(config.optim.lr_scheduler),
            track_names=track_names,
            epoch_callback=wandb_epoch_logger,
            show_progress=args.show_progress,
        )
    else:
        results = run_training_stage(
            model,
            train_loader,
            optimizer=stage1_optimizer,
            config=config,
            device=device,
            num_epochs=config.stage.num_epochs,
            stage="stage1",
            train_encoder=False,
            val_loader=val_loader,
            scheduler=stage1_scheduler,
            scheduler_step=stage1_scheduler_step,
            track_names=track_names,
            checkpoint_dir=run_dir / "stage1",
            epoch_callback=wandb_epoch_logger,
            show_progress=args.show_progress,
        )

    test_targets: list[tuple[str, dict[str, Any]]] = [("stage1", results)]
    if "stage2" in results:
        test_targets = [("stage1", results["stage1"]), ("stage2", results["stage2"])]

    final_stage = test_targets[-1][0]
    final_metrics: dict[str, float] | None = None
    final_epoch = 0.0
    for stage_name, stage_result in test_targets:
        checkpoint_path = stage_result.get("best_checkpoint_path")
        test_epoch = float(stage_result.get("best_epoch", 0))
        load_checkpoint(checkpoint_path, model, map_location=device)
        test_metrics = evaluate(model, test_loader, device, track_names=track_names, use_amp=config.runtime.use_amp)
        results[f"{stage_name}_test_metrics"] = test_metrics
        test_metric_parts = [f"test_{key}={value:.4f}" for key, value in sorted(test_metrics.items())]
        print(f"[{stage_name}] final test | epoch {test_epoch:g} | " + " | ".join(test_metric_parts))
        if wandb_epoch_logger is not None:
            test_payload = {"stage": stage_name, "epoch": test_epoch, "event": "final_test"}
            for key, value in test_metrics.items():
                test_payload[f"test_{key}"] = value
            wandb_epoch_logger(test_payload)
        if stage_name == final_stage:
            final_metrics = test_metrics
            final_epoch = test_epoch

    assert final_metrics is not None
    results["test_metrics"] = final_metrics
    results["history"]["test_loss"].append(final_metrics["loss"])
    results["history"]["test_pearson"].append(final_metrics.get("pearson", float("nan")))
    results["history"]["test_epoch"].append(final_epoch)
    add_metrics_to_history(results["history"], "test", final_metrics)

    with open(run_dir / "history.json", "w") as handle:
        json.dump(results["history"], handle, indent=2)

    if config.logging.use_wandb:
        import wandb

        wandb.finish()

    return results


if __name__ == "__main__":
    main()
