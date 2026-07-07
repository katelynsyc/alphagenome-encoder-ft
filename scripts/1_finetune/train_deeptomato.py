
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import torch.nn as nn

import numpy as np

import torch


from alphagenome_encoder_ft import (
    PlantMPRADataset,
    TrainConfig,
    create_dataloader,
    create_random_splits,
    create_deng_splits,
    create_optimizer,
    merge_train_config,
    parse_hidden_sizes,
    save_checkpoint,
    load_checkpoint,
)

from deeptomato import DengConvModel, run_epoch, correlation_metrics, plot_scatterplot

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DeepTOMATO, legnet and AG MPRA models")
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
    parser.add_argument("--weight_scheme", type=str, default=None, choices=["log", "sqrt", "lin"])

    parser.add_argument("--pooling_type", type=str, default=None, choices=["flatten", "center", "mean", "sum", "max"])
    parser.add_argument("--center_bp", type=int, default=None)
    parser.add_argument("--hidden_sizes", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])

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
    "These override values using args passed in from terminal for the memory"
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
        "weight_scheme": args.weight_scheme,
    }
    head_pairs = {
        "pooling_type": args.pooling_type,
        "center_bp": args.center_bp,
        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes) if args.hidden_sizes is not None else None,
        "dropout": args.dropout,
        "activation": args.activation,
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
    ): #if there was a passed in arg, value is not None and makes a key
        overrides[section_name] = {key: value for key, value in values.items() if value is not None}
    return overrides


def _make_dataset(config: TrainConfig, split: str) -> PlantMPRADataset: 
    use_augment = split == "train"
 
    return PlantMPRADataset( #passing these from config file when making the dataset
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
    )



def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args() #parse arguments

   # load the JSON once — _model_architecture is read here for deng_model,
    # the rest goes to TrainConfig (which ignores _-prefixed keys)
    if args.config is not None:
        with open(args.config) as f:
            raw = json.load(f) #the dictionary from json
    else:
        raw = {} #if there wasn't anything passed i as the json file, it'll just use the defaults
    arch_config = raw.get("_model_architecture", {}) #if no key config passed in, arch_config will be {}

    try:
        config = merge_train_config(TrainConfig.from_dict(raw), _build_overrides(args)) #merge the explicitly passed values with json configs (explictly passed have priority)
       #config.validate() validate w/this func is specific to the alphagenome requirements, it breaks for the deng paper, 
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(config.runtime.seed) #we want to expand the seeds to be the same for many others, dataloader 
    #set all seeds to be the same
    #seed_everything(config.runtime.seed, workers=True) #seeds random, numpy, torch & for multiple DataLoader workers

    device = torch.device(config.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu")) #pick a device to run on
    print(f"Using device: {device}", flush=True)
    
    run_dir = Path(config.checkpoint.checkpoint_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(config.to_dict(), handle, indent=2) #this saves the hyperparameters per run into a config file
    print(f"Saved config to {run_dir / 'config.json'}")

    print(f"Loading pretrained weights from {config.checkpoint.pretrained_weights}...")

    model = DengConvModel(arch_config, config.head).to(device)

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
        )
        print(f"Random split: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test")
        print(train_dataset.indices[:5])
    elif config.data.split_mode == "deng":
        train_dataset, val_dataset, test_dataset = create_deng_splits(
            config.data.train_txt,
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
    else:
        train_dataset = _make_dataset(config, "train")
        val_dataset = _make_dataset(config, "val")
        test_dataset = _make_dataset(config, "test")

        all_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        for ds in [train_dataset, val_dataset, test_dataset]:
            ds.chrom_stats(total=all_total)

    train_loader = create_dataloader( #make DataLoader for each of these
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
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

    #optimizer and wandb logger to visualize it on the interface
    # stage1_optimizer = create_optimizer(config.optim, model.trainable_parameters(include_encoder=False)) #only head is trainable in stage 1
    # stage1_scheduler = create_scheduler(config.optim, stage1_optimizer, config.stage.num_epochs) #scheduler adjusts learning rate during training, defines when to step
    # stage1_scheduler_step = scheduler_stepper(config.optim.lr_scheduler)

    #wandb tracking
    wandb_epoch_logger = None
    if config.logging.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_name,
                config=config.to_dict(),        # logs all your hyperparams
            )

            def wandb_epoch_logger(metrics: dict[str, Any]) -> None:
                # `stage` is already "train" or "test" here, giving the two top-level
                # wandb sections directly. `correlation_metrics()` prefixes its own keys
                # with "val/" or "test/" -- strip that so we don't nest redundantly
                # (e.g. avoid "train/val/Leaf_pearson", just "train/Leaf_pearson").
                stage = str(metrics["stage"])
                epoch = int(metrics["epoch"])
                payload = {}
                for key, value in metrics.items():
                    if key in {"stage", "epoch"}:
                        continue
                    clean_key = key.split("/", 1)[-1] if "/" in key else key
                    payload[f"{stage}/{clean_key}"] = value
                wandb.log(payload, step=epoch)

        except ImportError:
            print("wandb is not installed; continuing without wandb")
            config.logging.use_wandb = False

    optimizer = create_optimizer(config.optim, model.parameters())

    #weigh the loss function based on the barcode weights
    loss_fn = nn.MSELoss(reduction='none') #[batch, 4] size of per sample loss, don't reduce to scalar yet
    

    history: dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val, best_checkpoint_path, epochs_no_improve = float("-inf"), None, 0


    for epoch in range(config.stage.num_epochs):
        tr_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, vp, vt = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        corr = correlation_metrics(vp, vt, "val")
        mean_pearson = float(np.mean([v for k, v in corr.items() if "pearson" in k])) #why is this mean is this across conditions

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_pearson={mean_pearson:.4f} " +
              " ".join(f"{k}={v:.4f}" for k, v in corr.items()))

        if wandb_epoch_logger is not None:
            wandb_epoch_logger({"stage": "train", "epoch": epoch,
                                "train_loss": tr_loss, "val_loss": val_loss, **corr}) #this wraps wandb.log

        # if val_loss < best_val: #using the minimized loss to find the best model
        #     best_val = val_loss
        #     best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        #     torch.save(best_state, run_dir / "best_model.pt") #saves weights of the best model to disk file
        #     epochs_no_improve = 0 #for early stopping if exceeds patience threshold
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= config.stage.early_stopping_patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
        if mean_pearson > best_val: #using pearson to find what the best model is
            best_val = mean_pearson
            best_checkpoint_path = save_checkpoint(
                run_dir / "best_model.pt",
                model,
                config=config,
                save_mode=config.checkpoint.save_mode,
                stage="stage1",
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_pearson": mean_pearson, **corr},
            )
            epochs_no_improve = 0 #for early stopping if exceeds patience threshold
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.stage.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    load_checkpoint(best_checkpoint_path, model, map_location=device)
    test_loss, tp, tt = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False)
    #plot_scatterplot(tp, tt, "test") #plot predictions versus actual
    test_corr = correlation_metrics(tp, tt, "test")

    print("TEST: " + " ".join(f"{k}={v:.4f}" for k, v in {"test_loss": test_loss, **test_corr}.items()))

    if wandb_epoch_logger is not None:
        wandb_epoch_logger({"stage": "test", "epoch": epoch,
                            "test_loss": test_loss, **test_corr})

    results = {
        "history": history,
        "test_metrics": {"loss": test_loss, **test_corr},
    }

    with open(run_dir / "history.json", "w") as handle: #saves the full training history, each train_loss and val_loss
        json.dump(results["history"], handle, indent=2)

    if config.logging.use_wandb:
        import wandb
        wandb.finish()

    return results




if __name__ == "__main__":
    main()