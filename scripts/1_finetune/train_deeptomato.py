
from __future__ import annotations
import torch.nn as nn
import argparse
import json
from pathlib import Path
from typing import Any


from alphagenome_encoder_ft import (
    AlphaGenomeEncoderModel,
    PlantMPRADataset,
    TrainConfig,
    create_dataloader,
    create_optimizer,
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

class deng_model(nn.Module): 
    "Make a class for the Deng et al. model in pytorch, it has 3 Conv1D blocks (each: Conv -> BN -> ReLU -> MaxPool1) + two FC blocks)"
    def __init__(self, conv_layers: list[dict]): #model arch["conv_layers"]
        super().__init__() #calls initializer of parent class from inside a child class
        layers = []
        in_channels = 4 #for one-hot encoded DNA
        for spec in conv_layers: #for each of the 3 convolutional layers, take out the specs
            layers += [ #make a list of my layers from the config specs
                nn.Conv1d(in_channels, spec["out_channels"], spec["kernel_size"], padding=spec["padding"])
                nn.BatchNorm1d(spec["out_channels"]) #DO I WANT 1d? because this is looking at DNA sequence in one dimension?
                nn.ReLU(),
                nn.MaxPool1d(2) #size 2
            ] 
            in_channels = spec["out_channels"]
        self.conv = nn.Sequential(*layers) #look at what each of these lines is doing
        #now define the head, below is what was done for the deepstarr, but we want to adapt to use the json config values for this
        # self.head = nn.Sequential( #his is the second part with the two fully connected layers
        #     nn.Flatten(), #sequential process to put the data through
        #     nn.LazyLinear(256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        #     nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        #     nn.Linear(256, n_outputs),
        # )

    def forward(self, x): #UNDERSTAND this
        return self.head(self.conv(x))

    
# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Train encoder-only AlphaGenome MPRA model")
#     parser.add_argument("--config", type=str, default=None)

#     parser.add_argument("--input_tsv", type=str, default=None)
#     parser.add_argument("--pretrained_weights", type=str, default=None)
#     parser.add_argument("--checkpoint_dir", type=str, default=None)
#     parser.add_argument("--save_mode", type=str, default=None, choices=["minimal", "full", "head"])

#     parser.add_argument("--batch_size", type=int, default=None)
#     parser.add_argument("--sequence_length", type=int, default=None)
#     parser.add_argument("--barcode_min", type=int, default=None)
#     parser.add_argument(
#         "--construct_mode",
#         type=str,
#         default=None,
#         choices=["none", "adapters", "promoter", "promoter_barcode", "all"],
#     )
#     parser.add_argument("--num_workers", type=int, default=None)
#     parser.add_argument("--max_shift", type=int, default=None)
#     parser.add_argument("--subset_frac", type=float, default=None)
#     parser.add_argument("--rc_prob", type=float, default=None)
#     parser.add_argument("--shift_prob", type=float, default=None)
#     parser.add_argument("--reverse_complement", action=argparse.BooleanOptionalAction, default=None)
#     parser.add_argument("--random_shift", action=argparse.BooleanOptionalAction, default=None)
#     parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)

#     parser.add_argument("--pooling_type", type=str, default=None, choices=["flatten", "center", "mean", "sum", "max"])
#     parser.add_argument("--center_bp", type=int, default=None)
#     parser.add_argument("--hidden_sizes", type=str, default=None)
#     parser.add_argument("--dropout", type=float, default=None)
#     parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])

#     parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
#     parser.add_argument("--learning_rate", type=float, default=None)
#     parser.add_argument("--weight_decay", type=float, default=None)
#     parser.add_argument("--lr_scheduler", type=str, default=None, choices=["constant", "cosine", "plateau"])
#     parser.add_argument("--plateau_factor", type=float, default=None)
#     parser.add_argument("--plateau_patience", type=int, default=None)
#     parser.add_argument("--plateau_mode", type=str, default=None, choices=["min"])
#     parser.add_argument("--plateau_min_lr", type=float, default=None)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
#     parser.add_argument("--gradient_clip", type=float, default=None)

#     parser.add_argument("--num_epochs", type=int, default=None)
#     parser.add_argument("--early_stopping_patience", type=int, default=None)
#     parser.add_argument("--val_evals_per_epoch", type=int, default=None)
#     parser.add_argument("--second_stage_lr", type=float, default=None)
#     parser.add_argument("--second_stage_epochs", type=int, default=None)
#     parser.add_argument("--resume_from_stage2", action=argparse.BooleanOptionalAction, default=None)

#     parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
#     parser.add_argument("--wandb_project", type=str, default=None)
#     parser.add_argument("--wandb_name", type=str, default=None)
#     parser.add_argument("--device", type=str, default=None)
#     parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=None)
#     parser.add_argument("--seed", type=int, default=None)
#     parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=False)
#     return parser
    
def main() -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args() #parse arguments

    with open("configs/deeptomato.json") as f:
        config = json.load(f)

    conv_layer_specs = config["model_architecture"]["conv_layers"] #take these from the json file
    model = deng_model(conv_layers=conv_layer_specs)
    # with open(run_dir / "config.json", "w") as handle:
    #     json.dump(config.to_dict(), handle, indent=2)
    # print(f"Saved config to {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()