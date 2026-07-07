
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn

class DengConvModel(nn.Module): #custom class that inherits from nn.Module
    "Make a class for the Deng et al. model in pytorch, it has 3 Conv1D blocks (each: Conv -> BN -> ReLU -> MaxPool1) + two FC blocks)"
    def __init__(self, model_architecture: dict, head_config):
        super().__init__() #calls initializer of parent class from inside a child class

        conv_layers = model_architecture["conv_layers"]
        pool_size = model_architecture["pooling_size"]

        layers = []
        in_channels = 4 #for one-hot encoded DNA
        for spec in conv_layers: #for each of the 3 convolutional layers, take out the specs
            layers += [
                nn.Conv1d(in_channels, spec["out_channels"], spec["kernel_size"], padding=spec["padding"]),
                nn.BatchNorm1d(spec["out_channels"]),
                nn.ReLU(),
                nn.MaxPool1d(pool_size)
            ]
            in_channels = spec["out_channels"] #to stack layers, input of next layer is same as output of previous
        self.conv = nn.Sequential(*layers)

        activation_fn = nn.ReLU() if head_config.activation == "relu" else nn.GELU() #gaussian otherwise?
        fc_layers = []
        for units in head_config.hidden_sizes:  # [256, 256] from tomatompra.json head section
            fc_layers += [
                nn.LazyLinear(units),
                activation_fn,
                nn.Dropout(head_config.dropout) 
            ]
        self.head = nn.Sequential(
            nn.Flatten(),
            *fc_layers,
            nn.LazyLinear(head_config.num_outputs)
        )


    def forward(self, x): #when you call model(X) this runs, passes through conv layers then head
        x = x.permute(0, 2, 1)  # [batch, length, 4] -> [batch, 4, length] for Conv1d
        return self.head(self.conv(x))

    @property
    def encoder(self) -> nn.Module:
        """Alias for `.conv` so this model satisfies the same `.encoder` / `.head`
        naming that `save_checkpoint`/`load_checkpoint` (alphagenome_encoder_ft.train)
        expect from AlphaGenomeEncoderModel -- lets DengConvModel reuse the same
        checkpoint contract without any changes to train.py."""
        return self.conv

    def initialize_lazy_layers(self, sequence_length: int, device: torch.device | str = "cpu") -> None:
        """Run one dummy forward pass so the `nn.LazyLinear` layers in `.head`
        materialize their weight shapes. Must be called before `load_state_dict`
        (mirrors AlphaGenomeEncoderModel.initialize_head)."""
        device = torch.device(device)
        self.to(device)
        with torch.no_grad():
            dummy = torch.zeros(1, sequence_length, 4, device=device)
            self(dummy)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_architecture: dict,
        *,
        device: torch.device | str | None = None,
    ) -> "DengConvModel":
        """Reconstruct a DengConvModel from a checkpoint saved via
        `alphagenome_encoder_ft.train.save_checkpoint`.

        `model_architecture` must be the same `_model_architecture` dict used at
        training time (conv_layers, pooling_size, ...) -- unlike AlphaGenome
        (a fixed, parameter-free factory), the conv trunk here is config-defined,
        so it isn't stored in the checkpoint payload and must be supplied again.
        """
        from alphagenome_encoder_ft.config import HeadConfig
        from alphagenome_encoder_ft.train import load_checkpoint

        device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        head_config = HeadConfig(
            head_type=raw.get("head_type", "deeptomato"),
            **raw["head_config"],
        )
        model = cls(model_architecture, head_config)

        sequence_length = raw["construct_config"].get("sequence_length")
        if sequence_length is None:
            sequence_length = raw["config"]["data"]["sequence_length"]
        model.initialize_lazy_layers(int(sequence_length), device)

        load_checkpoint(checkpoint_path, model, map_location=device)
        model.to(device)
        model.eval()
        return model