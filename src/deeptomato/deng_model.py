
from __future__ import annotations
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