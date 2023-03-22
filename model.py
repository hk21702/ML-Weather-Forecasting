import torch
import torch.nn as nn
import pandas as pd


@dataclass
class ExperimentArgs():
    """Class to save experiment model arguments."""
    target_longitude: float
    target_latitude: float
    target_cubed_size: int

    context_cubed_size: int
    context_downsampling_factor: int
    context_time_size: int


class Encoder():
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 experiment_args: ExperimentArgs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.experiment_args = experiment_args

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        return output, hidden

    def crop_to_target(self):
        """Crops the input to the target region."""
        pass

    def downsample(self):
        """Downsamples the input by a given factor"""
        pass


class Model(nn.Module):
    pass
