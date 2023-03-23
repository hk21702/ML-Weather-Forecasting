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

    def crop_to_target(self, x):
        """Crops the input to the target region."""
        pass

    def crop_downsampled(self, x):
        """Crops the input to the downsampled region."""
        pass

    def merge_context(self, x):
        """Merges the full resolution target region and the downsampled context region.
        without any overlaping regions."""
        target_data = self.crop_to_target(x)
        downsampled_data = self.crop_downsampled(x)

        # Merge the two regions

        # Create numpy array of the sa
        pass

class Decoder():
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout,
                 experiment_args: ExperimentArgs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.experiment_args = experiment_args

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        prediction = self.fc(output.squeeze(0))
        prediction = self.dropout(prediction)
        return prediction, hidden
    
    def predictions_to_pandas(self, prediction):
        """Converts the predictions to a pandas dataframe with location and time labels."""
        pass


class Model(nn.Module):

    pass
