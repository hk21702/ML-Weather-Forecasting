import torch
from torch import nn

from network.layers.condition_time import ConditionTime
from network.layers.conv_lstm import ConvLSTM
from network.layers.down_sampler import DownSampler
from network.model_config import ModelConfig


class ConvLSTMModel(nn.Module):
    """Convolutional LSTM model"""

    def __init__(self,
                 dropout: float,
                 model_config: ModelConfig):
        super(ConvLSTMModel, self).__init__()

        self.model_config = model_config

        self.hidden_dims = self.model_config.lstm_channels
        self.hidden_layers = self.model_config.hidden_layers
        self.kernel_size = self.model_config.kernel_size

        self.input_channels = self.model_config.input_chans
        self.output_chans = self.model_config.output_chans

        self.horizon = self.model_config.horizon

        self.encoder = DownSampler(self.input_channels, self.input_channels)

        self.primary_layer = ConvLSTM(
            input_dim=self.input_channels,
            hidden_dim=self.hidden_dims,
            kernel_size=self.kernel_size,
            num_layers=self.hidden_layers,
        )

        self.drop = nn.Dropout(dropout)

        self.ct = ConditionTime(self.horizon)

        self.head = nn.Conv2d(
            in_channels=self.hidden_dims,
            out_channels=self.horizon * self.output_chans,
            kernel_size=self.kernel_size
        )

    def forward(self, x: torch.Tensor):
        res = []

        for step in self.horizon:
            x = self.encoder(x)
            x = self.ct(x, step)
            x = self.drop(x)
            res, _ = self.primary_layer(x)

            res = self.head(res)

            res.append(res)

        res = torch.stack(res, dim=1).squeeze()
        return res
