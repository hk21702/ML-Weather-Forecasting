import torch
from torch import nn

from network.layers.condition_time import ConditionTime
from network.layers.conv_lstm import ConvLSTM
from network.layers.down_sampler import DownSampler
from network.layers.simple_gru import SimpleGRU
from network.layers.simple_lstm import SimpleLSTM
from network.model_config import ModelConfig


class Model(nn.Module):
    """Main model layer"""

    def __init__(self,
                 dropout: float,
                 model_config: ModelConfig):
        super(Model, self).__init__()

        self.model_config = model_config

        self.hidden_dims = self.model_config.hidden_dims
        self.kernel_size = self.model_config.kernel_size
        self.drop = nn.Dropout(dropout)

        self.input_channels = self.model_config.input_chans
        self.output_chans = self.model_config.output_chans

        self.output_steps = self.model_config.target_steps

        self.encoder = DownSampler(self.input_channels, self.input_channels)

        if self.model_config.model_type == 'conv_lstm':
            self.primary_layer = ConvLSTM(
                input_dim=self.input_channels,
                hidden_dim=self.hidden_dims,
                kernel_size=self.kernel_size,
                num_layers=self.model_config.context_steps,
            )
        elif self.model_config.model_type == 'lstm':
            self.primary_layer = SimpleLSTM(
                input_size=self.input_channels,
                hidden_size=self.hidden_dims,
                num_layers=self.model_config.context_steps,
                dropout=dropout
            )
        elif self.model_config.model_type == 'gru':
            self.primary_layer = SimpleGRU(
                input_size=self.input_channels,
                hidden_size=self.hidden_dims,
                num_layers=self.model_config.context_steps,
                dropout=dropout
            )
        else:
            raise ValueError('Invalid model type')

        self.ct = ConditionTime(self.output_steps)

        self.head = nn.Conv2d(
            in_channels=self.hidden_dims,
            out_channels=self.output_steps * self.output_chans,
            kernel_size=self.kernel_size
        )

    def forward(self, x: torch.Tensor):
        res = []

        for step in self.output_steps:
            x = self.encoder(x)
            x = self.ct(x, step)
            x = self.drop(x)
            res, _ = self.primary_layer(x)

            res = self.head(res)

            res.append(res)

        res = torch.stack(res, dim=1).squeeze()
        return res
