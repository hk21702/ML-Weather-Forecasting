import torch
from torch import nn
from axial_attention import AxialAttention, AxialPositionalEmbedding

from network.layers.condition_time import ConditionTime
from network.layers.distribute_time import DistributeTime
from network.layers.conv_gru import ConvGRU
from network.layers.down_sampler import DownSampler
from network.model_config import ModelConfig
from network.models.model_utils import get_activation_func


class ConvGRUModel(nn.Module):
    """Convolutional LSTM model"""

    def __init__(self,
                 model_config: ModelConfig,
                 wb_config):
        super().__init__()

        self.model_config = model_config

        self.hidden_dims = self.model_config.lstm_chans
        self.hidden_layers = self.model_config.hidden_layers
        self.kernel_size = self.model_config.kernel_size

        self.input_channels = self.model_config.input_chans
        self.output_chans = self.model_config.output_chans

        self.horizon = self.model_config.horizon

        self.encoder = DistributeTime(DownSampler(
            self.input_channels + self.horizon, self.input_channels))

        self.activation_fn = get_activation_func(wb_config.activation_fn)

        if isinstance(self.kernel_size, tuple):
            conv_gru_ksize = (self.kernel_size[0], self.kernel_size[1])
        else:
            conv_gru_ksize = (self.kernel_size, self.kernel_size)

        self.primary_layer = ConvGRU(
            input_dim=self.input_channels,
            hidden_dim=self.hidden_dims,
            kernel_size=conv_gru_ksize,
            num_layers=self.hidden_layers,
            activation_fn=self.activation_fn,
        )

        dropout_rate = wb_config.dropout_rate
        late_dropout_rate = wb_config.post_conv_dropout_rate

        if wb_config.dropout_type == 'dropout':
            self.drop = nn.Dropout(dropout_rate)
            self.late_drop = nn.Dropout(late_dropout_rate)
        elif wb_config.dropout_type == 'alpha_dropout':
            self.drop = nn.AlphaDropout(dropout_rate)
            self.drop = nn.AlphaDropout(late_dropout_rate)
        else:
            raise ValueError(
                f'Unknown dropout type: {wb_config.dropout_type}')

        self.ct = ConditionTime(self.horizon)

        shape = self.model_config.target_width

        self.axial_pos_emb = AxialPositionalEmbedding(
            dim=self.hidden_dims,
            shape=(shape, shape),
        )

        self.agg_attention = nn.Sequential(
            *[
                AxialAttention(dim=self.hidden_dims, dim_index=1)
                for _ in range(2)
            ]
        )

        self.head = nn.Conv2d(
            in_channels=self.hidden_dims,
            out_channels=self.output_chans,
            kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor, step: int):
        x = self.ct(x, step)

        x = self.encoder(x)
        x = self.drop(x)

        _, state = self.primary_layer(x)

        # Use the last state
        emb = self.axial_pos_emb(state[-1])
        x_i = self.agg_attention(emb)

        res = self.head(x_i)
        return res
