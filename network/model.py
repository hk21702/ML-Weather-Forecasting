import torch
import torch.nn as nn
import pandas as pd

from network.layers.down_sampler import DownSampler
from network.model_config import ModelConfig


class Model(nn.Module):
    """Main model layer"""

    def __init__(self, hidden_dims: int,
                 kernel_size: int, layers: int, heads: int,
                 dropout: float, model_config: ModelConfig):
        super(Model, self).__init__()

        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.drop = nn.Dropout(dropout)
        self.model_config = model_config

        self.encoder = DownSampler(
            len(self.model_config.feature_variables) +
            self.model_config.context_steps,
            hidden_dims)
