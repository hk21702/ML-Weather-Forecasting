"""Simple GRU layer"""
import torch
from torch import nn


class SimpleGRU(nn.Module):
    """Simple GRU layer"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        return self.gru(x)

    def init_hidden(self, batch_size: int):
        return torch.zeros(1, batch_size, self.hidden_size)
