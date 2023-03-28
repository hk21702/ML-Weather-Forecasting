"""Simple LSTM layer"""
import torch
from torch import nn


class SimpleLSTM(nn.Module):
    """Simple LSTM layer"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, dropout=dropout)

    def forward(self, x):
        return self.lstm(x)

    def init_hidden(self, batch_size: int):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
