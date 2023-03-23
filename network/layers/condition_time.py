import torch
from torch import nn


class ConditionTime(nn.Module):
    """Layer to condition the time on the input data"""

    def __init__(self, seq_len_hor: int, dim: int = 2) -> None:
        super(ConditionTime, self).__init__()
        self.seq_len_hor = seq_len_hor
        self.dim = dim

    def forward(self, x: torch.Tensor, steps=0):
        batch_size, seq_len, channels, height, width = x.size()

        # Create the time coordinates
        times = self.c_time(x, steps,
                            (height, width)).repeat(
            batch_size, seq_len, 1, 1, 1)

        x = torch.cat([x, times], dim=self.dim)

        assert x.shape[self.dim] == channels + self.seq_len_hor, \
            f"Expected {channels + self.seq_len_hor} channels, got {x.shape[self.dim]}"
        return x

    def c_time(self, x: torch.Tensor, i: int, size: tuple[int, int]):
        # Create the time coordinates
        times = torch.eye(self.seq_len_hor, x.dtype, x.device)[i]
        times = times.unsequeeze(-1).unsequeeze(-1)
        ones = torch.ones(1, *size, x.dtype, x.device)
        return times * ones
