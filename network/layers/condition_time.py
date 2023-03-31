import torch
from torch import nn


class ConditionTime(nn.Module):
    """Layer to condition time onto a specific step in the sequence"""

    def __init__(self, horizon: int, dims: int = 2) -> None:
        super(ConditionTime, self).__init__()
        self.seq_len_hor = horizon
        self.dims = dims

    def forward(self, x: torch.Tensor, step=0):
        batch_size, seq_len, channels, height, width = x.size()

        # Create the time coordinates
        times = self.c_time(x, step,
                            (height, width)).repeat(
            batch_size, seq_len, 1, 1, 1)

        x = torch.cat([x, times], dim=self.dims)

        assert x.shape[self.dims] == channels + self.seq_len_hor, \
            f"Expected {channels + self.seq_len_hor} channels, got {x.shape[self.dims]}"
        return x

    def c_time(self, x: torch.Tensor, i: int, size: tuple[int, int]):
        # Create the time coordinates
        times = torch.eye(self.seq_len_hor, dtype=x.dtype, device=x.device)[i]
        times = times.unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones(1, *size, dtype=x.dtype, device=x.device)
        return times * ones
