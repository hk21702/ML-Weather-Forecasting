import torch
from torch import nn


class ConditionTime(nn.Module):
    """Layer to condition time onto a specific step in the sequence"""

    def __init__(self, horizon: int, dims: int = 2) -> None:
        super().__init__()
        self.horizon = horizon
        self.dims = dims

    def forward(self, x: torch.Tensor, step=0):
        batch_size, seq_len, channels, height, width = x.size()

        # Create the time coordinates
        times = torch.eye(self.horizon, dtype=x.dtype, device=x.device)[step]
        times = times.unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones(1, height, width, dtype=x.dtype, device=x.device)
        ct = (times * ones).repeat(
            batch_size, seq_len, 1, 1, 1
        )

        x = torch.cat([x, ct], dim=self.dims)

        assert x.shape[self.dims] == channels + self.horizon, \
            f"Expected {channels + self.horizon} channels, got {x.shape[self.dims]}"
        return x
