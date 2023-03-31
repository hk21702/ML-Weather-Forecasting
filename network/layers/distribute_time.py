import torch
from torch import nn


class DistributeTime(nn.Module):
    """"Distribute time dimension

    Expects (batch, time, channels, height, width)
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *ts, **kwargs):
        shape = ts[0].shape
        batch_size, seq_len = shape[0], shape[1]
        res = self.module(
            *[x.view(batch_size * seq_len, *x.shape[2:]) for x in ts], **kwargs)

        if isinstance(res, tuple):
            return tuple(
                torch.stack([t[i] for t in res], dim=1)
                for i in list(range(len(res[0])))
            )

        return res.view(batch_size, seq_len, *res.shape[1:])
