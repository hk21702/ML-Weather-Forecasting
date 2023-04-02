import torch
import torch.nn as nn

class DownSampler(nn.Module):
    """Layer to downsample context data to a given degree.
        Applied batch normalization multiple times to reduce the number of parameters.

        Downsamples width and height by a factor of 2.
    """

    def __init__(self, inputs: int, outputs: int = 256, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()

        self.module = nn.Sequential(
            nn.Conv2d(inputs,
                      160, 3, padding=1,
                      device=device),
            nn.MaxPool2d((2, 2), stride=2),
            nn.BatchNorm2d(160, device = device),
            nn.Conv2d(160,
                      outputs, 3, padding=1,
                      device=device),
            nn.BatchNorm2d(outputs, device=device),
            nn.Conv2d(outputs, outputs, 3, padding=1,
                      device=device),
            nn.MaxPool2d((2, 2), stride=2),
        )

    def forward(self, x):
        return self.module.forward(x)
