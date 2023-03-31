import torch.nn as nn

from network.layers.coordinate_conv import CoordinateConv


class DownSampler(nn.Module):
    """Layer to downsample context data to a given degree.
        Applied batch normalization multiple times to reduce the number of parameters.
    """

    def __init__(self, inputs: int, outputs: int = 256) -> None:
        super().__init__()

        self.module = nn.Sequential(
            CoordinateConv(inputs,
                           160, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.BatchNorm2d(160),
            CoordinateConv(160,
                           outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            CoordinateConv(outputs, outputs, 3, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
        )

    def forward(self, x):
        return self.module.forward(x)
