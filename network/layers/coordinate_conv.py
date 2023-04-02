"""Module involving classes relating to
    the coordinate convolutional layer and its implementation.
    
    Allows for the conversion of locational coordinate data to be used
    as a feature within the network."""
import torch
import torch.nn as nn


class CoordinateConv(nn.Module):
    """Convolutional Layer with added coordinates as input"""

    def __init__(self, inputs: int, outputs: int, kernel_size, ** kwargs) -> None:
        super().__init__()

        self.implementcoords = ImplementCoords()
        self.conv = nn.Conv2d(inputs + 2, outputs, kernel_size, **kwargs)

    def forward(self, x):
        x = self.implementcoords(x)
        x = self.conv(x)
        return x


class ImplementCoords(nn.Module):
    """Layer to implement the coordinates as input"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, in_tensor: torch.Tensor):
        """
            Args:
                in_tensor: (Batch, inputs, lon_size, lat_size) - 
                    lon being x-cord, lat being y-cord
        """
        batch_size, _, lon_size, lat_size = in_tensor.size()

        # Create the x and y coordinates
        xx_channel = torch.arange(lon_size).repeat(1, lon_size, 1)
        yy_channel = torch.arange(lat_size).repeat(
            1, lat_size, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (lon_size - 1)
        yy_channel = yy_channel.float() / (lon_size - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [in_tensor, xx_channel.type_as(
                in_tensor), yy_channel.type_as(in_tensor)],
            dim=1,
        )
        return ret
