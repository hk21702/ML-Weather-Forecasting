"""Module involving classes relating to
    the coordinate convolutional layer and its implementation.
    
    Allows for the conversion of locational coordinate data to be used
    as a feature within the network."""
import torch
import torch.nn as nn


class CoordinateConv(nn.Module):
    """Convolutional Layer with added coordinates as input"""

    def __init__(self, inputs: int, outputs: int, kernel_size, ** kwargs) -> None:
        super(CoordinateConv, self).__init__()

        self.implementcoords = ImplementCoords()
        self.conv = nn.Conv2d(inputs + 2, outputs, kernel_size, **kwargs)

    def forward(self, x):
        x = self.implementcoords(x)
        x = self.conv(x)
        return x


class ImplementCoords(nn.Module):
    """Layer to implement the coordinates as input"""

    def __init__(self) -> None:
        super(ImplementCoords, self).__init__()

    def forward(self, in_tensor: torch.Tensor):
        """
            Args:
                in_tensor: (Batch, inputs, lon_size, lat_size) - 
                    lon being x-cord, lat being y-cord
        """
        batch_size, _, lon_size, lat_size = in_tensor.size()

        # Create the x and y coordinates
        x_coords = torch.arange(0, lon_size).repeat(batch_size, 1, lat_size, 1)
        y_coords = torch.arange(0, lat_size).repeat(
            batch_size, 1, lon_size, 1).transpose(2, 3)

        # Concatenate the coordinates to the input
        in_tensor = torch.cat([in_tensor, x_coords.type_as(in_tensor),
                               y_coords.type_as(in_tensor)], dim=1)
        return in_tensor
