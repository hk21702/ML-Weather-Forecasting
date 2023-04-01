"""
Convolutional LSTM layer in order to handle
the image like time series data.

Heavy influcence from https://github.com/ndrplz/ConvLSTM_pytorch (MIT License)
"""
from typing import Optional
import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell
    """

    def __init__(self, input_dims: int, hidden_dims: int,
                 k_size: tuple[int, int], bias: bool) -> None:
        """
        Init conv LSTM cell.

        Args:
            input_dims (int): Number of channels of the input
            hidden_dims (int): Number of channels of the hidden state
            k_size (tuple[int, int]): Kernel size of the convolution
            bias (bool): Whether to use bias in the convolution
        """

        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.k_size = k_size
        self.bias = bias

        self.padding = (k_size // 2, k_size // 2)

        self.conv = nn.Conv2d(
            in_channels=self.input_dims + self.hidden_dims,
            out_channels=4 * self.hidden_dims,
            kernel_size=self.k_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x: torch.Tensor, current_state: tuple()) -> torch.Tensor:
        h_cur, c_cur = current_state
        cat_x = torch.cat([x, h_cur], dim=1)

        cat_conv = self.conv(cat_x)
        i, f, o, g = torch.split(cat_conv, self.hidden_dims, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, x: torch.Tensor) -> tuple():
        """
        Inits the hidden state of the LSTM.

        Args:
            x (torch.Tensor): Input tensor (batch_size, time, input_chans, height, width)
        """
        batch_size, _, _, height, width = x.size()
        return (torch.zeros(batch_size, self.hidden_dims, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dims, height, width, device=self.conv.weight.device))

    def reset_parameters(self) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain("selu"))
        self.conv.bias.data.zero_()


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: int, num_layers: int,
                 bias: bool = True,
                 return_all: bool = False) -> None:
        super().__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = True
        self.bias = bias
        self.return_all = return_all

        # Create cell list
        cell_list = []
        for i in range(self.num_layers):
            c_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dims=c_input_dim,
                                          hidden_dims=self.hidden_dim[i],
                                          k_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        for cell in self.cell_list:
            cell.reset_parameters()

    def forward(self, input_tensor: torch.Tensor,
                hidden_states: Optional[list] = None) -> tuple():
        c_layer_input = torch.unbind(input_tensor, dim=1)

        if not hidden_states:
            hidden_states = self._init_hidden(input_tensor)

        last_state_list = []

        seq_len = len(c_layer_input)

        for i in range(self.num_layers):
            h, c = hidden_states[i]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[i](c_layer_input[t], [h, c])
                output_inner.append(h)

            c_layer_input = output_inner
            last_state_list.append([h, c])

        layer_output = torch.stack(output_inner, dim=1)

        return layer_output, last_state_list

    def _init_hidden(self, input_tensor: torch.Tensor) -> list:
        return [
            self.cell_list[i].init_hidden(input_tensor)
            for i in range(self.num_layers)
        ]

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int) -> tuple[int, int]:
        """
        Extends the given param to a list with length num_layers.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
