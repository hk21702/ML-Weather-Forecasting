"""
Convolutional LSTM layer in order to handle
the image like time series data.

Heavy influcence from https://github.com/ndrplz/ConvLSTM_pytorch (MIT License)
"""
from typing import Union

import torch
from torch import jit, nn

from network.models.model_utils import get_activation_func


class ConvLSTMCell(jit.ScriptModule):
    """
    Convolutional LSTM cell
    """

    def __init__(self, input_dims: int, hidden_dims: int,
                 k_size: tuple[int, int], bias: bool = True,
                 activation_fn: nn.Module = torch.tanh,
                 device: torch.device = torch.device('cpu')) -> None:
        """
        Init conv LSTM cell.

        Args:
            input_dims (int): Number of channels of the input
            hidden_dims (int): Number of channels of the hidden state
            k_size (tuple[int, int]): Kernel size of the convolution
            bias (bool): Whether to use bias in the convolution
            activation_fn (torch.nn.Module): Activation function to use
        """

        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.k_size = k_size
        self.bias = bias
        self.activation_fn = activation_fn

        self.padding = (k_size // 2, k_size // 2)

        self.conv = nn.Conv2d(
            in_channels=self.input_dims + self.hidden_dims,
            out_channels=4 * self.hidden_dims,
            kernel_size=self.k_size,
            padding=self.padding,
            bias=self.bias,
            device=device
        )

    @jit.script_method
    def forward(self, x: torch.Tensor, current_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = current_state
        cat_x = torch.cat([x, h_cur], dim=1)

        cat_conv = self.conv(cat_x)
        i, f, o, g = torch.split(cat_conv, self.hidden_dims, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = self.activation_fn(g)

        c_next = f * c_cur + i * g
        h_next = o * self.activation_fn(c_next)

        return h_next, c_next

    def init_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inits the hidden state of the LSTM.

        Args:
            x (torch.Tensor): Input tensor (batch_size, time, input_chans, height, width)
        """
        batch_size, _, _, height, width = x.size()
        return (torch.zeros(batch_size, self.hidden_dims, height, width, device=x.device),
                torch.zeros(batch_size, self.hidden_dims, height, width, device=x.device))

    def reset_parameters(self, activation_type: str) -> None:
        """Resets parameters"""
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(activation_type))
        self.conv.bias.data.zero_()


class ConvLSTM(jit.ScriptModule):
    """
    Convolutional LSTM
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: int, num_layers: int,
                 bias: bool = True,
                 activation_type: str = 'tanh',
                 device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        activation_fn = get_activation_func(activation_type)
        # Create multiple instances of activation functions as some
        # may be trainable!!!!
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation_fn = self._extend_for_multilayer(activation_fn, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = True
        self.bias = bias

        # Create cell list
        cell_list = []
        for i in range(self.num_layers):
            c_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dims=c_input_dim,
                                          hidden_dims=self.hidden_dim[i],
                                          k_size=self.kernel_size[i],
                                          bias=self.bias,
                                          activation_fn=activation_fn[i],
                                          device=device))

        self.cell_list = nn.ModuleList(cell_list)

        for cell in self.cell_list:
            cell.reset_parameters(activation_type)

    @jit.script_method
    def forward(self, input_tensor: torch.Tensor,
                hidden_states: Union[list[tuple[torch.Tensor, torch.Tensor]], None] = None) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        c_layer_input = torch.unbind(input_tensor, dim=1)

        if hidden_states is None:
            hidden_states = self._init_hidden(input_tensor)

        last_state_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, cell in enumerate(self.cell_list):
            h, c = hidden_states[i]
            output_inner = []

            for c_input in c_layer_input:
                h, c = cell(c_input, (h, c))
                output_inner.append(h)

            c_layer_input = output_inner
            last_state_list.append((h, c))

        layer_output = torch.stack(output_inner, dim=1)

        return layer_output, last_state_list

    def _init_hidden(self, input_tensor: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [cell.init_hidden(input_tensor) for cell in self.cell_list]

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int) -> list:
        """
        Extends the given param to a list with length num_layers.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
