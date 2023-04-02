"""
Convolutional GRU layer
"""

import torch
from torch import nn
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell
    """

    def __init__(self, input_dims: int, hidden_dims: int,
                 k_size: tuple[int, int], bias: bool = True,
                 activation_fn: nn.Module = torch.tanh,
                 device: torch.device = torch.device('cpu')) -> None:
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.k_size = k_size
        self.bias = bias
        self.activation_fn = activation_fn

        self.padding = (self.k_size[0] // 2, self.k_size[1] // 2)

        self.conv_gates = nn.Conv2d(
            in_channels=self.input_dims + self.hidden_dims,
            out_channels=2 * self.hidden_dims,
            kernel_size=self.k_size,
            padding=self.padding,
            bias=self.bias,
            device=device
        )

        self.conv_update = nn.Conv2d(
            in_channels=self.input_dims + self.hidden_dims,
            out_channels=self.hidden_dims,
            kernel_size=self.k_size,
            padding=self.padding,
            bias=self.bias,
            device=device
        )

    def init_hidden(self, input: torch.Tensor):
        batch_size, _, h, w = input.shape
        return Variable(torch.zeros(batch_size, self.hidden_dims, h, w, device=input.device))

    def reset_parameters(self) -> None:
        """
        Re-initializes the parameters of the cell.
        """
        # Gates
        nn.init.xavier_uniform_(self.conv_gates.weight,
                                gain=nn.init.calculate_gain(self.activation_fn))
        self.conv_gates.bias.data.zero_()
        self.conv_gates.retain_grad()

        # Update
        nn.init.xavier_uniform_(self.conv_update.weight,
                                gain=nn.init.calculate_gain(self.activation_fn))
        self.conv_update.bias.data.zero_()
        self.conv_update.retain_grad()

    def forward(self, input: torch.Tensor, h_prev):
        cat_x = torch.cat([input, h_prev], dim=1)

        cat_conv = self.conv_gates(cat_x)

        r, z = torch.split(cat_conv, self.hidden_dims, dim=1)
        reset_gate = torch.sigmoid(r)
        update_gate = torch.sigmoid(z)

        cat_x = torch.cat([input, reset_gate * h_prev], dim=1)

        next_hidden_state = self.conv_update(cat_x)
        next_hidden_state = self.activation_fn(next_hidden_state)

        next_hidden_state = (1 - update_gate) * h_prev + \
            update_gate * next_hidden_state

        return next_hidden_state

class ConvGRU(nn.Module):
    """
    Convolutional GRU layer
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True,
                 activation_fn=torch.tanh,
                 device: torch.device = torch.device('cpu')) -> None:
        super().__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        activation_fn = self._extend_for_multilayer(activation_fn, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(self.num_layers):
            c_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(c_input_dim,
                                         self.hidden_dim[i],
                                         kernel_size[i],
                                         self.bias,
                                         activation_fn[i],
                                         device=device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state=None):
        c_layer_input = torch.unbind(input, dim=1)

        if hidden_state is None:
            hidden_state = self._init_hidden(c_layer_input[0])

        seq_len = len(c_layer_input)

        layer_output_list = []
        layer_state_list = []

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](c_layer_input[t], h)
                output_inner.append(h)

            c_layer_input = torch.stack(output_inner)

            layer_state_list.append(h)

        layer_output_list = torch.stack(output_inner, dim=1)
        layer_state_list = torch.stack(layer_state_list, dim=0)
        return layer_output_list, layer_state_list

    def _init_hidden(self, input):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(input))
        return init_states

    @ staticmethod
    def _extend_for_multilayer(param, num_layers: int) -> list:
        """
        Extends the given param to a list with length num_layers.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
