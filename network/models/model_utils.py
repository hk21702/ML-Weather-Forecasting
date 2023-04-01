import torch


def get_activation_func(activation_fn: str) -> torch.nn.Module:
    """Get activation function from string

    Args:
        activation_fn (str): Activation function name

    Returns:
        torch.nn.Module: Activation function
    """
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh":
        return torch.nn.Tanh()
    elif activation_fn == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation_fn == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation_fn == "selu":
        return torch.nn.SELU()
    else:
        raise ValueError("Activation function not supported")
