from dataclasses import dataclass
from typing import Union


@dataclass
class ModelConfig:
    """Configuration for the model

    Attributes:
        feature_variables (list[str]): The variables to use as features
        target_variables (list[str]): The variables to use as targets
        context_apothem (int): The apothem of the context cube
        context_steps (int): The number of time steps to look back in the context
        target_apothem (int): The apothem of the target cube
        target_steps (int): The number of time steps to predict into the future
        input_size (int): The size of the input to the model
        model_type (str): The type of model to use
        down_sample_channels (int): The number of channels to downsample to
        hidden_dims (int): The number of hidden dimensions
        kernel_size (Union(int, tuple[int, int])): The kernel size to use
    """
    feature_variables: list[str]
    target_variables: list[str]

    context_apothem: int = 12
    context_steps: int = 6

    target_apothem: int = 6
    target_steps: int = 3

    input_size: int = 128

    model_type: str = 'conv_lstm'
    down_sample_channels: int = 256
    hidden_dims: int = 256

    input_chans: int = 20

    kernel_size: Union[int, tuple[int, int]] = (3, 3)

    def __post_init__(self):
        self.total_steps = self.context_steps + self.target_steps

        self.output_chans = len(self.target_variables)

        self.target_dim = self.target_apothem * 2 + 1
        self.context_dim = self.context_apothem * 2 + 1

        # Check values
        assert self.model_type in ['conv_lstm', 'lstm', 'gru'], \
            f'Unknown model type {self.model_type}'

        assert self.input_chans < self.input_size, \
            f'Input size must be greater than the number of input channels'

        if self.model_type in ['lstm', 'gru']:
            assert self.context_dim == 1, \
                f'Context dim must be 1 for {self.model_type} model'
