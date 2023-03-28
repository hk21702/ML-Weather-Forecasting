from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the model

    Attributes:
        channels (int): Number of channels in the input
        target_dim (int): Squared dim of the target area
        context_dim (int): Squared dim of the context area
        target_delta (int): Number of time steps to predict into the future
        context_delta (int): Number of time steps to look back in the context
        feature_variables (list[str]): List of features to use (short variable names)
        target_variable (str): Target variable to predict (short variable name)
    """
    channels: int
    target_dim: int
    context_dim: int
    target_delta: int
    context_steps: int
    feature_variables: list[str]
    target_variable: str
    model_type: str = 'conv_lstm'

    def __post_init__(self):
        self.context_delta = self.context_steps * self.target_delta

        # Check values
        assert self.model_type in ['conv_lstm', 'lstm', 'gru'], \
            f'Unknown model type {self.model_type}'

        if self.model_type in ['lstm', 'gru']:
            assert self.context_dim == 1, \
                f'Context dim must be 1 for {self.model_type} model'
