from typing import Union

from network.window_iter_ds import WindowIterDS


class ModelConfig:
    """Configuration for the model

    Attributes:

    """

    def __init__(self, ds: WindowIterDS,
                 model_type: str = 'conv_lstm',
                 down_sample_channels: int = 256,
                 kernel_size: Union[int, tuple[int, int]] = (3, 3)):
        assert len(ds) > 0, 'Dataset must not be empty'

        self.context_size: int = ds.context_steps
        self.horizon: int = ds.horizon
        self.window_size: int = ds.window_size
        self.targets = ds.targets

        self.model_type = model_type
        self.down_sample_channels = down_sample_channels
        self.kernel_size = kernel_size

        sample_features, sample_labels = ds[0]

        # Check features
        self._feature_shape(sample_features)

        # Check labels
        self._label_shape(sample_labels)

    def _feature_shape(self, sample_features):
        time, channels, width, height = sample_features.shape

        self.input_chans = channels
        self.context_steps = time
        self.hidden_layers = time
        self.feature_apothem = int(width - 1 // 2)

        assert self.input_chans == channels, 'Expected input channels to be ' \
            f'{self.input_chans}, instead was {channels}'

        assert width == height, 'Expected square input, instead was ' \
            f'{width}x{height}'

    def _label_shape(self, sample_labels):
        time, channels, width, height = sample_labels.shape

        self.output_chans = channels
        self.target_apothem = int(width - 1 // 2)
        self.horizon = time

        assert self.output_chans == channels, 'Expected output channels to be ' \
            f'{self.output_chans}, instead was {channels}'

        assert width == height, 'Expected label square input, instead was ' \
            f'{width}x{height}'

    def __post_init__(self):
        # Check values
        assert self.model_type in ['conv_lstm', 'lstm', 'gru'], \
            f'Unknown model type {self.model_type}'
