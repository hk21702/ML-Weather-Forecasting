from typing import Union

from network.window_iter_ds import WindowIterDS


class ModelConfig:
    """Configuration for the model

    Attributes:

    """

    def __init__(self, ds: WindowIterDS,
                 model_type: str = 'conv_lstm',
                 down_sample_channels: int = 256,
                 lstm_chans: int = 256,
                 kernel_size: Union[int, tuple[int, int]] = (3, 3),
                 wb_config=None):

        self.context_size: int = ds.context_steps
        self.horizon: int = ds.horizon
        self.window_size: int = ds.window_size
        self.targets = ds.targets

        self.model_type = model_type
        self.down_sample_channels = down_sample_channels
        self.lstm_chans = lstm_chans
        self.kernel_size = kernel_size

        if wb_config is not None:
            wb_config.down_sample_channels = down_sample_channels
            wb_config.lstm_chans = lstm_chans
            wb_config.kernel_size = kernel_size

        sample_features, sample_labels = ds[0]

        # Check features
        self._feature_shape(sample_features, wb_config=wb_config)

        # Check labels
        self._label_shape(sample_labels, wb_config=wb_config)

    def _feature_shape(self, sample_features, wb_config=None):
        time, channels, width, height = sample_features.shape

        self.input_chans = channels
        self.context_steps = time
        self.hidden_layers = time
        self.context_width = width

        if wb_config is not None:
            wb_config.input_chans = channels
            wb_config.context_steps = time
            wb_config.hidden_layers = time
            wb_config.context_width = width

        assert self.input_chans == channels, 'Expected input channels to be ' \
            f'{self.input_chans}, instead was {channels}'

        assert width == height, 'Expected square input, instead was ' \
            f'{width}x{height}'

        # Print feature shape
        print(f'Feature shape: {sample_features.shape}')

    def _label_shape(self, sample_labels, wb_config=None):
        time, channels, width, height = sample_labels.shape

        # Print label shape, center of labels
        print(f'Label shape: {sample_labels.shape}')

        self.output_chans = channels
        self.target_width = width
        self.horizon = time

        if wb_config is not None:
            wb_config.output_chans = channels
            wb_config.target_width = width

        assert self.output_chans == channels, 'Expected output channels to be ' \
            f'{self.output_chans}, instead was {channels}'

        assert width == height, 'Expected label square input, instead was ' \
            f'{width}x{height}'

    def __post_init__(self):
        # Check values
        assert self.model_type in ['conv_lstm', 'lstm', 'gru'], \
            f'Unknown model type {self.model_type}'

        # Check ensure target width is 4x smaller than context width
        assert self.context_width // self.target_width == 4, \
            f'Expected target width to be 1/4 of context width, instead was ' \
            f'{self.target_width} and {self.context_width}'
