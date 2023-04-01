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
                 kernel_size: Union[int, tuple[int, int]] = (3, 3)):

        self.context_size: int = ds.context_steps
        self.horizon: int = ds.horizon
        self.window_size: int = ds.window_size
        self.targets = ds.targets

        self.model_type = model_type
        self.down_sample_channels = down_sample_channels
        self.lstm_chans = lstm_chans
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
        self.context_width = width

        assert self.input_chans == channels, 'Expected input channels to be ' \
            f'{self.input_chans}, instead was {channels}'

        assert width == height, 'Expected square input, instead was ' \
            f'{width}x{height}'
        
        #Print feature shape
        print(f'Feature shape: {sample_features.shape}')

    def _label_shape(self, sample_labels):
        time, channels, width, height = sample_labels.shape

        self.output_chans = channels
        self.target_width = width
        self.horizon = time

        assert self.output_chans == channels, 'Expected output channels to be ' \
            f'{self.output_chans}, instead was {channels}'

        assert width == height, 'Expected label square input, instead was ' \
            f'{width}x{height}'
        
        #Print label shape, center of labels
        print(f'Label shape: {sample_labels.shape}')
        print(f'Label center: {sample_labels[:, :, width//2, height//2]}')

    def __post_init__(self):
        # Check values
        assert self.model_type in ['conv_lstm', 'lstm', 'gru'], \
            f'Unknown model type {self.model_type}'

        # Check ensure target width is 4x smaller than context width
        assert self.context_width // self.target_width == 4, \
            f'Expected target width to be 1/4 of context width, instead was ' \
            f'{self.target_width} and {self.context_width}'
