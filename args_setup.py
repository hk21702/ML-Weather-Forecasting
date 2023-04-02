import argparse


def setup_train_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train a model on the data using given parameters')

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Max number of epochs to train the model for')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size to use when training the model. Try to keep it as a power of '
        '2 for better results. Default: 32')

    choices = ['conv_lstm', 'conv_gru']

    parser.add_argument(
        '--model_type',
        type=str,
        choices=choices,
        default='conv_gru',
        help=f'Model type to use. Note that non conv3d is single shot\
                Default: conv_lstm. Choices: {choices}')

    # Activation functions
    choices = ['relu', 'selu', 'tanh']

    parser.add_argument(
        '--activation_fn',
        type=str,
        choices=choices,
        default='selu',
        help=f'Activation function to use. Default: selu. Choices: {choices}')
    
    # Dropout variants
    choices = ['dropout', 'alpha_dropout']

    parser.add_argument(
        '--dropout_type',
        type=str,
        choices=choices,
        default='dropout',
        help=f'Dropout type to use. Default: dropout. Choices: {choices}')
    
    # Dropout rate
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='Dropout rate to use. Default: 0.5')
    
    # Post conv dropout rate
    parser.add_argument(
        '--post_conv_dropout_rate',
        type=float,
        default=0,
        help='Dropout rate to use after the conv layers. Set to 0 to disable. Suggested 0.2 Default: 0')
    

    parser.add_argument(
        '--train_path',
        type=str,
        default='cache/train.nc',
        help='Path to the netCDF4 file to train the model on')

    parser.add_argument(
        '--val_path',
        type=str,
        default='cache/val.nc',
        help='Path to the netCDF4 file to validate the model on')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to save the trained model to')

    parser.add_argument(
        '--model_name',
        type=str,
        default='model',
        help='Name of the model to save'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Directory to save the training logs to')

    # Target longitude and latitude
    parser.add_argument(
        '--target_lon',
        type=float,
        default=-80,
        help='Target longitude to predict')

    parser.add_argument(
        '--target_lat',
        type=float,
        default=43,
        help='Target latitude to predict')

    # Context cubed size
    parser.add_argument(
        '--context_apothem',
        type=int,
        default=3,
        help='Context apothem for input')

    # Context time size
    parser.add_argument(
        '--context_steps',
        type=int,
        default=10,
        help='Context time steps (hours) used for input')

    # Horizon
    parser.add_argument(
        '--horizon',
        type=int,
        default=3,
        help='How many time steps (hours) to predict into the future')

    # Debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode. Prints extra information')

    # Target features
    parser.add_argument('-tf', '--target_feats',
                        nargs='+', default=['t2m', 'tp'])

    # Initial learning rate
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate for the model, Default 1e-4')

    # Learning rate patience
    parser.add_argument(
        '--lr_patience',
        type=int,
        default=2,
        help='Number of epochs to wait before reducing learning rate on plateu, Default 2')

    # Early stop tolerance
    parser.add_argument(
        '--early_stop_tolerance',
        type=int,
        default=5,
        help='Number of epochs to wait before early stopping, Default 5')
    
    # Compile model?
    parser.add_argument(
        '--compile',
        action='store_true',
        default=False,
        help='Compile the model using Pytorch 2.0 before training. Only supported on Linux!')

    return parser
