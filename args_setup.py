import argparse


def setup_train_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train a model on the data using given parameters')

    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of epochs to train the model for')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size to use when training the model')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate to use when training the model')

    choices = ['conv_lstm', 'lstm', 'gru']

    parser.add_argument(
        '--model_type',
        type=str,
        choices=choices,
        default='conv_lstm',
        help=f'Model type to use. Note that non,\
            Convoluted LSTM models can only handle single point outputs. \
                Default: conv_lstm. Choices: {choices}')

    # Max number of workers
    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='Max number of workers to use when loading the data. \
            Note that memory requirements is a factor of this number. Default: 4, min 0')

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
        default=6,
        help='Context time steps (hours) used for input')

    # Prediction delta
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

    return parser
