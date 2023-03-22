"""Train a model on the data using given parameters"""
import argparse


def get_args() -> argparse.Namespace:
    """Returns the command line arguments"""

    parser = argparse.ArgumentParser(
        description='Train a model on the data using given parameters')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
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
        help='Learning rate to use when training the model')

    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='Model to use when training the model')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing the data to train the model on')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Directory to save the trained model to')

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

    # Target cubed size
    parser.add_argument(
        '--target_cubed_size',
        type=int,
        default=3,
        help='Target cubed size to predict')

    # Context cubed size
    parser.add_argument(
        '--context_cubed_size',
        type=int,
        default=3,
        help='Context cubed size for input')

    # Context downsampling factor
    parser.add_argument(
        '--context_downsampling_factor',
        type=int,
        default=2,
        help='Context downsampling factor for input')

    # Context time size
    parser.add_argument(
        '--context_time_size',
        type=int,
        default=3,
        help='Context time size (hours) for input')

    # Prediction delta
    parser.add_argument(
        '--prediction_delta',
        type=int,
        default=3,
        help='Prediction delta (hours). How many hours to predict into the future')

    return parser.parse_args()
