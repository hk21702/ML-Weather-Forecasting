"""Train a model on the data using given parameters"""
import argparse

import pandas as pd
import torch
import torch.nn as nn
import xarray as xr


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


def load_data(train_path: str, val_path: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Loads the data from the given paths

    Args:
        train_path (str): Path to the netCDF4 file to train the model on
        val_path (str): Path to the netCDF4 file to validate the model on

    Returns:
        tuple[xr.Dataset, xr.Dataset]: The training and validation data
    """
    # Load the training data
    train_data = xr.open_dataset(train_path)

    # Load the validation data
    val_data = xr.open_dataset(val_path)

    return train_data, val_data


def create_rolling(dataset: xr.Dataset,
                   context_steps: int,
                   target_delta: int,
                   context_apothem: int) -> xr.core.rolling.DatasetRolling:
    """Creates a rolling window on time and space dimensions.

        Args:
            context_steps (int): Number of time steps to look back in the context
            target_delta (int): Number of time steps to predict into the future
            context_apothem (int): Apothem of the context area

        Returns:
            xr.DataArray: The rolling time
    """
    window_size = context_steps + target_delta
    width = context_apothem * 2 + 1

    return dataset.rolling(dim={'time', 'longitude', 'latitude'},
                           center=True,
                           time=window_size,
                           longitude=width,
                           latitude=width)
