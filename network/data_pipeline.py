import xarray as xr
import numpy as np

from .model import ModelConfig


def get_inputs(dataset: xr.Dataset, t_lat: float, t_long: float, t_time, config: ModelConfig) -> np.ndarray:
    """Gets the inputs for the model

    Get
    Args:
        dataset (xr.Dataset): The dataset to get the inputs from
        config (ModelConfig): The model configuration
    Returns:
        np.ndarray: The inputs for the model
    """
    pass


def get_labels(dataset: xr.Dataset, t_lat: float, t_long: float, t_time, config: ModelConfig) -> np.ndarray:
    """Gets the labels for the model

    Args:
        dataset (xr.Dataset): The dataset to get the labels from
        config (ModelConfig): The model configuration
    Returns:
        np.ndarray: The labels for the model
    """
    pass
