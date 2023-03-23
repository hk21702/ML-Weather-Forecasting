import numpy as np
import pandas as pd


def downsample_data(data: pd.DataFrame, degree: float):
    """Downsamples the location data to a given degree, grouped by time.

    Args:
        data: The data to downsample.
        degree: The degree to downsample to, rounded to .25, must be <= than 0.25
    """

    # Round to a multiple of .25
    degree = round(degree * 4) / 4

    assert degree >= 0.25, 'Degree must be <= 0.25'

    # Get the min and max longitude and latitude


def normalize_feature_data(data: pd.DataFrame, features: list[str], target: list[str]) -> pd.DataFrame:
    """Normalizes the feature data. Target data is normalized but also retained."""
    # Make a copy of target columns to retain them
    target_data = data[target].copy()

    # Normalize the feature data
    data[features] = (data[features] - data[features].mean()
                      ) / data[features].std()

    # Create new target names for columns
    target_names = [f'{col}_target' for col in target]

    # Rename target columns
    target_data.columns = target_names

    # Concatenate the target data to the feature data
    data = pd.concat([data, target_data], axis=1)
