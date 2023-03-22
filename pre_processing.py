import numpy as np
import pandas as pd


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
