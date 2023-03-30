"""
Data pipeline to convert from Xarray dataset to a PyTorch Forecasting
time series dataset.
"""

import xarray as xr
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet


def get_ts_dataset(ds: xr.Dataset,
                   target_features: list[str],
                   context_steps: int,
                   target_steps: int,
                   target_lon: float,
                   target_lat: float,
                   context_apothem: int,
                   ) -> TimeSeriesDataSet:
    """
    Args:
        ds: (time, longitude, latitude) - Dataset to convert
        target_features: List of features to predict
        context_steps: Number of time steps to use as context
        target_steps: Number of time steps to predict
        target_lon: Longitude to predict
        target_lat: Latitude to predict
        context_apothem: Apothem of the context square
    Returns:
        TimeSeriesDataSet
    """

    # Crop the dataset to the context square
    ds = ds.sel(longitude=slice(target_lon - context_apothem,
                                target_lon + context_apothem),
                latitude=slice(target_lat - context_apothem,
                               target_lat + context_apothem))

    # Convert to Dataframe
    df = ds.to_dataframe()

    # Create a time series dataset
    ts_dataset = TimeSeriesDataSet(
        df,
        time_idx="time",
        target=target_features,
        group_ids=["longitude", "latitude"],
        min_encoder_length=context_steps,
        max_encoder_length=context_steps,
        min_prediction_length=target_steps,
        max_prediction_length=target_steps,
        static_categoricals=["longitude", "latitude"],
        add_relative_time_idx=True)

    return ts_dataset
