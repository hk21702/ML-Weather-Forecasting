"""
Data pipeline to convert from Xarray dataset to a PyTorch Forecasting
time series dataset.
"""

import xarray as xr
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet


def get_ts_dataset(ds: xr.Dataset,
                   target_features: list[str],
                   context_steps: int,
                   target_steps: int,
                   target_lon: float,
                   target_lat: float,
                   context_apothem: float,
                   debug: bool = True) -> TimeSeriesDataSet:
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
    # Calc the corners of the context square
    context_lon_min = target_lon - context_apothem
    context_lon_max = target_lon + context_apothem
    context_lat_min = target_lat - context_apothem
    context_lat_max = target_lat + context_apothem

    if debug:
        print(f"Target lon: {target_lon}, lat: {target_lat}")
        print(
            f"Context lon: {context_lon_min} to {context_lon_max}, lat: {context_lat_min} to {context_lat_max}")

    # Check that the context square is within the dataset
    assert context_lon_min >= ds.longitude.min(
    ), "Context square is outside of dataset, check longitude"
    assert context_lon_max <= ds.longitude.max(
    ), "Context square is outside of dataset, check longitude"
    assert context_lat_min >= ds.latitude.min(
    ), "Context square is outside of dataset, check latitude"
    assert context_lat_max <= ds.latitude.max(
    ), "Context square is outside of dataset, check latitude"

    # Filter the dataset to the context square
    ds = ds.sel(longitude=slice(context_lon_min, context_lon_max))

    ds = ds.sel(latitude=slice(context_lat_max, context_lat_min))

    # Convert time coordinate to integer representation on hour scale
    ds['time'] = ds.time.astype('int64') // 1e9 // 3600
    # Convert to int
    ds['time'] = ds.time.astype('int32')

    # Convert to pandas
    df = ds.to_dataframe().reset_index()

    # Sort index by time
    df = df.sort_values(by=["time"])

    if debug:
        print(df.head())

    none_normalizer = MultiNormalizer(
        [TorchNormalizer('identity'), TorchNormalizer('identity')])

    # Get all features other than longitude and latitude, time, hour, day, month, year
    # These are features we will not know in the future.

    time_varying_unknown_reals = [col for col in df.columns if col not in [
        'longitude', 'latitude', 'time', 'hour', 'day', 'month', 'year']]

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
        add_relative_time_idx=True,
        target_normalizer=none_normalizer,
        categorical_encoders={'longitude': NaNLabelEncoder(add_nan=True),
                              'latitude': NaNLabelEncoder(add_nan=True)},
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        allow_missing_timesteps=False)

    if debug:
        print("Created TimeSeriesDataSet object...")
        print(f"\tTarget features: {ts_dataset.target_names}")

    return ts_dataset
