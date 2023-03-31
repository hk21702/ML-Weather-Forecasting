"""
Preprocessing of raw data from the ERA5 dataset to a useable state.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from tabulate import tabulate


def get_args() -> argparse.Namespace:
    """Parses the command line arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess ERA5 data')

    # NetCRf folder path
    parser.add_argument('--netcdf_folder', type=str, default='data',
                        help='Path to the folder containing the NetCDF files')

    # Cache folder path
    parser.add_argument('--cache_folder', type=str, default='cache',
                        help='Path to the folder to cache the data to')

    # Validation ratio
    parser.add_argument('--val_ratio', type=float, default=.10,
                        help='Ratio of the validation set')

    # Test ratio
    parser.add_argument('--test_ratio', type=float, default=.10,
                        help='Ratio of the test set')

    # Validation file name
    parser.add_argument('--val_file', type=str, default='val.nc',
                        help='Name of the validation file')

    # Test file name
    parser.add_argument('--test_file', type=str, default='test.nc',
                        help='Name of the test file')

    # Training file name
    parser.add_argument('--train_file', type=str, default='train.nc',
                        help='Name of the training file')

    # Target longitude
    parser.add_argument('--target_lon', type=float, default=-80,
                        help='Target longitude')

    # Target latitude
    parser.add_argument('--target_lat', type=float, default=43,
                        help='Target latitude')

    # Context apothem
    parser.add_argument('--context_apothem', type=float, default=3,
                        help='Context apothem')

    return parser.parse_args()


def check_args(args: argparse.Namespace):
    """Checks the command line arguments"""

    # Apothem must be positive
    assert args.context_apothem > 0, 'Context apothem must be positive'

    # Target longitude must be between -180 and 180
    assert -180 <= args.target_lon <= 180, 'Target longitude must be between -180 and 180'

    # Target latitude must be between -90 and 90
    assert -90 <= args.target_lat <= 90, 'Target latitude must be between -90 and 90'

    # Ratios must be positive
    assert args.val_ratio >= 0, 'Validation ratio must be positive'
    assert args.test_ratio >= 0, 'Test ratio must be positive'


def get_data(netcdf_folder: str) -> xr.Dataset:
    """Loads the NetCDF data

    Args:
        netcdf_folder (str): Path to the folder containing the NetCDF files

    Returns:
        xr.Dataset: The NetCDF data
    """
    # Get the list of NetCDF files, only .nc files
    netcdf_files = [os.path.join(netcdf_folder, file)
                    for file in os.listdir(netcdf_folder) if file.endswith('.nc')]

    assert netcdf_files, 'No NetCDF files found'

    # Load the data
    return xr.open_mfdataset(netcdf_files, combine='by_coords')


def cache_data(dataset: xr.Dataset, cache_folder: str,
               file_name: str, clean: bool = True) -> None:
    """Caches the data to a nc file"""

    # Create the cache folder if it doesn't exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    # If filename does not have .nc extension, add it
    if not file_name.endswith('.nc'):
        file_name += '.nc'

    # Clear the cached data if it exists
    if clean & os.path.exists(os.path.join(cache_folder, file_name)):
        os.remove(os.path.join(cache_folder, file_name))

    print(f'Caching data to {cache_folder} as {file_name}...')

    # Cache the data
    dataset.to_netcdf(os.path.join(cache_folder, file_name))


def split_data(dataset: xr.Dataset,
               val_ratio: float = .10,
               test_ratio: float = .10) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Splits the data into a training, validation and test sets on the time dimension.
    The splits will have coninuous time ranges.

    Test set will be the last time range.

    Args:
        dataset (xr.Dataset): The dataset to split
        val_ratio (float, optional): The ratio of the validation set. Defaults to .10.
        test_ratio (float, optional): The ratio of the test set. Defaults to .10.

    Returns:
        tuple[xr.Dataset, xr.Dataset]: The training and validation set
    """
    # Check for valid ratios
    assert val_ratio + test_ratio < 1, 'Validation and test ratio must be less than 1'

    # Get the time range
    time_range = dataset.time.max().values - dataset.time.min().values

    # Get the time range of the various sets
    val_time_range = time_range * val_ratio
    test_time_range = time_range * test_ratio
    train_time_range = time_range - val_time_range - test_time_range

    # Split the data into their own datasets
    train_set = dataset.sel(time=slice(
        dataset.time.min().values, dataset.time.min().values + train_time_range))
    val_set = dataset.sel(time=slice(
        dataset.time.min().values + train_time_range, dataset.time.min().values + train_time_range + val_time_range))
    test_set = dataset.sel(time=slice(
        dataset.time.min().values + train_time_range + val_time_range, dataset.time.max().values))

    return train_set, val_set, test_set


def add_rh2m(dataset: xr.Dataset) -> xr.Dataset:
    """Add the relative humidity at 2 meters to the dataset.

    Args:
        dataset (xr.Dataset): The dataset to add the humidity to

    Returns:
        xr.Dataset: The dataset with the relative humidity at 2 meters
    """

    # Convert both t2m and d2m to Celsius from Kelvin
    t2m = dataset.t2m - 273.15
    d2m = dataset.d2m - 273.15

    # Calculate the saturation vapor pressure
    es = 6.112 * xr.apply_ufunc(lambda x: np.exp((17.67 * x) / (x + 243.5)),
                                t2m, dask='allowed', output_dtypes=[float],
                                input_core_dims=[[]])

    # Calculate the actual vapor pressure
    e = 6.112 * xr.apply_ufunc(lambda x: np.exp((17.67 * x) / (x + 243.5)),
                               d2m, dask='allowed', output_dtypes=[float],
                               input_core_dims=[[]])

    # Calculate the relative humidity
    rh = (e / es) * 100

    # Add the relative humidity to the dataset
    dataset['rh2m'] = rh

    # Add the long name and units
    dataset.rh2m.attrs['long_name'] = '2 metre relative humidity*'
    dataset.rh2m.attrs['units'] = '(0 - 1)'

    return dataset


def manipulate_time(dataset: xr.Dataset,
                    hour: bool = True,
                    day: bool = True,
                    month: bool = True,
                    year: bool = True) -> xr.Dataset:
    """
    Add discrete variables for the hour, day, month and year into the dataset.
    Also mainpulates the time coordinate to be in integer form on an hour timestep.

        Args:
            dataset (xr.Dataset): The dataset to add the discrete variables to
            hour (bool, optional): Add the hour variable. Defaults to True.
            day (bool, optional): Add the day variable. Defaults to True.
            month (bool, optional): Add the month variable. Defaults to True.
            year (bool, optional): Add the year variable. Defaults to True.
    """
    # Add the hour variable
    if hour:
        dataset['hour'] = dataset.time.dt.hour
        dataset.hour.attrs['long_name'] = 'Hour of the day'
        dataset.hour.attrs['units'] = 'hour'

    # Add the day variable
    if day:
        dataset['day'] = dataset.time.dt.day
        dataset.day.attrs['long_name'] = 'Day of the month'
        dataset.day.attrs['units'] = 'day'

    # Add the month variable
    if month:
        dataset['month'] = dataset.time.dt.month
        dataset.month.attrs['long_name'] = 'Month of the year'
        dataset.month.attrs['units'] = 'month'

    # Add the year variable
    if year:
        dataset['year'] = dataset.time.dt.year
        dataset.year.attrs['long_name'] = 'Year'

    # Convert time coordinate to integer representation on hour scale
    dataset['time'] = dataset.time.astype('int64') // 1e9 // 3600
    # Convert to int
    dataset['time'] = dataset.time.astype('int32')

    dataset.time.attrs['long_name'] = 'Time'
    dataset.time.attrs['units'] = 'hours since 1970-01-01 00:00:00'

    return dataset


def feature_engineering(dataset: xr.Dataset) -> xr.Dataset:
    """Feature engineering on the dataset

    Args:
        dataset (xr.Dataset): The dataset to do feature engineering on

    Returns:
        xr.Dataset: The dataset with the feature engineering applied
    """
    print('Applying feature engineering...')

    # Add the relative humidity at 2 meters
    dataset = add_rh2m(dataset)
    dataset = manipulate_time(dataset)

    return dataset


def crop_area(dataset: xr.Dataset,
              target_lat: float,
              target_lon: float,
              context_apothem: float,) -> xr.Dataset:
    """ Crop the dataset to the target area."""
    print("Cropping dataset...")

    # Get min and max lat and lon
    min_lat = target_lat - context_apothem
    max_lat = target_lat + context_apothem
    min_lon = target_lon - context_apothem
    max_lon = target_lon + context_apothem

    # Crop the dataset
    dataset = dataset.sel(latitude=slice(max_lat, min_lat),
                          longitude=slice(min_lon, max_lon))

    return dataset


def print_dataset_info(dataset: xr.Dataset) -> None:
    """Prints the dataset info.

    Dimensions, time range, variable count and variable info

    Args:
        dataset (xr.Dataset): The dataset to print info for
    """
    print(f'\tDimensions: {dataset.dims}')
    print(
        f'\tTime range: {dataset.time.min().values} - {dataset.time.max().values}')
    print(f'\tVariable count: {len(dataset.variables)}')

    # Get variable name and long name. Unit when attribute exists
    variable_info = pd.DataFrame(
        [[variable, dataset[variable].attrs.get('long_name', None), dataset[variable].attrs.get('units', None)]
         for variable in dataset.variables],
        columns=['Variable', 'Long Name', 'Unit']
    )

    print(tabulate(variable_info, headers='keys', tablefmt='psql'))


def main() -> None:
    """Main function"""
    args = get_args()
    check_args(args)

    dataset = get_data(args.netcdf_folder)

    print(f'Dataset loaded from {args.netcdf_folder}')
    print_dataset_info(dataset)

    # Crop the dataset to the target area
    dataset = crop_area(dataset, args.target_lat, args.target_lon,
                        args.context_apothem)

    # Apply feature engineering
    dataset = feature_engineering(dataset)

    print('Dataset after feature engineering:')
    print_dataset_info(dataset)

    # Split the data
    train_set, val_set, test_set = split_data(
        dataset, args.val_ratio, args.test_ratio)

    # Print the split time ranges
    print(
        f'Train set time range: {train_set.time.min().values} - {train_set.time.max().values}')
    print(
        f'Validation set time range: {val_set.time.min().values} - {val_set.time.max().values}')
    print(
        f'Test set time range: {test_set.time.min().values} - {test_set.time.max().values}')

    # Cache the datasets
    print("Caching datasets...")
    cache_data(train_set, args.cache_folder, args.train_file)
    cache_data(val_set, args.cache_folder, args.val_file)
    cache_data(test_set, args.cache_folder, args.test_file)


if __name__ == '__main__':
    main()
