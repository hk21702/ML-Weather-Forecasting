"""
Preprocessing of raw data from the ERA5 dataset to a useable state.
"""

import argparse
import concurrent.futures
import dataclasses
import os
import sys
import time

import cdsapi
import netCDF4
import numpy as np
import pandas as pd
from netCDF4 import Dataset


def get_args() -> argparse.Namespace:
    """Parses the command line arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess ERA5 data')

    # NetCRf folder path
    parser.add_argument('--netcdf_folder', type=str, default='data',
                        help='Path to the folder containing the NetCDF files')

    return parser.parse_args()


def main() -> None:
    pass
