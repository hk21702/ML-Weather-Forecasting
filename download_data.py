"""Script to download ERA5 data from the Copernicus Climate Data Store (CDS)"""
import argparse
import concurrent.futures
import dataclasses
import os
import sys
import time

import cdsapi


@dataclasses.dataclass
class VariableSet:
    """Class to store the variables to download"""
    variables: list[str]
    label: str


@dataclasses.dataclass
class RegionBox:
    """Class to store the region box coordinates"""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def to_list(self) -> list:
        """Returns the region box coordinates as a list"""
        return [self.lat_min, self.lon_min, self.lat_max, self.lon_max]


def download_data(cds: cdsapi.Client, year: str,
                  region_box: RegionBox, variables: VariableSet) -> None:
    """Downloads the data for the given year and region box"""
    cds.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables.variables,
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': region_box.to_list(),
            'format': 'netcdf',
        },
        f'data/{year}_{region_box.to_list()}_{variables.label}.nc')


def get_args() -> argparse.Namespace:
    """Returns the arguments passed to the script and ensures they are valid"""
    north_america = RegionBox(20, 70, -170, -50)

    # Argument parser
    parser = argparse.ArgumentParser(
        description='Download ERA5 data from the Copernicus Climate Data Store (CDS)')

    # Region box using lat/lon coordinates
    parser.add_argument('--lat_min', type=float, default=north_america.lat_min,
                        help='Minimum latitude')
    parser.add_argument('--lat_max', type=float, default=north_america.lat_max,
                        help='Maximum latitude')
    parser.add_argument('--lon_min', type=float, default=north_america.lon_min,
                        help='Minimum longitude')
    parser.add_argument('--lon_max', type=float, default=north_america.lon_max,
                        help='Maximum longitude')

    # Years interval to download
    parser.add_argument('--start_year', type=str, default='2000',
                        help='Start year to download data for')
    parser.add_argument('--end_year', type=str, default='2022',
                        help='End year to download data for')

    # Parse arguments
    args = parser.parse_args()

    # Assert end year is greater than start year
    assert int(args.end_year) >= int(
        args.start_year), 'End year must be greater than start year'

    # Assert end year is year before current year
    assert int(args.end_year) < int(time.strftime('%Y')
                                    ), 'End year must be year before current year'

    # Assert start year is 1940 or later
    assert int(args.start_year) >= 1940, 'Start year must be 1940 or later'

    # Assert lat lon coordinates are valid
    assert args.lat_min < args.lat_max, 'Minimum latitude must be less than maximum latitude'
    assert args.lon_min < args.lon_max, 'Minimum longitude must be less than maximum longitude'
    assert args.lat_min >= -90 and args.lat_max <= 90, 'Latitude must be between -90 and 90'
    assert args.lon_min >= - \
        180 and args.lon_max <= 180, 'Longitude must be between -180 and 180'

    return args


def get_cds() -> cdsapi.Client:
    """Wrapper to return the CDS API client
    and catch API key errors"""
    try:
        return cdsapi.Client()
    except Exception as error:
        # Check if .cdsapirc file exists at home directory
        if not os.path.isfile(os.path.join(os.path.expanduser('~'), '.cdsapirc')):
            print('Error: Could not find CDS API key')
            print('Please create a .cdsapirc file at your home directory')
            print('See https://cds.climate.copernicus.eu/api-how-to for more information')
            sys.exit(1)

        print('Error: Could not connect to CDS API')
        print('Check that your API key is valid and that you have an internet connection')
        print('Error:', error)
        sys.exit(1)


def main() -> None:
    """Main function"""
    args = get_args()

    # Create region box
    region_box = RegionBox(args.lat_min, args.lat_max,
                           args.lon_min, args.lon_max)

    cds = get_cds()

    vset1 = VariableSet([
        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
        '2m_temperature', 'mean_sea_level_pressure', 'mean_wave_direction',
        'mean_wave_period', 'sea_surface_temperature',
        'significant_height_of_combined_wind_waves_and_swell',
        'surface_pressure', 'total_precipitation',
    ], 'vset1')

    vset2 = VariableSet([
        '10m_u_component_of_neutral_wind', '10m_v_component_of_neutral_wind', 'evaporation',
        'forecast_albedo', 'high_cloud_cover', 'high_vegetation_cover',
        'low_cloud_cover', 'low_vegetation_cover', 'medium_cloud_cover',
        'skin_temperature', 'soil_temperature_level_1', 'total_cloud_cover',
        'volumetric_soil_water_layer_1',
    ],
        'vset2')

    # Create data folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download data using threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for year in range(int(args.start_year), int(args.end_year) + 1):
            print('Sending request for', year)
            executor.submit(download_data, cds, str(year), region_box, vset1)
            executor.submit(download_data, cds, str(year), region_box, vset2)


if __name__ == '__main__':
    main()
