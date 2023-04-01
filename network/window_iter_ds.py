import itertools

import torch
import xarray as xr
import xbatcher
from torch.utils.data import IterableDataset


class WindowIterDS(IterableDataset):
    """
        Iterable dataset that produced sliding windows of data.

        shape = (time, channels, width (latitude), height (longitude))

        Args:
            ds: Dataset to iterate over
            context_steps: Number of time steps to use as context
            horizon: Number of time steps to predict into the future
            targets: List of target variables
    """

    def __init__(self, ds: xr.Dataset, context_steps: int, horizon: int, targets: list[str],
                 device: str, steps: int = 1):
        super(WindowIterDS).__init__()

        self.context_steps = context_steps
        self.horizon = horizon
        self.window_size = context_steps + horizon
        self.steps = steps
        self.targets = targets
        self.device = device

        lat_size = len(ds.latitude)
        long_size = len(ds.longitude)

        # Effectively a rolling sample generator
        self.bgen = xbatcher.BatchGenerator(ds,
                                            input_dims={
                                                'time': self.window_size,
                                                'latitude': lat_size, 'longitude': long_size},
                                            input_overlap={'time': self.steps},
                                            )

    def _get_xy(self, sample):
        # Get the input and labels
        features = sample.isel(time=slice(0, self.context_steps))
        labels = sample.isel(time=slice(
            self.context_steps, self.window_size))

        # Only keep target variables in labels
        labels = labels[self.targets]

        # To DataArray
        features = features.to_array('channels')
        labels = labels.to_array('channels')

        # Crop label width and height to be 4x smaller than context width and height,
        # centered at same location
        # Get current width and height
        width = len(features.longitude)
        height = len(features.latitude)

        # Get center of width and height
        center_width = width // 2
        center_height = height // 2

        # Get target width and height
        target_width = width // 4
        target_height = height // 4

        # Get start and end of target width and height
        start_width = center_width - target_width // 2
        end_width = center_width + target_width // 2
        start_height = center_height - target_height // 2
        end_height = center_height + target_height // 2

        # Crop
        labels = labels.isel(
            longitude=slice(start_width, end_width + 1),
            latitude=slice(start_height, end_height + 1))

        # To numpy
        features = features.transpose(
            'time', 'channels', 'latitude', 'longitude')
        labels = labels.transpose('time', 'channels', 'latitude', 'longitude')

        features = torch.from_numpy(features.to_numpy()).to(
            self.device, dtype=torch.float)
        labels = torch.from_numpy(labels.to_numpy()).to(
            self.device, dtype=torch.float)
        return features, labels

    def __iter__(self):
        """for sample in self.bgen:
            # Get the input and labels
            yield self._get_xy(sample)"""
        if torch.utils.data.get_worker_info() is None:
            yield from map(self._get_xy, self.bgen)

        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id

        mapped_itr = map(self._get_xy, self.bgen)
        mapped_itr = itertools.islice(
            mapped_itr, worker_id, None, worker_total_num)

        return mapped_itr

    def __getitem__(self, index):
        return self._get_xy(self.bgen[index])
