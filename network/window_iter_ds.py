import itertools

import torch
import xarray as xr
import xbatcher
from torch.utils.data import IterableDataset


class WindowIterDS(IterableDataset):
    """
        Iterable dataset that produced sliding windows of data.

        Args:
            ds: Dataset to iterate over
            context_steps: Number of time steps to use as context
            horizon: Number of time steps to predict into the future
            targets: List of target variables
    """

    def __init__(self, ds: xr.Dataset, context_steps: int, horizon: int, targets: list[str],
                 steps: int = 1):
        super(WindowIterDS).__init__()

        self.context_steps = context_steps
        self.horizon = horizon
        self.window_size = context_steps + horizon
        self.steps = steps
        self.targets = targets

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

        return features.to_numpy(), labels.to_numpy()

    def __iter__(self):
        """for sample in self.bgen:
            # Get the input and labels
            yield self._get_xy(sample)"""
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id

        mapped_itr = map(self._get_xy, self.bgen)
        mapped_itr = itertools.islice(
            mapped_itr, worker_id, None, worker_total_num)

        return mapped_itr

    def __getitem__(self, index):
        return self._get_xy(self.bgen[index])

    def __len__(self):
        # Get number of windows
        return len(self.bgen)
