from copy import deepcopy
from typing import List, Tuple, Union

import dask.array as da
import h5py
import numpy as np

from morphospaces.datasets._base import BaseTiledDataset


class StandardHDF5Dataset(BaseTiledDataset):
    """
    Implementation of the HDF5 dataset which loads the data from
     all of the H5 files into the memory.
     Fast but might consume a lot of memory.
    """

    def __init__(
        self,
        file_path: str,
        dataset_keys: List[str],
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (
            96,
            96,
            96,
        ),
        stride_shape: Union[
            Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (0,),
        patch_filter_key: str = "label",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        super().__init__(
            file_path=file_path,
            dataset_keys=dataset_keys,
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_filter_key=patch_filter_key,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )

    @staticmethod
    def get_array(file_path, internal_path):
        with h5py.File(file_path, "r") as f:
            ds = f[internal_path][:]
        if ds.ndim == 2:
            # expand dims if 2d
            ds = np.expand_dims(ds, axis=0)
        return ds


class LazyHDF5Dataset(BaseTiledDataset):
    """Implementation of the HDF5 dataset which loads the data lazily.
    It's slower, but has a low memory footprint.
    """

    def __init__(
        self,
        file_path: str,
        dataset_keys: List[str],
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (
            96,
            96,
            96,
        ),
        stride_shape: Union[
            Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (0,),
        patch_filter_key: str = "label",
        patch_threshold: float = 0.6,
        patch_slack_acceptance: float = 0.01,
        store_unique_label_values: bool = False,
    ):
        super().__init__(
            file_path=file_path,
            dataset_keys=dataset_keys,
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_filter_key=patch_filter_key,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )

    @staticmethod
    def get_array(file_path, internal_path):
        return LazyHDF5File(file_path, internal_path)


class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None, mirror_padding=None):
        self.path = path
        self.internal_path = internal_path

        if self.internal_path:
            with h5py.File(self.path, "r") as f:
                self.ndim = f[self.internal_path].ndim
                self.shape = f[self.internal_path].shape

    def ravel(self):
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:].ravel()
        return data

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyHDF5File(self.path, arg)

        if arg == Ellipsis:
            return LazyHDF5File(self.path, self.internal_path)

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][arg]
        data_copy = deepcopy(data)
        del data
        return data_copy

    def to_array(self) -> da.Array:
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:]

        # https://github.com/pytorch/pytorch/issues/11201
        data_copy = deepcopy(data)
        del data
        return data_copy
