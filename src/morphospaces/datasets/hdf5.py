from copy import deepcopy
from typing import Tuple

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
        file_path,
        stage,
        transform,
        patch_shape: Tuple[int, ...] = (96, 96, 96),
        stride_shape: Tuple[int, ...] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (0,),
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        mirror_padding=(16, 32, 32),
        raw_internal_path="raw",
        label_internal_path="label",
        weight_internal_path=None,
    ):
        super().__init__(
            file_path=file_path,
            stage=stage,
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            mirror_padding=mirror_padding,
            raw_internal_path=raw_internal_path,
            label_internal_path=label_internal_path,
            weight_internal_path=weight_internal_path,
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
        file_path,
        stage,
        transform,
        patch_shape: Tuple[int, ...] = (96, 96, 96),
        stride_shape: Tuple[int, ...] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (0,),
        patch_threshold: float = 0,
        patch_slack_acceptance=0.01,
        mirror_padding=(16, 32, 32),
        raw_internal_path="raw",
        label_internal_path="label",
        weight_internal_path=None,
    ):
        super().__init__(
            file_path=file_path,
            stage=stage,
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            mirror_padding=mirror_padding,
            raw_internal_path=raw_internal_path,
            label_internal_path=label_internal_path,
            weight_internal_path=weight_internal_path,
        )

        # logger.info("Using modified HDF5Dataset!")

    @staticmethod
    def get_array(file_path, internal_path):
        return LazyHDF5File(file_path, internal_path)


class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None, mirror_padding=None):
        self.path = path
        self.internal_path = internal_path
        self.mirror_padding = mirror_padding
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
        # make the dask array
        # h5py_file = h5py.File(self.path, "r")

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:]

        # https://github.com/pytorch/pytorch/issues/11201
        data_copy = deepcopy(data)
        del data
        return data_copy
        # lazy_array = da.from_array(h5py_dataset, chunks=h5py_dataset.chunks)
        #
        # # add padding if necessary
        # if self.mirror_padding is not None:
        #     z, y, x = self.mirror_padding
        #     pad_width = ((z, z), (y, y), (x, x))
        #
        #     if lazy_array.ndim == 4:
        #         channels = [
        #             da.pad(r, pad_width=pad_width, mode="reflect")
        #             for r in lazy_array
        #         ]
        #         lazy_array = da.stack(channels)
        #     else:
        #         lazy_array = da.pad(
        #             lazy_array, pad_width=pad_width, mode="reflect"
        #         )
        # # h5py_file.close()
        # return lazy_array
