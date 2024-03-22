from typing import Tuple

import numpy as np
import zarr

from morphospaces.datasets._base import BaseTiledDataset


class LazyTiledZarrDataset(BaseTiledDataset):
    """Implementation of the zarr dataset which loads the data lazily.
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
        store_unique_label_values: bool = False,
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
            store_unique_label_values=store_unique_label_values,
        )

    @staticmethod
    def get_array(file_path, internal_path):
        return LazyZarrFile(file_path, internal_path)


class LazyZarrFile:
    """Implementation of the LazyZarrFile class for the LazyZarrDataset."""

    def __init__(self, path, internal_path=None):
        self.path = path
        self.internal_path = internal_path
        if self.internal_path:
            array = self.to_array()
            try:
                self.ndim = array.ndim
                self.shape = array.shape
            except AttributeError:
                print(path)
            except KeyError:
                print(path)

    def to_array(self) -> zarr.core.Array:
        # return da.from_zarr(self.path, self.internal_path)
        return zarr.open(self.path, path=self.internal_path)

    def ravel(self) -> np.ndarray:
        return np.ravel(self.to_array())

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyZarrFile(self.path, arg)

        if arg == Ellipsis:
            return LazyZarrFile(self.path, self.internal_path)

        array = self.to_array()
        item = array[arg]
        del array
        return item
