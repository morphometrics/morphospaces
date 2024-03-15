import glob
import logging
from typing import Dict, List, Tuple, Union

import dask.array as da
import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import ConcatDataset, Dataset

from morphospaces.datasets.utils import (
    FilterSliceBuilder,
    PatchManager,
    SliceBuilder,
)

logger = logging.getLogger("lightning.pytorch")


class BaseTiledDataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset,
    which iterates over the raw and label datasets
    patch by patch with a given stride.
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
        self.file_path = file_path
        self.patch_filter_key = patch_filter_key
        self.patch_filter_ignore_index = patch_filter_ignore_index
        self.patch_threshold = patch_threshold
        self.patch_slack_acceptance = patch_slack_acceptance

        # load the data (will be lazy if get_array() returns lazy object)
        assert len(dataset_keys) > 0, "dataset_keys must be a non-empty list"
        self.data: Dict[str, ArrayLike] = {
            key: self.get_array(self.file_path, key) for key in dataset_keys
        }

        # make the slices
        assert (
            patch_filter_key in self.data
        ), "patch_filter_key must be a dataset key"
        self.patches = PatchManager(
            data=self.data,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=self.patch_filter_ignore_index,
            patch_filter_key=self.patch_filter_key,
            patch_threshold=self.patch_threshold,
            patch_slack_acceptance=self.patch_slack_acceptance,
        )

        logger.info(
            f"Loaded: {file_path}\n    Number of patches: {self.patch_count}"
        )

        # store the transformation
        self.transform = transform

        if store_unique_label_values:
            self.unique_label_values = self._get_unique_labels()
            logger.info(f"    labels: {self.unique_label_values }")
        else:
            self.unique_label_values = None

    def _get_unique_labels(self) -> List[int]:
        """Get the unique values in the patch_filter_key dataset"""
        unique_labels = set()
        for slice_indices in self.patches.slices:
            array = self.data[self.patch_filter_key]
            data_patch = array[slice_indices]
            label_values = np.unique(data_patch)
            unique_labels.update(label_values)

        return list(unique_labels)

    @property
    def patch_count(self) -> int:
        return len(self.patches)

    @staticmethod
    def get_array(file_path: str, internal_path: str) -> ArrayLike:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, ArrayLike]:
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        slice_indices = self.patches.slices[idx]

        # get the data
        data_patch = {
            key: array[slice_indices] for key, array in self.data.items()
        }

        if self.transform is not None:
            # transform the data
            data_patch = self.transform(data_patch)

        return data_patch

    def __len__(self) -> int:
        return self.patch_count

    @classmethod
    def from_glob_pattern(
        cls,
        glob_pattern: str,
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
        file_paths = glob.glob(glob_pattern)
        datasets = []
        for path in file_paths:
            # make a dataset for each file that matches the pattern
            try:
                dataset = cls(
                    file_path=path,
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
                datasets.append(dataset)
            except AssertionError:
                logger.info(f"{path} skipped")

        if store_unique_label_values:
            unique_label_values = set()
            for dataset in datasets:
                # get the unique label values across all datasets
                unique_label_values.update(dataset.unique_label_values)

            return ConcatDataset(datasets), list(unique_label_values)

        return ConcatDataset(datasets)


class BaseTiledDataset2(Dataset):
    """
    Implementation of torch.utils.data.Dataset,
     which iterates over the raw and label datasets
     patch by patch with a given stride.
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
        logger.info(f"creating dataset: {file_path}")
        assert stage in ["train", "val", "test"]
        if stage in ["train", "val"]:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert (
                    len(mirror_padding) == 3
                ), f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = stage
        self.file_path = file_path

        self.raw = self.get_array(file_path, raw_internal_path)

        self.transform = transform

        if stage != "test":
            # create label/weight transform only in train/val phase
            if label_internal_path is not None:
                self.label = self.get_array(
                    self.file_path, label_internal_path
                )
            else:
                self.label = None

            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_map = self.get_array(
                    self.file_path, weight_internal_path
                )
            else:
                self.weight_map = None

            self._check_volume_sizes(self.raw, self.label)
        else:
            # 'test' phase used only for predictions
            # so ignore the label dataset
            self.label = None
            self.weight_map = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                if isinstance(self.raw, da.Array):
                    self._raw.mirror_padding = self.mirror_padding
                else:
                    z, y, x = self.mirror_padding
                    pad_width = ((z, z), (y, y), (x, x))

                    if self.raw.ndim == 4:
                        channels = [
                            np.pad(r, pad_width=pad_width, mode="reflect")
                            for r in self.raw
                        ]
                        self.raw = np.stack(channels)
                    else:
                        self.raw = np.pad(
                            self.raw, pad_width=pad_width, mode="reflect"
                        )

        # build slice indices for raw and label data sets
        if (patch_threshold > 0) and (self.label is not None):
            slice_builder = FilterSliceBuilder(
                self.raw,
                self.label,
                self.weight_map,
                patch_shape,
                stride_shape,
                filter_ignore_index=patch_filter_ignore_index,
                threshold=patch_threshold,
                slack_acceptance=patch_slack_acceptance,
            )
        elif patch_threshold == 0:
            slice_builder = SliceBuilder(
                self.raw,
                self.label,
                self.weight_map,
                patch_shape,
                stride_shape,
                ignore_index=patch_filter_ignore_index,
                threshold=patch_threshold,
                slack_acceptance=patch_slack_acceptance,
            )
        else:
            raise ValueError(
                "patch_threshold must be >= 0, "
                f"patch_threshold was {patch_threshold}"
            )

        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f"Number of patches: {self.patch_count}")

    @property
    def raw(self):
        if isinstance(self._raw, np.ndarray):
            return self._raw
        return self._raw.to_array()

    @raw.setter
    def raw(self, raw):
        self._raw = raw

    @property
    def label(self):
        if self._label is None:
            return None
        elif isinstance(self._label, np.ndarray):
            return self._label
        else:
            return self._label.to_array()

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def weight_map(self):
        if self._weight_map is None:
            return None
        elif isinstance(self._weight_map, np.ndarray):
            return self._weight_map
        else:
            return self._weight_map.to_array()

    @weight_map.setter
    def weight_map(self, weight_map):
        self._weight_map = weight_map

    @staticmethod
    def get_array(file_path, internal_path):
        raise NotImplementedError

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]

        # get the raw data patch for a given slice
        data_patch = {"raw": np.asarray(self.raw[raw_idx])}

        # add the label data
        if self.label is not None:
            label_idx = self.label_slices[idx]
            data_patch.update({"label": np.asarray(self.label[label_idx])})

        # add the weight maps
        if self.weight_map is not None:
            weight_idx = self.weight_slices[idx]
            data_patch.update(
                {"weights": np.asarray(self.weight_map[weight_idx])}
            )

        if self.transform is not None:
            # transform the data
            data_patch = self.transform(data_patch)

        return data_patch

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [
            3,
            4,
        ], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
        assert label.ndim in [
            3,
            4,
        ], "Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)"

        assert _volume_shape(raw) == _volume_shape(
            label
        ), "Raw and labels have to be of the same size"

    @classmethod
    def from_glob_pattern(
        cls,
        glob_pattern: str,
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

        file_paths = glob.glob(glob_pattern)
        datasets = [
            cls(
                file_path=path,
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
            for path in file_paths
        ]

        return ConcatDataset(datasets)
