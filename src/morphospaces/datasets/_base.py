import glob
import logging
from typing import Tuple

import dask.array as da
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from morphospaces.datasets.utils import FilterSliceBuilder, SliceBuilder

logger = logging.getLogger("lightning")


class BaseTiledDataset(Dataset):
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
                ignore_index=patch_filter_ignore_index,
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
