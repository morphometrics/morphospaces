from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray
    based on the patch and stride shape.
    """

    def __init__(
        self,
        raw_dataset,
        label_dataset,
        weight_dataset,
        patch_shape,
        stride_shape,
        **kwargs,
    ):
        """
        :param raw_dataset: ndarray of raw data
        :param label_dataset: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)

        self._raw_slices = self._build_slices(
            raw_dataset, patch_shape, stride_shape
        )
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(
                label_dataset, patch_shape, stride_shape
            )
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(
                weight_dataset, patch_shape, stride_shape
            )
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, "Sample size has to be bigger than the patch size"
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(
        self,
        raw_dataset,
        label_dataset,
        weight_dataset,
        patch_shape,
        stride_shape,
        ignore_index=(0,),
        threshold=0.6,
        slack_acceptance=0.01,
        **kwargs,
    ):
        super().__init__(
            raw_dataset,
            label_dataset,
            weight_dataset,
            patch_shape,
            stride_shape,
            **kwargs,
        )
        if label_dataset is None:
            return

        rand_state = np.random.RandomState(47)

        # load the full image up front to reduce IO
        slicer = ()
        for dim_index in range(label_dataset.ndim):
            slicer += (slice(0, label_dataset.shape[dim_index]),)
        label_image = np.copy(label_dataset[slicer])
        for ignore_value in ignore_index:
            label_image[label_image == ignore_value] == 0

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_image[label_idx]
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return (
                non_ignore_counts > threshold
                or rand_state.rand() < slack_acceptance
            )

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class PatchManager:
    def __init__(
        self,
        data: ArrayLike,
        patch_shape: Tuple[int, ...] = (96, 96, 96),
        stride_shape: Tuple[int, ...] = (24, 24, 24),
        patch_filter_index: Tuple[int, ...] = (0,),
        patch_filter_key: str = "label",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
    ):

        self._shape = patch_shape
        self._stride = stride_shape
        self._filter_index = patch_filter_index
        self._filter_key = patch_filter_key
        self._threshold = patch_threshold
        self._slack_acceptance = patch_slack_acceptance

        self._slices = self._compute_slices(data)

    def _compute_slices(self, data: ArrayLike):
        if self.threshold > 0:
            slice_builder = FilterSliceBuilder(
                self.raw,
                self.label,
                self.weight_map,
                self.shape,
                self.stride,
                filter_index=self.filter_index,
                threshold=self.threshold,
                slack_acceptance=self.slack_acceptance,
            )
        elif self.threshold == 0:
            slice_builder = SliceBuilder(
                self.raw,
                self.label,
                self.weight_map,
                self.shape,
                self.stride,
                filter_index=self.filter_index,
                threshold=self.threshold,
                slack_acceptance=self.slack_acceptance,
            )
        else:
            raise ValueError(
                "patch_threshold must be >= 0, "
                f"patch_threshold was {self.threshold}"
            )

        return slice_builder.slices

    @property
    def shape(self):
        return self._shape

    @property
    def stride(self):
        return self._stride

    @property
    def filter_index(self):
        return self._filter_index

    @property
    def filter_key(self):
        return self._filter_key

    @property
    def threshold(self):
        return self._threshold

    @property
    def slack_acceptance(self):
        return self._slack_acceptance

    @property
    def slices(self):
        return self._slices
