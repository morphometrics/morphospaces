from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray
    based on the patch and stride shape.
    """

    def __init__(
        self,
        dataset: ArrayLike,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
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

        self._slices = self._build_slices(
            dataset=dataset, patch_shape=patch_shape, stride_shape=stride_shape
        )

    @property
    def slices(self):
        return self._slices

    @staticmethod
    def _build_slices(
        dataset: Dict[str, ArrayLike],
        patch_shape: Tuple[int, ...],
        stride_shape: Tuple[int, ...],
    ):
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
        dataset: ArrayLike,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        filter_ignore_index: Tuple[int, ...] = (0,),
        threshold: float = 0.6,
        slack_acceptance: float = 0.01,
    ):
        super().__init__(
            dataset,
            patch_shape,
            stride_shape,
        )

        rand_state = np.random.RandomState(47)

        # load the full image up front to reduce IO
        slicer = ()
        for dim_index in range(dataset.ndim):
            slicer += (slice(0, dataset.shape[dim_index]),)
        label_image = np.copy(dataset[slicer])
        for ignore_value in filter_ignore_index:
            label_image[label_image == ignore_value] = 0

        def ignore_predicate(slice_to_filter):
            patch = label_image[slice_to_filter]
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return (
                non_ignore_counts > threshold
                or rand_state.rand() < slack_acceptance
            )

        # filter slices with less than the requested volume fraction
        # of non-ignore_index
        self._slices = list(filter(ignore_predicate, self.slices))


class PatchManager:
    def __init__(
        self,
        data: Dict[str, ArrayLike],
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
    ):

        self._shape = patch_shape
        self._stride = stride_shape
        self._filter_ignore_index = patch_filter_ignore_index
        self._filter_key = patch_filter_key
        self._threshold = patch_threshold
        self._slack_acceptance = patch_slack_acceptance

        self._slices = self._compute_slices(data)

    def _compute_slices(
        self, data: Dict[str, ArrayLike]
    ) -> List[Tuple[slice, ...]]:
        if self.threshold > 0:
            dataset = data[self.filter_key]
            slice_builder = FilterSliceBuilder(
                dataset=dataset,
                patch_shape=self.shape,
                stride_shape=self.stride,
                filter_ignore_index=self.filter_ignore_index,
                threshold=self.threshold,
                slack_acceptance=self.slack_acceptance,
            )
        elif self.threshold == 0:
            # take the first data array in the dict
            dataset = next(iter(data.values()))
            slice_builder = SliceBuilder(
                dataset,
                self.shape,
                self.stride,
            )
        else:
            raise ValueError(
                "patch_threshold must be >= 0, "
                f"patch_threshold was {self.threshold}"
            )

        return slice_builder.slices

    @property
    def shape(self) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
        return self._shape

    @property
    def stride(self) -> Union[Tuple[int, int, int], Tuple[int, int, int, int]]:
        return self._stride

    @property
    def filter_ignore_index(self) -> Tuple[int, ...]:
        return self._filter_ignore_index

    @property
    def filter_key(self) -> str:
        return self._filter_key

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def slack_acceptance(self) -> float:
        return self._slack_acceptance

    @property
    def slices(self) -> List[Tuple[slice, ...]]:
        return self._slices

    def __len__(self) -> int:
        return len(self.slices)
