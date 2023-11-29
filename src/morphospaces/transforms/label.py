from typing import Dict, List, Union

import numpy as np
from monai.data import MetaTensor
from skimage.measure import block_reduce
from skimage.segmentation import find_boundaries


class LabelToBoundaryd:
    """Convert dense labels to a boundary image.

    This is intended to be used with datasets where the
    data are loaded as a dictionary.

    Parameters
    ----------
    label_key : str
        The key in the dataset for the label image.
    background_value: int
        The value in the label image corresponding to background.
        The default value is 0.
    connectivity : int
        The connectivity rule for determine pixel neighborhoods.
        See the skimage documentation for the find_boundaries
        function for details. Default value is 2.
    mode : str
        The mode for drawing boundaries. See the skimage documentation
        for the find_boundaries function for details.
        Default value is "thick".
    """

    def __init__(
        self,
        label_key: str,
        background_value: int = 0,
        connectivity: int = 2,
        mode: str = "thick",
    ):
        self.label_key = label_key
        self.background_value = background_value
        self.connectivity = connectivity
        self.mode = mode

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        label_image = data_item[self.label_key]

        label_image_ndim = label_image.ndim
        if label_image_ndim == 4:
            assert (
                label_image.shape[0] == 1
            ), "label image must have singleton channel dimension"
            label_image = label_image[0]

        boundary_image = find_boundaries(
            label_image,
            background=self.background_value,
            mode=self.mode,
            connectivity=self.connectivity,
        )

        if label_image_ndim == 4:
            # if the original image was 4D, expand dims
            boundary_image = np.expand_dims(boundary_image, axis=0)

        data_item.update({self.label_key: boundary_image})
        return data_item


class LabelToMaskd:
    """Convert dense labels to a mask image.

    Parameters
    ----------
    input_key : str
        The key to make mask from
    output_key : str
        The key to save the mask to
    background_value : int
        The value in the label image corresponding to background.
        All voxels matching the background_value are set to False.
        All others are True.
    """

    def __init__(
        self, input_key: str, output_key: str, background_value: int = 0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.background_value = background_value

    def __call__(self, data_item: Dict[str, np.ndarray]):
        mask = data_item[self.input_key] != self.background_value

        data_item.update({self.output_key: mask})
        return data_item


class ExtractSingleLabeld:
    """Create a new label image containing a single class from a label image.

    The resulting mask will have the same shape as the input label image and
    have the selected label value set to 1.

    Parameters
    ----------
    input_key : str
        The key to make mask from
    output_key : str
        The key to save the mask to
    label_value : int
        The value in the label to extract.
    """

    def __init__(self, input_key: str, output_key: str, label_value: int = 1):
        self.input_key = input_key
        self.output_key = output_key
        self.label_value = label_value

    def __call__(self, data_item: Dict[str, np.ndarray]):
        original_labels = data_item[self.input_key]
        new_labels = np.zeros_like(original_labels)
        new_labels[original_labels == self.label_value] = 1

        data_item.update({self.output_key: new_labels})
        return data_item


class DownscaleLabelsd:
    """Downscale the skeleton ground truth.

    Parameters
    ----------
    label_key : str
        The key in the dataset for the label image.
    downscaling_factors : List[int]
        The factors by which to downscale the label image before
        making the ground truth.
    """

    def __init__(
        self,
        label_key: str,
        downscaling_factors: List[int],
    ):
        self.label_key = label_key
        self.downscaling_factors = downscaling_factors

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        label_image = data_item[self.label_key]

        if isinstance(label_image, MetaTensor):
            label_image = label_image.array

        for downscaling_factor in self.downscaling_factors:
            # reduce the label image
            reduced_labels = block_reduce(
                label_image, block_size=downscaling_factor, func=np.max
            )

            # store the downscaled data
            reduced_labels_key = (
                f"{self.label_key}_reduced_{downscaling_factor}"
            )
            data_item.update({reduced_labels_key: reduced_labels})

        return data_item


class LabelsAsFloat32:
    """Convert a label image to a float32 array.

    Parameters
    ----------
    keys : Union[str, List[str]]
        The keys in the dataset to apply the transform to.
    """

    def __init__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys: List[str] = keys

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        for key in self.keys:
            image = data_item[key]
            data_item.update({key: np.asarray(image, dtype=np.single)})

        return data_item


class LabelsAsLong:
    """Convert a label image to a long array.

    Parameters
    ----------
    keys : Union[str, List[str]]
        The keys in the dataset to apply the transform to.
    """

    def __init__(self, keys: Union[str, List[str]]):
        if isinstance(keys, str):
            keys = [keys]
        self.keys: List[str] = keys

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        for key in self.keys:
            image = data_item[key]
            data_item.update({key: np.asarray(image, dtype=np.int_)})

        return data_item
