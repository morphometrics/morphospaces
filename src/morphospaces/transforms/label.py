from typing import Dict

import numpy as np
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
