from typing import Dict, List, Union

import numpy as np
from skimage.util import img_as_float32


class ExpandDimsd:
    """Prepend a singleton dimensions.

    For example, this would go from shape (Z, Y, X) -> (1, Z, Y, X).

    This is intended to be used with datasets where the
    data are loaded as a dictionary.

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
            data_item.update({key: np.expand_dims(image, axis=0)})

        return data_item


class Squeezed:
    """Remove singleton dimensions.

    For example, this would go from shape (1, Z, Y, X) -> (Z, Y, X).

    This is intended to be used with datasets where the
    data are loaded as a dictionary.

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
            data_item.update({key: image.squeeze()})

        return data_item


class ImageAsFloat32:
    """Convert an image to a float32 ranging from 0 to 1.

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
            data_item.update({key: img_as_float32(image)})

        return data_item


class MaskFromVectorField:
    """Make a mask that is true where vector field magnitude is greater than 0.

    The vector field is assumed to be CZYX.
    The resulting mask has shape (z, y, x).

    Parameters
    ----------
    input_key : str
        The key for the vector field
    output_key : str
        The key to save the
    """

    def __init__(self, input_key: str, output_key: str):
        self.key = input_key
        self.output_key = output_key

    def __call__(self, data_item: Dict[str, np.ndarray]):
        vector_field = data_item[self.key]
        non_zero_components = vector_field != 0

        mask = np.sum(non_zero_components, axis=0) > 0

        data_item.update({self.output_key: np.expand_dims(mask, axis=0)})
        return data_item


class NormVectorField:
    """Compute the magnitude of a vector field.

    The vector field is assumed to be CZYX.
    The resulting image has shape (1, z, y, x).

    Parameters
    ----------
    input_key : str
        The key for the vector field
    output_key : str
        The key to save the image to.
    """

    def __init__(self, input_key: str, output_key: str):
        self.key = input_key
        self.output_key = output_key

    def __call__(self, data_item: Dict[str, np.ndarray]):
        vector_field = data_item[self.key]
        magnitude = np.linalg.norm(vector_field, axis=0)
        data_item.update({self.output_key: np.expand_dims(magnitude, axis=0)})
        return data_item
