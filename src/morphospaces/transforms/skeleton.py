from typing import Dict, List, Tuple

import numpy as np
from monai.data import MetaTensor
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.measure import block_reduce


def make_skeleton_target(
    skeleton_mask: np.ndarray,
    gaussian_sigma: float,
    normalization_neighborhood_size: int,
) -> np.ndarray:
    """
    Make a skeleton target from a skeleton mask.

    Parameters
    ----------
    skeleton_mask : np.ndarray
        A skeleton mask.
    gaussian_sigma : float
        The sigma for the gaussian blur.
    normalization_neighborhood_size : int
        The size of the neighborhood for computing the max for normalization.

    Returns
    -------
    skeleton_target : np.ndarray
        The prediction target for the skeleton.
    """
    # blur the target
    skeleton_target = gaussian_filter(
        skeleton_mask.astype(float), gaussian_sigma
    )

    # compute normalization values
    normalization_factors = maximum_filter(
        skeleton_target, size=normalization_neighborhood_size
    )

    # only normalize pixels where the local max is greater than 0
    normalization_mask = normalization_factors > 0
    values_to_normalize = skeleton_target[normalization_mask]
    masked_normalization_factors = normalization_factors[normalization_mask]
    skeleton_target[normalization_mask] = (
        values_to_normalize / masked_normalization_factors
    )

    return skeleton_target


def make_downscaled_skeleton_ground_truth(
    label_image: np.ndarray,
    downscaling_factor: int = 2,
    gaussian_sigma: float = 1,
    norm_neighborhood_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a ground truth image from a label image.

    Parameters
    ----------
    label_image : np.ndarray
        The label image.
    downscaling_factor : int
        The factor by which to downscale the label image before
        making the ground truth.
    gaussian_sigma : float
        The sigma for the gaussian blur.
    norm_neighborhood_size : int
        The size of the neighborhood for computing the max for normalization.

    Returns
    -------
    skeleton_target : np.ndarray
        The prediction target for the skeleton.
    reduced_labels : np.ndarray
        The downscaled label image.
    """
    # reduce the label image
    reduced_labels = block_reduce(
        label_image, block_size=downscaling_factor, func=np.max
    )

    skeleton_mask = reduced_labels == 2
    skeleton_target = make_skeleton_target(
        skeleton_mask=skeleton_mask,
        gaussian_sigma=gaussian_sigma,
        normalization_neighborhood_size=norm_neighborhood_size,
    )

    return skeleton_target, reduced_labels


class DownscaleSkeletonGroundTruth:
    """Downscale the skeleton ground truth.

    Parameters
    ----------
    label_key : str
        The key in the dataset for the label image.
    skeletonization_target_key : str
        The key in the dataset for the skeletonization target.
    downscaling_factors : List[int]
        The factors by which to downscale the label image before
        making the ground truth.
    gaussian_sigma : float
        The sigma for the gaussian blur.
    normalization_neighborhood_sizes : int
        The size of the neighborhood for computing the max for normalization.
    """

    def __init__(
        self,
        label_key: str,
        skeletonization_target_key: str,
        downscaling_factors: List[int],
        gaussian_sigma: float = 1,
        normalization_neighborhood_sizes: int = 5,
    ):
        self.label_key = label_key
        self.skeletonization_target_key = skeletonization_target_key
        self.downscaling_factors = downscaling_factors
        self.gaussian_sigmas = gaussian_sigma
        self.normalization_neighborhood_sizes = (
            normalization_neighborhood_sizes
        )

    def __call__(
        self, data_item: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        label_image = data_item[self.label_key]

        if isinstance(label_image, MetaTensor):
            label_image = label_image.array

        if label_image.ndim == 4:
            # do the downscaling on ZYX, but expand back to CZYX after
            label_image = np.squeeze(label_image, axis=0)
            expand_dims = True
        else:
            expand_dims = False
        for scale in self.downscaling_factors:

            (
                skeleton_target,
                reduced_labels,
            ) = make_downscaled_skeleton_ground_truth(
                label_image=label_image,
                downscaling_factor=scale,
                gaussian_sigma=self.gaussian_sigmas,
                norm_neighborhood_size=self.normalization_neighborhood_sizes,
            )

            if expand_dims:
                # expand up to CZYX if the input was CZYX
                skeleton_target = np.expand_dims(skeleton_target, axis=0)
                reduced_labels = np.expand_dims(reduced_labels, axis=0)

            # store the downscaled data
            skeleton_key = f"{self.skeletonization_target_key}_reduced_{scale}"
            reduced_labels_key = f"{self.label_key}_reduced_{scale}"
            new_ground_truth = {
                skeleton_key: skeleton_target,
                reduced_labels_key: reduced_labels,
            }

            data_item.update(new_ground_truth)

        return data_item
