from typing import Tuple

import numpy as np
from numpy.random import default_rng


def random_indices_in_bounding_box(
    n_points: int,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
    random_seed: float = 42,
) -> np.ndarray:
    """Get an array of random indices contained in a bounding box.

    Parameters
    ----------
    n_points : int
        The number of points to get.
    lower_left_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with lowest coordinate values.
    upper_right_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with the highest coordinate values.
    random_seed : float
        The value to seed the random number generator with.
    """
    rng = default_rng(random_seed)

    return rng.integers(
        lower_left_corner,
        upper_right_corner,
        size=(n_points, len(lower_left_corner)),
    )


def find_indices_within_radius(
    array_shape: Tuple[int, int, int], center_point: np.ndarray, radius: int
) -> np.ndarray:
    index_array = np.indices(array_shape).reshape(len(array_shape), -1).T
    city_block_distance = np.sum(np.abs(index_array - center_point), axis=1)
    within_radius_mask = city_block_distance <= radius
    return index_array[within_radius_mask]


def select_points_in_bounding_box(
    points: np.ndarray,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
) -> np.ndarray:
    """From an array of points, select all points inside a specified
    axis-aligned bounding box.

    Parameters
    ----------
    points : np.ndarray
        The n x d array containing the n, d-dimensional points to check.
    lower_left_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with lowest coordinate values.
    upper_right_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with the highest coordinate values.

    Returns
    -------
    points_in_box : np.ndarray
        The n x d array containing the n points inside of the
        specified bounding box.
    """
    in_box_mask = np.all(
        np.logical_and(
            lower_left_corner <= points, upper_right_corner >= points
        ),
        axis=1,
    )
    return points[in_box_mask]


def draw_line_segment(
    start_point: np.ndarray,
    end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
):
    """Draw a line segment in-place.

    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.

    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    """
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # filter for points within the image
    image_bounds = np.asarray(skeleton_image.shape) - 1
    skeleton_points = select_points_in_bounding_box(
        points=skeleton_points,
        lower_left_corner=np.array([0, 0, 0]),
        upper_right_corner=image_bounds,
    ).astype(int)
    skeleton_image[
        skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]
    ] = fill_value
