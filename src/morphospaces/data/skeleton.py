from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
from skimage.morphology import ball, binary_dilation

from morphospaces.io.hdf5 import write_multi_dataset_hdf


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


def make_single_branch_point_skeleton(
    root_point: np.ndarray,
    branch_point: np.ndarray,
    tip_points: np.ndarray,
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    r"""Make a skeleton with a single branching point.
                       _______ tip
                      /
    root_point -----< branch_point
                     \_______ tip

    Parameters
    ----------
    root_point : np.ndarray
        (d,) array containing the starting point for the skeleton.
    branch_point : np.ndarray
        (d,) array containing the point where the root branches into the tips.
    tip_points : np.ndarray
        (n, d) array containing the tip n, d-dimensional tip points.
    image_shape : Tuple[int, int, int]
        Shape of the image to embed the skeleton into.

    Returns
    -------
    skeleton_image : np.ndarray
        The image containing the specified skeleton.
    """
    # make the image to embed the skeleton into
    skeleton_image = np.zeros(image_shape)

    # add the root
    draw_line_segment(
        start_point=root_point,
        end_point=branch_point,
        skeleton_image=skeleton_image,
    )

    # add the tips
    for tip_point in tip_points:
        draw_line_segment(
            start_point=branch_point,
            end_point=tip_point,
            skeleton_image=skeleton_image,
        )
    return skeleton_image


def compute_skeleton_vector_field(
    skeleton_image: np.ndarray, segmentation_image: np.ndarray
) -> np.ndarray:
    """Compute the vector field pointing towards the nearest
    skeleton point for each voxel in the segmentation.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The image containing the skeleton. Skeleton should
        be a single value greater than zero.
    segmentation_iamge : np.ndarray
        The segmentation corresponding to the skeleton.

    Returns
    -------
    vector_image : np.ndarray
        The (3, m, n, p) image containing the vector field of the
        (m, n, p) segmentation/skeleton image.
        vector_image[0, ...] contains the 0th component of the vector field.
        vector_image[1, ...] contains the 1st component of the vector field.
        vector_image[2, ...] contains the 2nd component of the vector field.
    """
    # get the skeleton coordinates and make a tree
    skeleton_coordinates = np.column_stack(np.where(skeleton_image))
    skeleton_kdtree = KDTree(skeleton_coordinates)

    # get the segmentation coordinates
    segmentation_coordinates = np.column_stack(np.where(segmentation_image))

    # get the nearest skeleton point for each
    (
        distance_to_skeleton,
        nearest_skeleton_point_indices,
    ) = skeleton_kdtree.query(segmentation_coordinates)
    nearest_skeleton_point = skeleton_coordinates[
        nearest_skeleton_point_indices
    ]

    # get the pixels that are not containing the skeleton
    non_skeleton_mask = distance_to_skeleton > 0
    nearest_skeleton_point = nearest_skeleton_point[non_skeleton_mask]
    segmentation_coordinates = segmentation_coordinates[non_skeleton_mask]
    distance_to_skeleton = distance_to_skeleton[non_skeleton_mask]

    # get the vector pointing to the skeleton
    vector_to_skeleton = nearest_skeleton_point - segmentation_coordinates
    unit_vector_to_skeleton = (
        vector_to_skeleton / distance_to_skeleton[:, None]
    )

    # flip and rescale the distance
    normalized_distance = distance_to_skeleton / distance_to_skeleton.max()
    flipped_distance = 1 - normalized_distance

    # scale the vectors by the flipped/normalized distance
    scaled_vector_to_skeleton = (
        unit_vector_to_skeleton * flipped_distance[:, None]
    )

    # embed the vectors into an image
    image_shape = (3,) + skeleton_image.shape
    vector_image = np.zeros(image_shape)
    for dimension_index in range(3):
        vector_image[
            dimension_index,
            segmentation_coordinates[:, 0],
            segmentation_coordinates[:, 1],
            segmentation_coordinates[:, 2],
        ] = scaled_vector_to_skeleton[:, dimension_index]

    return vector_image


def make_segmentation_distance_image(
    segmentation_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create images where each voxel describes the relationship
    to the nearest background pixel.

    Parameters
    ----------
    segmentation_image : np.ndarray
        The image containing the segmentation.

    Returns
    -------
    distance_image : np.ndarray
        Image where each voxel is the euclidian distance to the nearest
        background pixel.
    scaled_background_vector_image : np.ndarray
        (3, m, n, p) image where each voxel is a vector pointing
        towards the nearest background pixel.
        The vector magnitude is normalized to the maximum distance.
        vector_image[0, ...] contains the 0th component of the vector field.
        vector_image[1, ...] contains the 1st component of the vector field.
        vector_image[2, ...] contains the 2nd component of the vector field.
    """
    distance_image, background_indices = distance_transform_edt(
        segmentation_image, return_distances=True, return_indices=True
    )

    # get the vector pointing towards the nearest background
    image_index_image = np.indices(segmentation_image.shape)
    background_vector_image = background_indices - image_index_image

    # scale the vectors by the max distance
    max_distance = distance_image.max()
    scaled_background_vector_image = background_vector_image / max_distance

    return distance_image, scaled_background_vector_image


def make_single_branch_point_skeleton_dataset(
    file_name: str,
    root_point: np.ndarray,
    branch_point: np.ndarray,
    tip_points: np.ndarray,
    dilation_size: int,
    image_shape: Tuple[int, int, int],
):
    """Write an hdf5 dataset containing a single branch point skeleton
    and auxillary images.

    Parameters
    ----------
    file_name : str
        The path to the file to write the dataset to.
    root_point : np.ndarray
        (d,) array containing the starting point for the skeleton.
    branch_point : np.ndarray
        (d,) array containing the point where the root branches into the tips.
    tip_points : np.ndarray
        (n, d) array containing the tip n, d-dimensional tip points.
    dilation_size : np.ndarray
        The size of the morphological dilation (ball kernel) to apply in to
        create the segmentation from the skeleton.
    image_shape : Tuple[int, int, int]
        Shape of the image to embed the skeleton into.

    """
    skeleton_image = make_single_branch_point_skeleton(
        root_point=root_point,
        branch_point=branch_point,
        tip_points=tip_points,
        image_shape=image_shape,
    )

    segmentation_image = binary_dilation(
        skeleton_image, footprint=ball(dilation_size)
    )
    vector_image = compute_skeleton_vector_field(
        skeleton_image=skeleton_image, segmentation_image=segmentation_image
    )

    (
        segmentation_distance_image,
        background_vector_image,
    ) = make_segmentation_distance_image(segmentation_image=segmentation_image)

    # store all end points in a single array
    end_points = np.concatenate(
        (np.atleast_2d(root_point), np.atleast_2d(tip_points))
    )

    # write the file
    write_multi_dataset_hdf(
        file_path=file_name,
        compression="gzip",
        skeleton_image=skeleton_image,
        segmentation_image=segmentation_image,
        branch_point=np.atleast_2d(branch_point),
        end_points=end_points,
        skeleton_vector_image=vector_image,
        segmentation_distance_image=segmentation_distance_image,
        background_vector_image=background_vector_image,
    )
