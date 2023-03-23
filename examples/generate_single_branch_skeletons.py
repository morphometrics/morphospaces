import os

import numpy as np
from numpy.random import default_rng

from morphospaces.data.data_utils import random_indices_in_bounding_box
from morphospaces.data.skeleton import (
    make_single_branch_point_skeleton_dataset,
)

n_skeletons = 400

# kernel sizes for making the skeletonization target
skeleton_dilation_size = 1
skeleton_gaussian_size = 1
point_gaussian_size = 1.5
point_radius = 2

# make the tip points
tip_points_0 = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 0, 60]),
    upper_right_corner=np.array([75, 20, 75]),
    random_seed=0,
)
tip_points_1 = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 60, 60]),
    upper_right_corner=np.array([75, 75, 75]),
    random_seed=1,
)
tip_points = np.stack((tip_points_0, tip_points_1), axis=1)

# branch point
branch_points = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 20, 30]),
    upper_right_corner=np.array([75, 60, 40]),
    random_seed=2,
)

# root point
root_points = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 20, 5]),
    upper_right_corner=np.array([75, 40, 20]),
    random_seed=3,
)

# dilation sizes
rng = default_rng(4)
dilation_sizes = rng.integers(5, 15, size=(n_skeletons,))

image_shape = (120, 120, 120)


# make the directory
dataset_folder_name = "single_branch_point_skeleton_datasets_20230323"
folder_path = os.path.join(".", dataset_folder_name)
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

for index, (root, branch, tip, dilation) in enumerate(
    zip(root_points, branch_points, tip_points, dilation_sizes)
):
    file_name = os.path.join(folder_path, f"skeleton_dataset_{index}.h5")
    make_single_branch_point_skeleton_dataset(
        file_name=file_name,
        root_point=root,
        branch_point=branch,
        tip_points=tip,
        segmentation_dilation_size=dilation,
        image_shape=image_shape,
        skeleton_dilation_size=skeleton_dilation_size,
        skeleton_gaussian_size=skeleton_gaussian_size,
        point_gaussian_size=point_gaussian_size,
        point_radius=point_radius,
    )
