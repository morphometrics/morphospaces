import numpy as np
from numpy.random import default_rng

from morphospaces.data.data_utils import random_indices_in_bounding_box
from morphospaces.data.skeleton import make_single_branch_point_skeleton_dataset


n_skeletons = 300

# make the tip points
tip_points_0 = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 0, 60]),
    upper_right_corner=np.array([75, 20, 75]),
    random_seed=0
)
tip_points_1 = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 60, 60]),
    upper_right_corner=np.array([75, 75, 75]),
    random_seed=1
)
tip_points = np.stack((tip_points_0, tip_points_1), axis=1)

# branch point
branch_points = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 20, 30]),
    upper_right_corner=np.array([75, 60, 40]),
    random_seed=2
)

# root point
root_points = random_indices_in_bounding_box(
    n_points=n_skeletons,
    lower_left_corner=np.array([0, 20, 5]),
    upper_right_corner=np.array([75, 40, 20]),
    random_seed=3
)

# dilation sizes
rng = default_rng(4)
dilation_sizes = rng.integers(5, 15, size=(n_skeletons,))

image_shape = (80, 80, 80)

for index, (root, branch, tip, dilation) in enumerate(
    zip(root_points, branch_points, tip_points, dilation_sizes)
):
    make_single_branch_point_skeleton_dataset(
        file_name=f"./skeleton_datasets/skeleton_dataset_{index}.h5",
        root_point=root,
        branch_point=branch,
        tip_points=tip,
        dilation_size=dilation,
        image_shape=image_shape
    )
