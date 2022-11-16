import numpy as np
from numpy.random import default_rng


def random_indices_in_bounding_box(
    n_points: int,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
    random_seed: float = 42
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
        size=(n_points, len(lower_left_corner))
    )
