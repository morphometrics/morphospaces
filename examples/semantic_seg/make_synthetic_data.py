import os

import h5py
import napari
import numpy as np


def generate_random_cubes(
    image_shape=(100, 100, 100),
    num_cubes=10,
    seed=None,
    background_value=0.75,
    foreground_value=0.25,
):
    """
    Generate a 3D integer labels image with randomly placed axis-aligned cubes
    and a corresponding float image.

    Parameters
    ----------
    image_shape : tuple of int, optional
        Shape of the output image (depth, height, width)
    num_cubes : int, optional
        Number of cubes to generate
    seed : int, optional
        Random seed for reproducibility
    background_value : float, optional
        Value to use for background (where labels=0) in the float image
    foreground_value : float, optional
        Value to use for foreground (where labels=1) in the float image

    Returns
    -------
    labels : ndarray
        3D integer image with randomly placed cubes (1) on a background (0)
    image : ndarray
        3D float image where labels=0 has background_value and labels=1
        has foreground_value
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize empty integer image for labels
    labels = np.zeros(image_shape, dtype=np.int32)

    # Get dimensions
    depth, height, width = image_shape

    # Generate random cubes
    for _ in range(num_cubes):
        # Random cube size (between 3 and 20% of each dimension)
        max_size = np.array([depth, height, width]) // 5
        min_size = np.array([3, 3, 3])
        cube_size = np.random.randint(min_size, max_size + 1)

        # Random position ensuring cube fits within image bounds
        start_pos = np.random.randint(
            low=[0, 0, 0],
            high=[
                depth - cube_size[0] + 1,
                height - cube_size[1] + 1,
                width - cube_size[2] + 1,
            ],
        )

        # Calculate end positions
        end_pos = start_pos + cube_size

        # Draw the cube using indexing
        labels[
            start_pos[0] : end_pos[0],  # noqa
            start_pos[1] : end_pos[1],  # noqa
            start_pos[2] : end_pos[2],  # noqa
        ] = 1

    # Create the float image using the labels as a mask
    image = np.full(image_shape, background_value, dtype=np.float32)
    image[labels == 1] = foreground_value

    return labels, image


def generate_image_series(
    image_shape=(100, 100, 100),
    cube_counts=[5, 10, 15, 20],
    base_seed=42,
    output_dir=".",
    chunk_size=None,
    background_value=0.75,
    foreground_value=0.25,
):
    """
    Generate 3D images with different numbers of cubes and save to HDF5 files.

    Parameters
    ----------
    image_shape : tuple of int, optional
        Shape of each output image (depth, height, width)
    cube_counts : list of int, optional
        List of different cube counts to generate
    base_seed : int, optional
        Base random seed for reproducibility
    output_dir : str, optional
        Directory to save the HDF5 files
    chunk_size : tuple of int, optional
        Size of chunks for HDF5 compression. If None, a default is chosen.
    background_value : float, optional
        Value to use for background (where labels=0) in the float image
    foreground_value : float, optional
        Value to use for foreground (where labels=1) in the float image

    Returns
    -------
    file_paths : list of str
        List of paths to the saved HDF5 files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set default chunk size if not provided
    if chunk_size is None:
        # Default chunk size of 16×16×16
        chunk_size = (
            min(16, image_shape[0]),
            min(16, image_shape[1]),
            min(16, image_shape[2]),
        )

    file_paths = []

    for i, count in enumerate(cube_counts):
        # Use a different but deterministic seed for each image
        seed = base_seed + i if base_seed is not None else None

        # Generate the labels and image
        labels, image = generate_random_cubes(
            image_shape, count, seed, background_value, foreground_value
        )

        # Create file path
        file_path = os.path.join(output_dir, f"cube_image_{count}.h5")
        file_paths.append(file_path)

        # Save image to HDF5 file with gzip compression
        with h5py.File(file_path, "w") as f:
            # Save labels
            labels_dset = f.create_dataset(
                "labels",
                data=labels,
                chunks=chunk_size,
                compression="gzip",
                compression_opts=9,  # Maximum compression level
            )

            # Save float image
            image_dset = f.create_dataset(
                "image",
                data=image,
                chunks=chunk_size,
                compression="gzip",
                compression_opts=9,  # Maximum compression level
            )

            # Add metadata attributes
            labels_dset.attrs["num_cubes"] = count
            labels_dset.attrs["shape"] = image_shape
            labels_dset.attrs["seed"] = seed if seed is not None else -1

            image_dset.attrs["num_cubes"] = count
            image_dset.attrs["shape"] = image_shape
            image_dset.attrs["seed"] = seed if seed is not None else -1
            image_dset.attrs["background_value"] = background_value
            image_dset.attrs["foreground_value"] = foreground_value

        print(f"Saved data with {count} cubes to {file_path}")

    return file_paths


def view_example_in_napari(file_path):
    """
    Open an example image and labels pair in napari viewer.
    If a file path is provided, loads the data from the file.
    Otherwise, generates a new example.

    Parameters
    ----------
    file_path : str, optional
        Path to an HDF5 file containing 'image' and 'labels' datasets
    image_shape : tuple of int, optional
        Shape of the image if generating a new one
    num_cubes : int, optional
        Number of cubes if generating a new image
    seed : int, optional
        Random seed if generating a new image
    background_value : float, optional
        Background value if generating a new image
    foreground_value : float, optional
        Foreground value if generating a new image
    """
    # Either load from file
    print(f"Loading data from {file_path}")
    with h5py.File(file_path, "r") as f:
        image = f["image"][:]
        labels = f["labels"][:]
        # Print some metadata if available
        if "num_cubes" in f["image"].attrs:
            print(f"Number of cubes: {f['image'].attrs['num_cubes']}")
        if "background_value" in f["image"].attrs:
            print(f"Background value: {f['image'].attrs['background_value']}")
        if "foreground_value" in f["image"].attrs:
            print(f"Foreground value: {f['image'].attrs['foreground_value']}")

    # Create napari viewer
    viewer = napari.Viewer()

    # Add image layer
    viewer.add_image(
        image, name="Image", colormap="gray", contrast_limits=(0, 1)
    )

    # Add labels layer
    viewer.add_labels(
        labels,
        name="Labels",
    )

    # Set camera settings for better view
    viewer.dims.ndisplay = 3  # Switch to 3D view

    # Return the viewer for interactive use
    return viewer


# Example usage
if __name__ == "__main__":
    # Define parameters
    output_dir = "./cube_data"
    os.makedirs(output_dir, exist_ok=True)
    image_shape = (100, 100, 100)  # Smaller size for quick visualization
    cube_counts = [20, 30, 50]

    # Generate some example data
    file_paths = generate_image_series(
        image_shape=image_shape, cube_counts=cube_counts, output_dir=output_dir
    )

    # View the first generated file
    if file_paths:
        print(f"\nOpening {file_paths[0]} in napari...")
        viewer = view_example_in_napari(file_paths[0])
        napari.run()  # Start the Qt event loop
