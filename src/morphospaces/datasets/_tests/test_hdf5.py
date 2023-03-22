import h5py
import numpy as np

from morphospaces.datasets import LazyHDF5Dataset, StandardHDF5Dataset


def _make_hdf5_dataset(tmp_path) -> str:
    file_path = tmp_path / "test.h5"

    image = np.zeros((20, 20, 20))
    label = np.zeros((20, 20, 20), dtype=int)
    label[0:4, 0:4, 0:4] = 1

    with h5py.File(file_path, "w") as f:
        f.create_dataset("raw", data=image, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
    return file_path


def test_standard_hdf5_dataset(tmp_path):
    patch_shape = (4, 4, 4)
    dataset_path = _make_hdf5_dataset(tmp_path)
    ds = StandardHDF5Dataset(
        file_path=dataset_path,
        stage="train",
        transform=None,
        patch_shape=patch_shape,
        stride_shape=(2, 2, 2),
        patch_filter_ignore_index=(0,),
        patch_threshold=0,
        patch_slack_acceptance=0.01,
        mirror_padding=(16, 32, 32),
        raw_internal_path="raw",
        label_internal_path="label",
        weight_internal_path=None,
    )

    assert len(ds) == (9 * 9 * 9)
    assert ds[0]["raw"].shape == patch_shape
    assert ds[0]["label"].shape == patch_shape


def test_lazy_hdf5_dataset(tmp_path):
    patch_shape = (4, 4, 4)
    dataset_path = _make_hdf5_dataset(tmp_path)
    ds = LazyHDF5Dataset(
        file_path=dataset_path,
        stage="train",
        transform=None,
        patch_shape=patch_shape,
        stride_shape=(2, 2, 2),
        patch_filter_ignore_index=(0,),
        patch_threshold=0,
        patch_slack_acceptance=0.01,
        mirror_padding=(16, 32, 32),
        raw_internal_path="raw",
        label_internal_path="label",
        weight_internal_path=None,
    )

    assert len(ds) == (9 * 9 * 9)
    assert ds[0]["raw"].shape == patch_shape
    assert ds[0]["label"].shape == patch_shape
