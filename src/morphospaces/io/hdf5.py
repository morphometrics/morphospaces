from typing import Any, Dict

import h5py
import numpy as np


def _write_dataset_from_array(
    file_handle: h5py.File,
    dataset_name: str,
    dataset_array: np.ndarray,
    compression: str = "gzip",
):
    file_handle.create_dataset(
        dataset_name,
        dataset_array.shape,
        dtype=dataset_array.dtype,
        data=dataset_array,
        compression=compression,
    )


def _write_dataset_from_dict(
    file_handle: h5py.File,
    dataset_name: str,
    dataset: Dict[str, Any],
    compression: str = "gzip",
):
    dataset_array = dataset["data"]
    dset = file_handle.create_dataset(
        dataset_name,
        dataset_array.shape,
        dtype=dataset_array.dtype,
        data=dataset_array,
        compression=compression,
    )

    dataset_attrs = dataset.get("attrs", None)
    if dataset_attrs is not None:
        for k, v in dataset_attrs.items():
            dset.attrs[k] = v


def write_multi_dataset_hdf(
    file_path: str, compression: str = "gzip", **kwargs
):
    """Write a multidataset hdf5 file.

    Parameters
    ----------
    file_path : str
        Path to write the file to.
    compression : str
        Default value is "gzip"
    **kwargs
        Datasets are passed as keyword arguments.
        When passing an array, the dataset name is the
        keyword argument name and the data is the keyword
        argument value. For example,
            my_data=my_array
        would yield a dataset named "my_data" containing "my_array".
    """
    if len(kwargs) == 0:
        raise ValueError("Must supply at least one dataset as a kwarg")
    with h5py.File(file_path, "w") as f:
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                _write_dataset_from_array(
                    file_handle=f,
                    dataset_name=k,
                    dataset_array=v,
                    compression=compression,
                )
            else:
                _write_dataset_from_dict(
                    file_handle=f,
                    dataset_name=k,
                    dataset=v,
                    compression=compression,
                )
