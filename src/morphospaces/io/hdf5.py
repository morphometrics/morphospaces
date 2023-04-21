from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch


def _write_dataset_from_array(
    file_handle: h5py.File,
    dataset_name: str,
    dataset_array: np.ndarray,
    compression: str = "gzip",
    chunks: Optional[Tuple[int, ...]] = None,
):
    dataset_kwargs = {
        "name": dataset_name,
        "data": dataset_array,
        "compression": compression,
    }
    if chunks is not None:
        dataset_kwargs.update({"chunks": chunks})

    file_handle.create_dataset(**dataset_kwargs)


def _write_dataset_from_dict(
    file_handle: h5py.File,
    dataset_name: str,
    dataset: Dict[str, Any],
    compression: str = "gzip",
    chunks: Optional[Tuple[int, ...]] = None,
):
    dataset_kwargs = {
        "name": dataset_name,
        "data": dataset["data"],
        "compression": compression,
    }
    if chunks is not None:
        dataset_kwargs.update({"chunks": chunks})

    if "chunks" in dataset:
        dataset_kwargs.update({"chunks": dataset["chunks"]})

    dset = file_handle.create_dataset(**dataset_kwargs)

    dataset_attrs = dataset.get("attrs", None)
    if dataset_attrs is not None:
        for k, v in dataset_attrs.items():
            dset.attrs[k] = v


def write_multi_dataset_hdf(
    file_path: str,
    compression: str = "gzip",
    chunks: Optional[Tuple[int, ...]] = None,
    **kwargs
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
                    chunks=chunks,
                )
            elif isinstance(v, torch.Tensor):
                _write_dataset_from_array(
                    file_handle=f,
                    dataset_name=k,
                    dataset_array=v.cpu().numpy(),
                    compression=compression,
                    chunks=chunks,
                )
            elif isinstance(v, torch.Tensor):
                _write_dataset_from_array(
                    file_handle=f,
                    dataset_name=k,
                    dataset_array=v.cpu().numpy(),
                    compression=compression,
                )
            else:
                _write_dataset_from_dict(
                    file_handle=f,
                    dataset_name=k,
                    dataset=v,
                    compression=compression,
                    chunks=chunks,
                )
