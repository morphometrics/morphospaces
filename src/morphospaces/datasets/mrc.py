import glob
import dask.array as da
import mrcfile
import numpy as np

from morphospaces.datasets._base import BaseTiledDataset


from typing import Dict, List, Tuple, Union
from numpy.typing import ArrayLike
from torch.utils.data import ConcatDataset



class MrcDataset(BaseTiledDataset):
    """
    Implementation of the mrc dataset that loads both image mrcfiles and their corresponding maskmrc files into numpy arrays, \
    constructing a map-style dataset, such as {'mrc_tomogram': Array([...], dtype=np.float32), 'mrc_mask': Array([...], dtype=np.float32)}.
    """

    def __init__(
        self,
        mrcfile_path: dict, # {'mrc_tomogram': path_to_raw_data, 'mrc_mask': path_to_label_mask}
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (
            96,
            96,
            96,
        ),
        stride_shape: Union[
            Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "mrc_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        
        dataset_keys = mrcfile_path.keys()
        self.mrcfile_path = mrcfile_path
        
        super().__init__(
            dataset_keys=dataset_keys, # List['mrc_tomogram', 'mrc_mask']
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_filter_key=patch_filter_key,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )

        self.data: Dict[str, ArrayLike] = {
            key: self.get_array(self.mrcfile_path[key], key=key) for key in dataset_keys
        }
        self._init_states()

    
    @staticmethod
    def get_array(file_path, key):
        if key == 'mrc_mask' or 'mrc_tomo':
            ds = mrcfile.read(file_path)
            return ds.astype(np.float32)  # both data and label are float32


    @classmethod
    def from_glob_pattern(
        cls,
        glob_pattern: dict, # {'mrc_tomogram': path_to_raw_data/*.mrc, 'mrc_mask': path_to_label/*.mrc}
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (
            96,
            96,
            96,
        ),
        stride_shape: Union[
            Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "mrc_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        
        n = len(glob.glob(list(glob_pattern.values())[0]))
        mrcfile_paths = [dict()] * n # List[dict]        
        for key, path in glob_pattern.items():
            file_paths = glob.glob(path)
            assert len(file_paths) == n, 'The number of tomogram mrc files is not the same as the number of label mask files'
            for i,path in enumerate(file_paths):
                mrcfile_paths[i][key] = path

        datasets = []
        for path in mrcfile_paths:
            # make a dataset for each file that matches the pattern
            try:
                dataset = cls(
                    mrcfile_path=path,
                    transform=transform,
                    patch_shape=patch_shape,
                    stride_shape=stride_shape,
                    patch_filter_ignore_index=patch_filter_ignore_index,
                    patch_filter_key=patch_filter_key,
                    patch_threshold=patch_threshold,
                    patch_slack_acceptance=patch_slack_acceptance,
                    store_unique_label_values=store_unique_label_values,
                )
                datasets.append(dataset)
            except AssertionError:
                logger.info(f"{path} skipped")

        if store_unique_label_values:
            unique_label_values = set()
            for dataset in datasets:
                # get the unique label values across all datasets
                unique_label_values.update(dataset.unique_label_values)

            return ConcatDataset(datasets), list(unique_label_values)

        return ConcatDataset(datasets)