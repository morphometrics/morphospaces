from morphospaces.datasets.hdf5 import StandardHDF5Dataset

dataset_path = "test.h5"
ds = StandardHDF5Dataset(
    file_path=dataset_path,
    stage="train",
    transform=None,
    label_transform=None,
    weight_transform=None,
    patch_shape=(4, 4, 4),
    stride_shape=(2, 2, 2),
    patch_filter_ignore_index=(0,),
    patch_threshold=0,
    patch_slack_acceptance=0.01,
    mirror_padding=(16, 32, 32),
    raw_internal_path="raw",
    label_internal_path="label",
    weight_internal_path=None,
)

print(len(ds))
print(ds[0][0].shape)
print(type(ds[0][0]))
