import glob
import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def copy_files(file_list, output_directory):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    for file in tqdm(file_list):
        shutil.copy2(file, output_directory)


all_datasets = glob.glob(
    "../single_branch_point_skeleton_datasets_20230322/*.h5"
)
train, val = train_test_split(all_datasets, test_size=0.1, random_state=0)

copy_files(train, "./train")
copy_files(val, "./val")
