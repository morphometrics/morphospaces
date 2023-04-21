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

bumpy_datasets = glob.glob("../randomized_bumpy_20230325/*.h5")
train_bumpy, val_bumpy = train_test_split(
    bumpy_datasets, test_size=0.1, random_state=0
)

copy_files(train_bumpy, "./train_bumpy")
copy_files(val_bumpy, "./val_bumpy")

short_datasets = glob.glob("../randomized_short_20230326/*.h5")
train_short, val_short = train_test_split(
    short_datasets, test_size=0.1, random_state=0
)

copy_files(train_short, "./train_bumpy")
copy_files(val_short, "./val_bumpy")
