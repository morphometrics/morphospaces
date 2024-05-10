import glob
import os
from typing import Tuple, Callable

import h5py
from monai.inferers import sliding_window_inference
import mrcfile
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR


def predict_mrc(
    input_file: str,
    output_directory: str,
    image_key: str,
    labels_key: str,
    prediction_function: Callable[[torch.Tensor], torch.Tensor],
    roi_size: Tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
    gpu_index: int = 0,
):
    os.makedirs(output_directory, exist_ok=True)

    # load the image
    image = mrcfile.read(input_file)

    image = torch.from_numpy(np.expand_dims(image, axis=(0, 1)).astype(np.float32))

    # make the prediction
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=image,
            sw_batch_size=batch_size,
            sw_device=torch.device("cuda", gpu_index),
            predictor=prediction_function,
            roi_size=roi_size,
            overlap=overlap,
            mode=stitching_mode,
            device=torch.device("cpu"),
            progress=progress_bar
        )

    # save the image
    input_file_base = os.path.basename(input_file)
    output_path = os.path.join(output_directory, input_file_base)
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset(
            name=image_key,
            data=np.squeeze(image),
            compression="gzip",
            chunks=(64, 64, 64)
        )
        f_out.create_dataset(
            name="embedding",
            data=np.squeeze(result),
            compression="gzip",
            chunks=(1, 64, 64, 64)
        )


if __name__ == "__main__":
    checkpoint_path = "pe-best-v1.ckpt"
    image_key = "raw"
    labels_key = "label"
    roi_size = (64, 64, 64)

    # load the network
    net = PixelEmbeddingSwinUNETR.load_from_checkpoint(checkpoint_path)

    def predict_embedding(patch: torch.Tensor) -> torch.Tensor:
        # standardize the patch
        patch = (patch - patch.mean()) / patch.std()
        return net(patch)

    predict_mrc(
        input_file="cropped_covid.mrc",
        output_directory="./embedding_seg",
        prediction_function=predict_embedding,
        roi_size=roi_size,
        image_key=image_key,
        labels_key=labels_key,
    )
