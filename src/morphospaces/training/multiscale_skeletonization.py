import logging
import sys
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.datasets import LazyHDF5Dataset, StandardHDF5Dataset
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)
from morphospaces.transforms.image import ExpandDimsd
from morphospaces.transforms.label import LabelsAsFloat32
from morphospaces.transforms.skeleton import DownscaleSkeletonGroundTruth

logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# use mixed precision for Tensor cores (speedup but less precise)
# torch.set_float32_matmul_precision("medium")


def train(
    train_data_pattern: str,
    val_data_pattern: str,
    logdir_path: Path,
    batch_size: int = 2,
    patch_shape: Tuple[int, int, int] = (120, 120, 120),
    patch_stride: Tuple[int, int, int] = (120, 120, 120),
    patch_threshold: float = 0.01,
    lr: float = 0.0004,
    skeletonization_target: str = "skeletonization_target",
    log_every_n_iterations: int = 25,
    val_check_interval: float = 0.25,
    lr_reduction_patience: int = 25,
    lr_scheduler_step: int = 1000,
    accumulate_grad_batches: int = 4,
    seed_value: int = 42,
    n_dataset_workers: int = 4,
    gpus: Tuple[int, ...] = (0,),
):
    """Train the multiscale skeletonization network.

    Parameters
    ----------
    train_data_pattern : str
        The glob pattern to use to find the training data.
        Generally, this will be something like *.h5.
    val_data_pattern : str
        The glob pattern to use to find the validation data.
        Generally, this will be something like *.h5.
    logdir_path : Path
        The directory to save the logs and checkpoints.
    batch_size : int
        The number of patches to include in each batch.
    patch_shape : Tuple[int, int, int]
        The shape of the patches to extract from the data.
        The default value is (120, 120, 120).
    patch_stride : Tuple[int, int, int]
        The stride to use when extracting patches.
        The default value is (120, 120, 120).
    patch_threshold : float
        The minimum fraction of voxels in the patch that must
        include a positive label to be included in training.
        The default value is 0.01.
    lr : float
        The learning rate for the optimizer. The default value is 0.0004.
    skeletonization_target : str
        The key in the HDF5 file to use for the skeletonization ground truth.
        The default value is "skeletonization_target".
    log_every_n_iterations : int
        The number of iterations between logging. The default value is 25.
        See the pytorch lightning docs for details.
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    val_check_interval : float
        The interval for validation in fractions of an epoch.
        The default value is 0.25. See the pytorch lightning docs for details.
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    lr_reduction_patience : int
        The number of epochs to wait before reducing the learning rate.
        The default value is 25. See the pytorch docs for details.
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    lr_scheduler_step : int
        The number of iterations between learning rate scheduler updates.
        The default value is 1000.
    accumulate_grad_batches : int
        The number of batches to accumulate before performing a backward pass.
        The default value is 4. See the pytorch lightnight docs for details.
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    seed_value : int
        The value to use for seeding the RNGs. The default value is 42.
    n_dataset_workers : int
        The number of workers to use for each dataset loader
        (i.e., train and validation).
    gpus : Tuple[int, ...]
        The indices of the GPUs to use for training. The default value is (0,).
    """
    # seed RNGs
    pl.seed_everything(seed_value, workers=True)

    # setup the training transforms/augmentations
    train_transform = Compose(
        [
            LabelsAsFloat32(keys="label_image"),
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ]
            ),
            RandScaleIntensityd(
                keys="normalized_vector_background_image",
                factors=(-0.5, 0),
                prob=0.25,
            ),
            RandFlipd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.2,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.2,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.2,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.25,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.25,
                spatial_axes=(0, 2),
            ),
            RandRotate90d(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.25,
                spatial_axes=(1, 2),
            ),
            RandAffined(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.5,
                mode="nearest",
                rotate_range=(1.5, 1.5, 1.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
            DownscaleSkeletonGroundTruth(
                label_key="label_image",
                skeletonization_target_key=skeletonization_target,
                downscaling_factors=[2, 4],
                gaussian_sigma=1,
                normalization_neighborhood_sizes=5,
            ),
        ]
    )

    # make the training dataset
    train_ds = LazyHDF5Dataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        dataset_keys=[
            "normalized_vector_background_image",
            "label_image",
            "skeletonization_target",
        ],
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key="label_image",
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_dataset_workers,
    )

    # make the validation transforms
    val_transform = Compose(
        [
            LabelsAsFloat32(keys="label_image"),
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ]
            ),
            DownscaleSkeletonGroundTruth(
                label_key="label_image",
                skeletonization_target_key=skeletonization_target,
                downscaling_factors=[2, 4],
                gaussian_sigma=1,
                normalization_neighborhood_sizes=5,
            ),
        ]
    )

    val_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=val_data_pattern,
        dataset_keys=[
            "normalized_vector_background_image",
            "label_image",
            "skeletonization_target",
        ],
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key="label_image",
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_dataset_workers,
    )

    # make the checkpoint callback
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="skel-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="skel-last",
    )

    # learning rate monitor
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    net = MultiscaleSkeletonizationNet(
        in_channels=1,
        image_key="normalized_vector_background_image",
        labels_key="label_image",
        skeletonization_target_key="skeletonization_target",
        learning_rate=lr,
        lr_reduction_patience=lr_reduction_patience,
        lr_scheduler_step=lr_scheduler_step,
    )

    # logger
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=list(gpus),
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback,
            learning_rate_monitor,
        ],
        logger=logger,
        max_epochs=2000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
