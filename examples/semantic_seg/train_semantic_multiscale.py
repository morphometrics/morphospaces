import logging
import sys

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
from morphospaces.networks import MultiscaleSemanticSegmentationNet
from morphospaces.transforms.image import ExpandDimsd
from morphospaces.transforms.label import DownscaleLabelsd, LabelsAsLong

logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == "__main__":
    batch_size = 2
    patch_shape = (64, 64, 64)
    patch_stride = (32, 32, 32)
    patch_threshold = 0
    lr = 0.0004
    logdir_path = "./checkpoints_semantic"
    image_key = "image"
    labels_key = "labels"
    train_data_pattern = "./cube_data/*.h5"
    val_data_pattern = "./cube_data/*.h5"

    # frequency of logging (training steps)
    log_every_n_iterations = 5

    # perform validation once every epoch
    val_check_interval = 1.0

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            RandScaleIntensityd(
                keys=image_key,
                factors=(-0.5, 0),
                prob=0.2,
            ),
            RandFlipd(
                keys=[image_key, labels_key],
                prob=0.2,
            ),
            RandRotate90d(
                keys=[image_key, labels_key],
                prob=0.1,
                spatial_axes=(0, 1),
            ),
            RandAffined(
                keys=[image_key, labels_key],
                prob=0.2,
                mode="nearest",
                rotate_range=(0.5, 0.5, 0.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
            DownscaleLabelsd(
                label_key=labels_key,
                downscaling_factors=[2, 4],
            ),
            ExpandDimsd(
                keys=[
                    image_key,
                ]
            ),
        ]
    )

    train_ds = LazyHDF5Dataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        dataset_keys=[image_key, labels_key],
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            DownscaleLabelsd(
                label_key=labels_key,
                downscaling_factors=[2, 4],
            ),
            ExpandDimsd(
                keys=[
                    image_key,
                ]
            ),
        ]
    )

    val_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=val_data_pattern,
        dataset_keys=[
            image_key,
            labels_key,
        ],
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # make the checkpoint callback
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="seg-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="seg-last",
    )

    # learning rate monitor
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    net = MultiscaleSemanticSegmentationNet(
        in_channels=1,
        image_key=image_key,
        labels_key=labels_key,
        learning_rate=lr,
    )

    # logger
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[
            best_checkpoint_callback,
            last_checkpoint_callback,
            learning_rate_monitor,
        ],
        logger=logger,
        max_epochs=2000,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
