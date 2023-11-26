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
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSemanticSkeletonizationNet,
)
from morphospaces.transforms.image import ExpandDimsd
from morphospaces.transforms.label import DownscaleLabelsd, LabelsAsLong

logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == "__main__":
    batch_size = 2
    patch_shape = (120, 120, 120)
    patch_stride = (120, 120, 120)
    patch_threshold = 0.01
    lr = 0.0004
    logdir_path = "./checkpoints_semantic"
    # train_data_pattern = (
    #     "/local1/kevin/code/morphospaces/examples/"
    #     "skeletonization/train_multiscale/train/*.h5"
    # )
    # val_data_pattern = (
    #     "/local1/kevin/code/morphospaces/examples/"
    #     "skeletonization/train_multiscale/val/*.h5"
    # )
    train_data_pattern = "./test_multiscale/*.h5"
    val_data_pattern = "./test_multiscale/*.h5"
    log_every_n_iterations = 5
    val_check_interval = 1.0

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsLong(keys="label_image"),
            RandScaleIntensityd(
                keys="normalized_vector_background_image",
                factors=(-0.5, 0),
                prob=0.2,
            ),
            RandFlipd(
                keys=[
                    "normalized_vector_background_image",
                    "label_image",
                ],
                prob=0.2,
            ),
            RandRotate90d(
                keys=[
                    "normalized_vector_background_image",
                    "label_image",
                ],
                prob=0.1,
                spatial_axes=(0, 1),
            ),
            RandAffined(
                keys=[
                    "normalized_vector_background_image",
                    "label_image",
                ],
                prob=0.2,
                mode="nearest",
                rotate_range=(0.5, 0.5, 0.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
            DownscaleLabelsd(
                label_key="label_image",
                downscaling_factors=[2, 4],
            ),
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                ]
            ),
        ]
    )

    train_ds = LazyHDF5Dataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        dataset_keys=[
            "normalized_vector_background_image",
            "label_image",
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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            LabelsAsLong(keys="label_image"),
            DownscaleLabelsd(
                label_key="label_image",
                downscaling_factors=[2, 4],
            ),
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                ]
            ),
        ]
    )

    val_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=val_data_pattern,
        dataset_keys=[
            "normalized_vector_background_image",
            "label_image",
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
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
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

    net = MultiscaleSemanticSkeletonizationNet(
        in_channels=1,
        image_key="normalized_vector_background_image",
        labels_key="label_image",
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
