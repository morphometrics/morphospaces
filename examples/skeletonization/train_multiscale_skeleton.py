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
from pytorch_lightning.callbacks import ModelCheckpoint
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


if __name__ == "__main__":
    batch_size = 2
    patch_shape = (120, 120, 120)
    patch_stride = (120, 120, 120)
    patch_threshold = 0.01
    lr = 0.0004
    logdir_path = "./checkpoints"
    skeletonization_target = "skeletonization_target"
    train_data_pattern = "./data/*.h5"
    val_data_pattern = "./data_val/*.h5"
    # train_data_pattern = "./test_multiscale/*.h5"
    # val_data_pattern = "./test_multiscale/*.h5"
    log_every_n_iterations = 100
    val_check_interval = 0.1

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsFloat32(keys="label_image"),
            RandScaleIntensityd(
                keys="normalized_vector_background_image",
                factors=(-0.5, 0),
                prob=0.2,
            ),
            RandFlipd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.2,
            ),
            RandRotate90d(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.1,
                spatial_axes=(0, 1),
            ),
            RandAffined(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                ],
                prob=0.2,
                mode="nearest",
                rotate_range=(0.5, 0.5, 0.5),
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
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                    "skeletonization_target_reduced_2",
                    "label_image_reduced_2",
                    "skeletonization_target_reduced_4",
                    "label_image_reduced_4",
                ]
            ),
        ]
    )

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
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            LabelsAsFloat32(keys="label_image"),
            DownscaleSkeletonGroundTruth(
                label_key="label_image",
                skeletonization_target_key=skeletonization_target,
                downscaling_factors=[2, 4],
                gaussian_sigma=1,
                normalization_neighborhood_sizes=5,
            ),
            ExpandDimsd(
                keys=[
                    "normalized_vector_background_image",
                    "skeletonization_target",
                    "label_image",
                    "skeletonization_target_reduced_2",
                    "label_image_reduced_2",
                    "skeletonization_target_reduced_4",
                    "label_image_reduced_4",
                ]
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

    net = MultiscaleSkeletonizationNet(
        in_channels=1,
        image_key="normalized_vector_background_image",
        labels_key="label_image",
        skeletonization_target_key="skeletonization_target",
        learning_rate=lr,
    )

    # logger
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
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
