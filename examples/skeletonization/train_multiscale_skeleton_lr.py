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


if __name__ == "__main__":
    batch_size = 2
    patch_shape = (120, 120, 120)
    patch_stride = (120, 120, 120)
    patch_threshold = 0.01
    lr = 0.002
    logdir_path = "./checkpoints_lr_20231123"
    skeletonization_target = "skeletonization_target"
    train_data_pattern = (
        "/local1/kevin/code/morphospaces/examples/"
        "skeletonization/train_multiscale/train/*.h5"
    )
    val_data_pattern = (
        "/local1/kevin/code/morphospaces/examples/"
        "skeletonization/train_multiscale/val/*.h5"
    )
    # train_data_pattern = "./test_multiscale/*.h5"
    # val_data_pattern = "./test_multiscale/*.h5"
    log_every_n_iterations = 25
    val_check_interval = 0.25
    lr_reduction_patience = 25
    lr_scheduler_step = 1000
    accumulate_grad_batches = 4

    pl.seed_everything(42, workers=True)

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
        devices=[0],
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
