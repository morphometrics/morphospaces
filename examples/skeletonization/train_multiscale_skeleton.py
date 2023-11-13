import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import Compose
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.datasets import StandardHDF5Dataset
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)
from morphospaces.transforms.image import ExpandDimsd, ImageAsFloat32
from morphospaces.transforms.skeleton import DownscaleSkeletonGroundTruth

if __name__ == "__main__":
    batch_size = 1
    patch_shape = (120, 120, 120)
    patch_stride = (120, 120, 120)
    patch_threshold = 0.0005
    lr = 0.0002
    logdir_path = "./checkpoints"
    skeletonization_target = "normalized_vector_background_image"

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            ImageAsFloat32(keys="normalized_vector_background_image"),
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

    train_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern="./test/*.h5",
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
        patch_slack_acceptance=0.01,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            ImageAsFloat32(keys="normalized_vector_background_image"),
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
        glob_pattern="./test/*.h5",
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
        patch_slack_acceptance=0.01,
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
    logger = TensorBoardLogger(
        save_dir=logdir_path, version=1, name="lightning_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=logger,
        max_epochs=2000,
    )
    trainer.fit(
        net,
        train_dataloaders=val_loader,
        val_dataloaders=val_loader,
    )
