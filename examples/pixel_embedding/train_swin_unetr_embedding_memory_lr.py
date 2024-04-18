"""Sample training script for the train SwinUnetr pixel embedding network.

This is a work in process and the hyper parameters/augmentations haven't
been tuned.

The memory_banks flag turns on/off the memory banks and contrastive loss.
    - if set to False, training is only on the sementic segmentation loss.
    - if set to True, traning is on the semantic segmentation
      + contrastive loss
"""

import argparse
import logging
import sys

import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import Compose, RandAffined, RandFlipd, RandRotate90d
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.datasets import StandardHDF5Dataset
from morphospaces.networks.swin_unetr import PixelEmbeddingSwinUNETR
from morphospaces.transforms.image import ExpandDimsd, StandardizeImage
from morphospaces.transforms.label import LabelsAsFloat32

# setup logging
logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser("train the pixel embedding model.")
    parser.add_argument("lr", help="Learning rate.", type=float)
    return parser.parse_args()


if __name__ == "__main__":

    # CLI arguments
    args = parse_args()
    lr = args.lr
    # lr = 0.0008

    # patch parameters
    batch_size = 1
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.1

    loss_temperature = 0.1
    # train_data_pattern = (
    #     "./train/*.h5"
    # )
    # val_data_pattern = (
    #     "./val/*.h5"
    # )
    train_data_pattern = "./test_simulated/*.h5"
    val_data_pattern = "./test_simulated/*.h5"

    image_key = "raw"
    labels_key = "label"

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = f"./checkpoints_swin_{learning_rate_string}_memory_20240319"

    # pretrained weights
    pretrained_weights_path = "model_swinvit.pt"

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.15
    lr_reduction_patience = 25
    lr_scheduler_step = 1500
    # lr_scheduler_step = 1000
    accumulate_grad_batches = 4
    memory_banks: bool = False
    n_pixel_embeddings_per_class: int = 1000
    n_pixel_embeddings_to_update: int = 10
    n_label_embeddings_per_class: int = 50
    n_memory_warmup: int = 1000

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 2),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(1, 2),
            ),
            RandAffined(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.5,
                mode="nearest",
                rotate_range=(1.5, 1.5, 1.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
        ]
    )
    (
        train_ds,
        unique_train_label_values,
    ) = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        dataset_keys=[
            image_key,
            labels_key,
        ],
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
        store_unique_label_values=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # get the unique label values
    print(f"Unique train label values: {unique_train_label_values}")

    val_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            ),
        ]
    )
    val_ds, unique_val_label_values = StandardHDF5Dataset.from_glob_pattern(
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
        store_unique_label_values=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Unique val label values: {unique_val_label_values}")

    # get all unique label values
    unique_label_values = set(unique_train_label_values).union(
        set(unique_val_label_values)
    )
    print(f"Unique label values: {unique_label_values}")

    # make the checkpoint callback
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="pe-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="pe-last",
    )

    # learning rate monitor
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    # make the model
    net = PixelEmbeddingSwinUNETR(
        pretrained_weights_path=pretrained_weights_path,
        image_key=image_key,
        labels_key=labels_key,
        in_channels=1,
        n_embedding_dims=48,
        lr_scheduler_step=lr_scheduler_step,
        lr_reduction_patience=lr_reduction_patience,
        learning_rate=lr,
        loss_temperature=loss_temperature,
        n_samples_per_class=n_samples_per_class,
        label_values=unique_label_values,
        memory_banks=memory_banks,
        n_pixel_embeddings_per_class=n_pixel_embeddings_per_class,
        n_pixel_embeddings_to_update=n_pixel_embeddings_to_update,
        n_label_embeddings_per_class=n_label_embeddings_per_class,
        n_memory_warmup=n_memory_warmup,
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
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        # val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
