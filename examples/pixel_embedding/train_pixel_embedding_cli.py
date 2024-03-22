"""Script to train the pixel embedding model.

The script takes the following command line arguments:
- lr: the learning rate for the model.
- n_layers: the number of layers in the UNet.

Example execution:

python train_pixel_embedding_cli.py 0.0001 4

would train with a learning rate of 0001 and 4 layers in the UNet.
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
from morphospaces.networks.pixel_embedding import PixelEmbedding
from morphospaces.transforms.image import ExpandDimsd, StandardizeImage
from morphospaces.transforms.label import LabelsAsFloat32

# setup logging
logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser("train the pixel embedding model.")
    parser.add_argument("lr", help="Learning rate.", type=float)
    parser.add_argument(
        "n_layers", help="number of layers in the UNet", type=int
    )

    return parser.parse_args()


if __name__ == "__main__":
    # CLI arguments
    args = parse_args()
    lr = args.lr
    n_layers = args.n_layers

    # patch parameters
    batch_size = 1
    patch_shape = (64, 64, 64)
    patch_stride = (64, 64, 64)
    patch_threshold = 0.01

    loss_temperature = 0.1
    train_data_pattern = "./train/*.h5"
    val_data_pattern = "./val/*.h5"
    # train_data_pattern = (
    #     "./data/test/*.h5"
    # )
    # val_data_pattern = (
    #     "./data/test/*.h5"
    # )

    image_key = "raw"
    labels_key = "label"

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = (
        f"./checkpoints_nce_memory_{n_layers}_{learning_rate_string}_20240308"
    )

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.5
    lr_reduction_patience = 25
    lr_scheduler_step = 5000
    # lr_scheduler_step = 1000
    accumulate_grad_batches = 4
    memory_banks: bool = True
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
        store_unique_label_values=memory_banks,
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
    net = PixelEmbedding(
        image_key=image_key,
        labels_key=labels_key,
        in_channels=1,
        n_layers=n_layers,
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
        val_check_interval=val_check_interval,
        # check_val_every_n_epoch=3
    )
    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
