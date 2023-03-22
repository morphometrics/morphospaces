import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import Compose, DivisiblePadd, SpatialPadd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.datasets import StandardHDF5Dataset
from morphospaces.networks.unet import SemanticSegmentationUnet
from morphospaces.transforms.image import ExpandDimsd, ImageAsFloat32
from morphospaces.transforms.label import LabelToBoundaryd

if __name__ == "__main__":
    batch_size = 1
    patch_shape = (80, 96, 96)
    lr = 0.0002
    logdir_path = "./checkpoints"

    # set the RNG seeds
    pl.seed_everything(42, workers=True)

    # train_ds = StandardHDF5Dataset.from_glob_pattern(
    #     glob_pattern="./train/*.h5",
    #     stage="train",
    #     transform=LabelToBoundaryd(label_key="label"),
    #     patch_shape=patch_shape,
    #     stride_shape=(20, 40, 40),
    #     patch_filter_ignore_index=(0,),
    #     patch_threshold=0.6,
    #     patch_slack_acceptance=0.01,
    #     mirror_padding=(16, 32, 32),
    #     raw_internal_path="raw",
    #     label_internal_path="label",
    #     weight_internal_path=None,
    # )
    # train_loader = DataLoader(
    #     train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    # )

    val_transform = Compose(
        [
            ImageAsFloat32(keys="raw"),
            ExpandDimsd(keys=["raw", "label"]),
            LabelToBoundaryd(label_key="label"),
            SpatialPadd(keys=["raw", "label"], spatial_size=[96, 96, 96]),
            DivisiblePadd(keys=["raw", "label"], k=16),
        ]
    )

    val_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern="./val/*.h5",
        stage="val",
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_shape,
        patch_filter_ignore_index=(0,),
        patch_threshold=0.6,
        patch_slack_acceptance=0.01,
        mirror_padding=(16, 32, 32),
        raw_internal_path="raw",
        label_internal_path="label",
        weight_internal_path=None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(val_ds[0]["raw"].shape)

    # make the model
    unet = SemanticSegmentationUnet(
        image_key="raw",
        label_key="label",
        learning_rate=lr,
        roi_size=patch_shape,
    )

    # make the checkpoint callback
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=2,
        filename="vit-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=2,
        filename="vit-last",
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
    )
    trainer.fit(
        unet,
        train_dataloaders=val_loader,
        val_dataloaders=val_loader,
    )
