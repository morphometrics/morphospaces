import os
import pickle

import pytorch_lightning as pl
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    OneOf,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from monai.utils import first
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.networks.constants import (
    VIT_AC_AUG_VIEW_1_KEY,
    VIT_AC_AUG_VIEW_2_KEY,
    VIT_AC_GT_KEY,
)
from morphospaces.networks.vit_autoencoder import VitAutoencoder

# Define Hyper-paramters for training loop
max_epochs = 500
val_interval = 2
batch_size = 4
lr = 1e-4


# path to the data
json_path = os.path.normpath("dataset_split.json")
data_root = os.path.normpath("/local1/kevin/ssl/tcia")
logdir_path = os.path.normpath("./ssl_checkpoints")


with open("train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("val_data.pkl", "rb") as f:
    val_data = pickle.load(f)

# Define Training Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=[VIT_AC_AUG_VIEW_1_KEY]),
        EnsureChannelFirstd(keys=[VIT_AC_AUG_VIEW_1_KEY]),
        Spacingd(
            keys=[VIT_AC_AUG_VIEW_1_KEY],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=[VIT_AC_AUG_VIEW_1_KEY],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=[VIT_AC_AUG_VIEW_1_KEY], source_key=VIT_AC_AUG_VIEW_1_KEY
        ),
        SpatialPadd(keys=[VIT_AC_AUG_VIEW_1_KEY], spatial_size=(96, 96, 96)),
        RandSpatialCropSamplesd(
            keys=[VIT_AC_AUG_VIEW_1_KEY],
            roi_size=(96, 96, 96),
            random_size=False,
            num_samples=2,
        ),
        CopyItemsd(
            keys=[VIT_AC_AUG_VIEW_1_KEY],
            times=2,
            names=[VIT_AC_GT_KEY, VIT_AC_AUG_VIEW_2_KEY],
            allow_missing_keys=False,
        ),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=[VIT_AC_AUG_VIEW_1_KEY],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=[VIT_AC_AUG_VIEW_1_KEY],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(
            keys=[VIT_AC_AUG_VIEW_1_KEY], prob=0.8, holes=10, spatial_size=8
        ),
        # Please note that that if image, image_2 are called
        # via the same transform call because of the determinism
        # they will get augmented the exact same way which
        # is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=[VIT_AC_AUG_VIEW_2_KEY],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=[VIT_AC_AUG_VIEW_2_KEY],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(
            keys=[VIT_AC_AUG_VIEW_2_KEY], prob=0.8, holes=10, spatial_size=8
        ),
    ]
)


check_ds = Dataset(data=train_data, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image = check_data[VIT_AC_AUG_VIEW_1_KEY][0][0]
print(f"image shape: {image.shape}")

if __name__ == "__main__":
    # set the RNG seeds
    pl.seed_everything(42, workers=True)

    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_ds = Dataset(data=val_data, transform=train_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # make the model
    autoencoder = VitAutoencoder(learning_rate=lr)

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
        devices=2,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
        logger=logger,
    )
    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
