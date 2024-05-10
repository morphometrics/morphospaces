from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.random import randint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.losses.ssl import SSLLoss
from morphospaces.networks._components.ssl_vit import SSLVit


class SSLViT(pl.LightningModule):
    def __init__(
        self,
        image_key: str = "raw",
        in_channels: int = 1,
        n_embedding_dims: int = 48,
        n_spatial_dims: int = 3,
        dropout_path_rate: float = 0,
        use_checkpoint: bool = False,
        dim: int = 768,
        total_batch_size: int = 4,
    ):
        """Vision transformer for self-supervised pretraining.

        This is generally used to pretrain the encoder for the
        Swin-UNETR models.

        Parameters
        ----------
        in_channels : int
           The number of input channels.
        n_embedding_dims : int
            The dimensionality of the feature maps. Default value is 48.
        n_spatial_dims : int
            The number of spatial dims for the input image. Default value is 3.
        dropout_path_rate : float
            Stochastic depth rate. Defaults value is 0.0.
        use_checkpoint : bool
            Use gradient checkpointing to reduce memory usage.
            Default value is False
        dim : int
            The dimensionality of the encoder output.
            Default value is 768.
        total_batch_size : int
            The total batch size including both the size of the input from
            the data loader and the number of views provided by
            the augmentation. Should be data_loader_batch_size * n_views.
        """
        super().__init__()
        self._model = SSLVit(
            in_channels=in_channels,
            feature_size=n_embedding_dims,
            n_spatial_dims=n_spatial_dims,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            dim=dim,
        )

        self.loss = SSLLoss(batch_size=self.hparams.total_batch_size)

        # parameters to track training
        self.iteration_count = 0
        self.validation_step_outputs = []

    def configure_optimizers(self):
        """Set up the Adam optimzier.
        See the pytorch-lightning module documentation for details.
        """
        optimizer = torch.optim.Adam(
            self._model.parameters(), self.hparams.learning_rate
        )
        learning_rate_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=self.hparams.lr_reduction_factor,
            patience=self.hparams.lr_reduction_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": learning_rate_scheduler,
                "interval": "step",
                "frequency": self.hparams.lr_scheduler_step,
                "monitor": "val_loss",
            },
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Implementation of one training step.
        See the pytorch-lightning module documentation for details.
        """
        images = batch[self.hparams.image_key]

        x1, rot1 = rot_rand(images)
        x2, rot2 = rot_rand(images)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)
        x1_augment = x1_augment
        x2_augment = x2_augment

        rot1_p, contrastive1_p, rec_x1 = self._model(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self._model(x2_augment)

        # compute the loss
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = self.loss(
            rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs
        )

        # do the loggine
        rotation_loss, contrast_loss, reconstruction_loss = losses_tasks
        self.log("training_loss", loss, batch_size=len(images), prog_bar=True)
        self.log("lr", self.hparams.learning_rate, batch_size=len(images))
        self.log(
            "training_rotation_loss", rotation_loss, batch_size=len(images)
        )
        self.log(
            "training_contrast_loss", contrast_loss, batch_size=len(images)
        )
        self.log(
            "training_reconstruction_loss",
            reconstruction_loss,
            batch_size=len(images),
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images = batch[self.hparams.image_key]

        x1, rot1 = rot_rand(images)
        x2, rot2 = rot_rand(images)
        x1_augment = aug_rand(x1)
        x2_augment = aug_rand(x2)

        rot1_p, contrastive1_p, rec_x1 = self._model(x1_augment)
        rot2_p, contrastive2_p, rec_x2 = self._model(x2_augment)

        # compute the loss
        rot_p = torch.cat([rot1_p, rot2_p], dim=0)
        rots = torch.cat([rot1, rot2], dim=0)
        imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
        imgs = torch.cat([x1, x2], dim=0)
        loss, losses_tasks = self.loss(
            rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs
        )

        # do the loggine
        rotation_loss, contrast_loss, reconstruction_loss = losses_tasks
        val_outputs = {
            "val_loss": loss,
            "val_rotation_loss": rotation_loss,
            "val_contrast_loss": contrast_loss,
            "val_reconstruction_loss": reconstruction_loss,
            "val_number": len(images),
        }
        self.validation_step_outputs.append(val_outputs)

        return val_outputs

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        val_rotation_loss, val_contrast_loss, val_reconstruction_loss = 0, 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
            val_rotation_loss += output["val_rotation_loss"]
            val_contrast_loss += output["val_contrast_loss"]
            val_reconstruction_loss += output["val_reconstruction_loss"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        mean_val_rotation_loss = val_rotation_loss / num_items
        mean_val_contrast_loss = val_contrast_loss / num_items
        mean_val_reconstruction_loss = val_reconstruction_loss / num_items

        # write the logs

        self.log("val_loss", mean_val_loss, batch_size=num_items)
        self.log(
            "val_cosine_sim_pos", mean_val_rotation_loss, batch_size=num_items
        )
        self.log(
            "val_cosine_sim_neg", mean_val_contrast_loss, batch_size=num_items
        )
        self.log(
            "val_embedding_loss",
            mean_val_reconstruction_loss,
            batch_size=num_items,
        )

        # free memory
        self.validation_step_outputs.clear()
        self.val_metric.reset()
        return {"val_loss": mean_val_loss}


def patch_rand_drop(x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    device = x.get_device()
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0
    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)
        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s),
                dtype=x.dtype,
                device=device,
            ).normal_()
            x_uninitialized = (
                x_uninitialized - torch.min(x_uninitialized)
            ) / (torch.max(x_uninitialized) - torch.min(x_uninitialized))
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[
                :, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z
            ]
        total_pix = total_pix + (rnd_h - rnd_r) * (rnd_w - rnd_c) * (
            rnd_z - rnd_s
        )
    return x


def rot_rand(x_s: torch.Tensor):
    device = x_s.get_device()
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    x_rot = torch.zeros(img_n).long().to(device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(samples: torch.Tensor):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(x_aug[i], x_aug[idx_rnd])
    return x_aug
