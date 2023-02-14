from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from monai.losses import ContrastiveLoss
from monai.networks.nets import ViTAutoEnc
from torch.nn import L1Loss

from morphospaces.networks.constants import (
    VIT_AC_AUG_VIEW_1_KEY,
    VIT_AC_AUG_VIEW_2_KEY,
    VIT_AC_GT_KEY,
)


class VitAutoencoder(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        positional_embedding: str = "conv",
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        reconstruction_loss_temperature: float = 0.05,
    ):
        super().__init__()

        # make the model
        self.model = ViTAutoEnc(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            pos_embed=positional_embedding,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
        )

        # define the losses
        self.reconstruction_loss = L1Loss()
        self.contrastive_loss = ContrastiveLoss(
            temperature=reconstruction_loss_temperature
        )

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Compute the loss"""

        # get the images
        inputs = batch[VIT_AC_AUG_VIEW_1_KEY]
        inputs_2 = batch[VIT_AC_AUG_VIEW_2_KEY]
        ground_truth_image = batch[VIT_AC_GT_KEY]

        # forward pass on the augmentations
        outputs_v1, hidden_v1 = self.model(inputs)
        outputs_v2, hidden_v2 = self.model(inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        reconstruction_loss = self.reconstruction_loss(
            outputs_v1, ground_truth_image
        )
        contrastive_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        total_loss = reconstruction_loss + (
            contrastive_loss * reconstruction_loss
        )

        return {"loss": total_loss}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> float:
        aug_images = batch[VIT_AC_AUG_VIEW_1_KEY]
        gt_images = batch[VIT_AC_GT_KEY]
        total_val_loss = 0
        val_step = 0
        for inputs, gt_input in zip(aug_images, gt_images):
            val_step += 1
            outputs, outputs_v2 = self.model(inputs)
            val_loss = self.reconstruction_loss(outputs, gt_input)
            total_val_loss += val_loss.item()

        total_val_loss /= val_step

        return total_val_loss
