from typing import Dict, List

import pytorch_lightning as pl
import torch

from morphospaces.logging.image import log_images
from morphospaces.losses.regression import MaskedSmoothL1Loss
from morphospaces.networks._components.unet_model import MultiscaleUnet3D


class MultiscaleSkeletonizationNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        image_key: str = "image",
        labels_key: str = "skeleton_labels",
        skeletonization_target_key: str = "skeletonization_target",
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        # store parameters
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.labels_key = labels_key
        self.skeletonization_target_key = skeletonization_target_key

        # make the model
        self._model = MultiscaleUnet3D(
            in_channels=in_channels,
            out_channels=1,
        )

        # setup the loss functions
        self.scale_0_loss = MaskedSmoothL1Loss(reduction="mean")
        self.scale_1_loss = MaskedSmoothL1Loss(reduction="mean")
        self.scale_2_loss = MaskedSmoothL1Loss(reduction="mean")

        self.iteration_count = 0

    def forward(self, x) -> torch.Tensor:
        """Inference forward pass"""
        return self._model(x)

    def configure_optimizers(self):
        """Set up the Adam optimzier.
        See the pytorch-lightning module documentation for details.
        """
        optimizer = torch.optim.Adam(
            self._model.parameters(), self.learning_rate
        )
        return optimizer

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Implementation of one training step.
        See the pytorch-lightning module documentation for details.
        """
        # get scale 0 data
        images = batch[self.image_key]
        labels_scale_0 = batch[self.labels_key]
        skeletonization_target_scale_0 = batch[self.skeletonization_target_key]

        # get scale 1 data (downscaled by 2)
        skeleton_1_key = f"{self.skeletonization_target_key}_reduced_2"
        labels_scale_1_key = f"{self.label_key}_reduced_2"
        skeletonization_target_scale_1 = batch[skeleton_1_key]
        labels_scale_1 = batch[labels_scale_1_key]

        # get scale 2 data (downscaled by 4)
        skeleton_2_key = f"{self.skeletonization_target_key}_reduced_4"
        labels_scale_2_key = f"{self.label_key}_reduced_4"
        skeletonization_target_scale_2 = batch[skeleton_2_key]
        labels_scale_2 = batch[labels_scale_2_key]

        # make the prediction
        skeleton, decoder_outputs = self._model.training_forward(images)

        loss = self._compute_loss(
            skeleton_prediction=skeleton,
            decoder_outputs=decoder_outputs,
            labels_scale_0=labels_scale_0,
            labels_scale_1=labels_scale_1,
            labels_scale_2=labels_scale_2,
            skeletonization_target_scale_0=skeletonization_target_scale_0,
            skeletonization_target_scale_1=skeletonization_target_scale_1,
            skeletonization_target_scale_2=skeletonization_target_scale_2,
        )

        self.log("training_loss", loss, batch_size=len(images))
        self.log("lr", self.learning_rate, batch_size=len(images))

        # log the images
        if (self.iteration_count % 200) == 0:
            # only log every 200 iterations
            log_images(
                input=images,
                target=skeletonization_target_scale_0,
                prediction=skeleton,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="scale 0",
            )
            log_images(
                input=images,
                target=skeletonization_target_scale_1,
                prediction=decoder_outputs[1],
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="scale 1",
            )

            log_images(
                input=images,
                target=skeletonization_target_scale_2,
                prediction=decoder_outputs[0],
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="scale 2",
            )

        self.iteration_count += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        # get scale 0 data
        images = batch[self.image_key]
        labels_scale_0 = batch[self.labels_key]
        skeletonization_target_scale_0 = batch[self.skeletonization_target_key]

        # get scale 1 data (downscaled by 2)
        skeleton_1_key = f"{self.skeletonization_target_key}_reduced_2"
        labels_scale_1_key = f"{self.label_key}_reduced_2"
        skeletonization_target_scale_1 = batch[skeleton_1_key]
        labels_scale_1 = batch[labels_scale_1_key]

        # get scale 2 data (downscaled by 4)
        skeleton_2_key = f"{self.skeletonization_target_key}_reduced_4"
        labels_scale_2_key = f"{self.label_key}_reduced_4"
        skeletonization_target_scale_2 = batch[skeleton_2_key]
        labels_scale_2 = batch[labels_scale_2_key]

        # make the prediction
        skeleton, decoder_outputs = self._model.training_forward(images)

        loss = self._compute_loss(
            skeleton_prediction=skeleton,
            decoder_outputs=decoder_outputs,
            labels_scale_0=labels_scale_0,
            labels_scale_1=labels_scale_1,
            labels_scale_2=labels_scale_2,
            skeletonization_target_scale_0=skeletonization_target_scale_0,
            skeletonization_target_scale_1=skeletonization_target_scale_1,
            skeletonization_target_scale_2=skeletonization_target_scale_2,
        )

        return {"val_loss": loss, "val_number": len(images)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        self.log("val_loss", mean_val_loss, batch_size=num_items)
        return {"val_loss": mean_val_loss}

    def _compute_loss(
        self,
        skeleton_prediction: torch.Tensor,
        decoder_outputs: List[torch.Tensor],
        labels_scale_0: torch.Tensor,
        labels_scale_1: torch.Tensor,
        labels_scale_2: torch.Tensor,
        skeletonization_target_scale_0: torch.Tensor,
        skeletonization_target_scale_1: torch.Tensor,
        skeletonization_target_scale_2: torch.Tensor,
    ) -> torch.Tensor:

        # scale 0 loss
        scale_0_mask = labels_scale_0 > 0
        scale_0_loss = self.scale_0_loss(
            input=skeleton_prediction,
            target=skeletonization_target_scale_0,
            maks=scale_0_mask,
        )

        # scale 1 loss
        scale_1_mask = labels_scale_1 > 0
        scale_1_loss = self.scale_1_loss(
            input=decoder_outputs[1],
            target=skeletonization_target_scale_1,
            maks=scale_1_mask,
        )

        # scale 2 loss
        scale_2_mask = labels_scale_2 > 0
        scale_2_loss = self.scale_2_loss(
            input=decoder_outputs[0],
            target=skeletonization_target_scale_2,
            maks=scale_2_mask,
        )

        return scale_0_loss + scale_1_loss + scale_2_loss
