from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet as MonaiUnet

from morphospaces.logging.image import log_images
from morphospaces.losses.regression import MaskedSmoothL1Loss


class SkeletonizationNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        image_key: str = "image",
        vectors_gt_key: str = "skeleton_vectors_target",
        skeleton_gt_key: str = "skeletonization_target",
        mask_key: str = "mask",
        learning_rate: float = 1e-4,
    ):
        super().__init__()

        # store parameters
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.vectors_gt_key = vectors_gt_key
        self.skeleton_gt_key = skeleton_gt_key
        self.mask_key = mask_key

        # make the model
        self._model = SkeletonizationModel(
            in_channels=in_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        # setup the loss functions
        self.vector_loss_function = MaskedSmoothL1Loss(reduction="mean")
        self.skeleton_loss_function = MaskedSmoothL1Loss(reduction="mean")

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
        images = batch[self.image_key]
        skeleton_vectors_target = batch[self.vectors_gt_key]
        skelton_target = batch[self.skeleton_gt_key]
        mask = batch[self.mask_key]

        skeleton_vectors, skeleton_predictions = self._model.training_forward(
            images
        )

        loss = self._compute_loss(
            vectors_prediction=skeleton_vectors,
            skeleton_prediction=skeleton_predictions,
            vectors_gt_target=skeleton_vectors_target,
            skeleton_gt_target=skelton_target,
            mask=mask,
        )

        self.log("training_loss", loss, batch_size=len(images))

        # log the images
        if (self.iteration_count % 100) == 0:
            # only log every 100 iterations
            log_images(
                input=images,
                target=skeleton_vectors_target,
                prediction=skeleton_vectors,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="vectors_",
            )
            log_images(
                input=images,
                target=skelton_target,
                prediction=skeleton_predictions,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="skeleton_",
            )

        self.iteration_count += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images = batch[self.image_key]
        skeleton_vectors_target = batch[self.vectors_gt_key]
        skelton_target = batch[self.skeleton_gt_key]
        mask = batch[self.mask_key]

        skeleton_vectors, skeleton_predictions = self._model.training_forward(
            images
        )

        loss = self._compute_loss(
            vectors_prediction=skeleton_vectors,
            skeleton_prediction=skeleton_predictions,
            vectors_gt_target=skeleton_vectors_target,
            skeleton_gt_target=skelton_target,
            mask=mask,
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
        vectors_prediction: torch.Tensor,
        skeleton_prediction: torch.Tensor,
        vectors_gt_target: torch.Tensor,
        skeleton_gt_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        vectors_loss = self.vector_loss_function(
            vectors_prediction, vectors_gt_target, mask=mask
        )
        skeleton_loss = self.skeleton_loss_function(
            skeleton_prediction, skeleton_gt_target, mask=mask
        )

        return vectors_loss + skeleton_loss


class SkeletonizationModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
    ):
        super().__init__()

        self._skeleleton_vector_net = MonaiUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=9,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )
        self._skeleton_net = MonaiUnet(
            spatial_dims=3,
            in_channels=9,
            out_channels=3,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )

    def forward(self, x) -> torch.Tensor:
        """Inference forward pass"""
        skeleton_vectors = self._skeleleton_vector_net(x)
        return self._skeleton_net(skeleton_vectors)

    def training_forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for traning

        Returns
        -------
        skeleton_vectors : torch.Tensor
            The skeleton vector field prediction.
        skeleton : torch.Tensor
            The skeletonization prediction
        """
        skeleton_vectors = self._skeleleton_vector_net(x)
        skeleton = self._skeleton_net(skeleton_vectors)
        return skeleton_vectors, skeleton
