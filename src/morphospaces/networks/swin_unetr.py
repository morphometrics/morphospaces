from typing import Dict

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.logging.image import log_images
from morphospaces.networks._components.embedding_swin_unetr import (
    EmbeddingSwinUNETR,
)


class PixelEmbeddingSwinUNETR(pl.LightningModule):
    def __init__(
        self,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        n_classes: int = 3,
        n_embedding_dims: int = 32,
        lr_scheduler_step: int = 1000,
        lr_reduction_factor: float = 0.2,
        lr_reduction_patience: int = 15,
    ):
        """The pixel embedding SwinUNETR.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        n_layers : int
            The number of layers in the ResNet.
            Default value is 2.
        n_embedding_dims : int
            The number of embedding dimensions.
            Default value is 32.
        n_classes : int
            The number of semantic classes in the training
            and validation datasets.
        lr_scheduler_step : int
            The number of steps between updating the learning rate scheduler.
            Default value is 1000.
        lr_reduction_factor : float
            The factor by which to reduce the learning rate. When training has
            plateued. Default value is 0.2.
        lr_reduction_patience : int
            The number of steps to wait before reducing the learning rate.
            Default value is 15.
        """
        super().__init__()

        # store parameters
        self.save_hyperparameters()

        self._model = EmbeddingSwinUNETR(
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=n_embedding_dims,
        )
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )

        # parameters to track training
        self.iteration_count = 0
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass."""
        return self._model(x)

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
        labels = batch[self.hparams.labels_key]
        embeddings, logits = self._model(images)

        # compute the loss
        loss = self.loss(logits, labels)

        # log the loss and learning rate
        self.log("training_loss", loss, batch_size=len(images), prog_bar=True)
        self.log("lr", self.hparams.learning_rate, batch_size=len(images))

        # log the images
        if (self.iteration_count % 200) == 0:
            # only log every 200 iterations
            log_images(
                input=images,
                target=labels,
                prediction=logits,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="segmentation",
            )

        self.iteration_count += 1

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images = batch[self.hparams.image_key]
        labels = batch[self.hparams.labels_key]
        embeddings, logits = self._model(images)

        # compute the loss
        loss = self.loss(logits, labels)

        # compute the metric
        self.val_metric(y_pred=logits, y=labels)

        # log the loss and learning rate
        val_outputs = {
            "val_loss": loss,
            "val_number": len(images),
            # "cosine_sim_pos": cosine_sim_pos,
            # "cosine_sim_neg": cosine_sim_neg,
        }
        self.validation_step_outputs.append(val_outputs)

        return val_outputs

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        # val_cosine_sim_pos, val_cosine_sim_neg = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
            # val_cosine_sim_pos += output["cosine_sim_pos"]
            # val_cosine_sim_neg += output["cosine_sim_neg"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        # mean_val_cosine_sim_pos = val_cosine_sim_pos / num_items
        # mean_val_cosine_sim_neg = val_cosine_sim_neg / num_items
        self.log("val_loss", mean_val_loss, batch_size=num_items)
        self.validation_step_outputs.clear()  # free memory
        self.val_metric.reset()
        return {"val_loss": mean_val_loss}
