from typing import Dict

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.losses.embedding import MultiPosConLoss
from morphospaces.losses.util import sample_fixed_points
from morphospaces.networks._components.unet_model import ResidualUNet3D


class PixelEmbedding(pl.LightningModule):
    def __init__(
        self,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        n_layers: int = 2,
        n_embedding_dims: int = 32,
        loss_temperature: float = 0.1,
        n_samples_per_class: int = 10,
        learning_rate: float = 1e-4,
        lr_scheduler_step: int = 1000,
        lr_reduction_factor: float = 0.2,
        lr_reduction_patience: int = 15,
    ):
        """The PixelEmbedding model.

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
        loss_temperature : float
            The temperature for the MultiPosLoss
            Default value is 0.1.
        n_samples_per_class : int
            The number of embeddings samples to take per class when
            computing the loss. Default value is 10.
        learning_rate : float
            The learning rate for the Adam optimizer.
            Default value is 1e-4.
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

        self._model = ResidualUNet3D(
            in_channels=in_channels,
            out_channels=n_embedding_dims,
            num_levels=n_layers,
            conv_padding=1,
        )
        self.loss = MultiPosConLoss(temperature=self.hparams.loss_temperature)

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
        embeddings = self._model(images)

        # compute the loss
        loss = self._compute_loss(embeddings, labels)

        # log the loss and learning rate
        self.log("training_loss", loss, batch_size=len(images), prog_bar=True)
        self.log("lr", self.hparams.learning_rate, batch_size=len(images))

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
        embeddings = self._model(images)

        # compute the loss
        loss = self._compute_loss(embeddings, labels)

        # log the loss and learning rate
        val_outputs = {"val_loss": loss, "val_number": len(images)}
        self.validation_step_outputs.append(val_outputs)

        return val_outputs

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        self.log("val_loss", mean_val_loss, batch_size=num_items)
        self.validation_step_outputs.clear()  # free memory
        return {"val_loss": mean_val_loss}

    def _compute_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Compute the contrastive loss for the embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            (n, d) array containing the embeddings.
        labels : torch.Tensor
            (n,) array containing the label value for each embedding.

        Returns
        -------
        float
            The computed loss.
        """
        # sample the embeddings and loss
        # sampled_features = sample_random_features(
        # features=embeddings,
        # labels=labels,
        # num_samples_per_class=self.hparams.n_samples_per_class
        # )
        sampled_features = sample_fixed_points(
            features=embeddings, labels=labels
        )

        # compute the loss
        return self.loss(sampled_features)
