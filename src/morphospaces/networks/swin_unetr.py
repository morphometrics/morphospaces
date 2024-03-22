from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.logging.image import log_images
from morphospaces.losses.embedding import NCELoss
from morphospaces.losses.util import (
    cosine_similarities,
    sample_random_features,
)
from morphospaces.networks._components.embedding_swin_unetr import (
    EmbeddingSwinUNETR,
)
from morphospaces.networks._components.memory_bank import (
    LabelMemoryBank,
    PixelMemoryBank,
)


class PixelEmbeddingSwinUNETR(pl.LightningModule):
    def __init__(
        self,
        pretrained_weights_path: Optional[str] = None,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        n_embedding_dims: int = 32,
        loss_temperature: float = 0.1,
        n_samples_per_class: int = 10,
        learning_rate: float = 1e-4,
        lr_scheduler_step: int = 1000,
        lr_reduction_factor: float = 0.2,
        lr_reduction_patience: int = 15,
        memory_banks: bool = False,
        label_values: Optional[List[int]] = None,
        n_pixel_embeddings_per_class: int = 50,
        n_pixel_embeddings_to_update: int = 5,
        n_label_embeddings_per_class: int = 10,
        n_memory_warmup: int = 1000,
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

        n_classes = len(label_values)
        self._model = EmbeddingSwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=n_embedding_dims,
        )

        if pretrained_weights_path is not None:
            # load the weights
            weights = torch.load(pretrained_weights_path)

            # update the model
            self._model.load_from(weights)

        self.segmentation_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.contrastive_loss = NCELoss(
            temperature=self.hparams.loss_temperature
        )
        self.val_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )

        # make the memory banks if requested
        if self.hparams.memory_banks:
            if self.hparams.label_values is None:
                raise ValueError(
                    "If memory_banks is True, label_values must not be None."
                )

            self.pixel_memory_bank = PixelMemoryBank(
                n_embeddings_per_class=self.hparams.n_pixel_embeddings_per_class,  # noqa E501
                n_embeddings_to_update=self.hparams.n_pixel_embeddings_to_update,  # noqa E501
                n_dimensions=self.hparams.n_embedding_dims,
                label_values=self.hparams.label_values,
            )

            self.label_memory_bank = LabelMemoryBank(
                n_embeddings_per_class=self.hparams.n_label_embeddings_per_class,  # noqa E501
                n_dimensions=self.hparams.n_embedding_dims,
                label_values=self.hparams.label_values,
            )

        else:
            self.pixel_memory_bank = None
            self.label_memory_bank = None

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
        embeddings, logits = self._model.training_forward(images)

        # compute the loss
        segmentation_loss = self.segmentation_loss(logits, labels)

        (
            embedding_loss,
            cosine_sim_pos,
            cosine_sim_neg,
        ) = self._compute_embedding_loss(
            embeddings, labels, update_memory_bank=True
        )

        if self.hparams.memory_banks:
            loss = segmentation_loss + embedding_loss
        else:
            loss = segmentation_loss

        # log the loss and learning rate
        self.log("embedding_loss", embedding_loss, batch_size=len(images))
        self.log(
            "segmentation_loss", segmentation_loss, batch_size=len(images)
        )
        self.log("training_loss", loss, batch_size=len(images), prog_bar=True)
        self.log("lr", self.hparams.learning_rate, batch_size=len(images))
        self.log("cosine_sim_pos", cosine_sim_pos, batch_size=len(images))
        self.log("cosine_sim_neg", cosine_sim_neg, batch_size=len(images))

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
        embeddings, logits = self._model.training_forward(images)

        # compute the loss
        segmentation_loss = self.segmentation_loss(logits, labels)

        (
            embedding_loss,
            cosine_sim_pos,
            cosine_sim_neg,
        ) = self._compute_embedding_loss(
            embeddings, labels, update_memory_bank=False
        )

        if self.hparams.memory_banks:
            loss = segmentation_loss + embedding_loss
        else:
            loss = segmentation_loss

        # compute the metric
        self.val_metric(y_pred=logits, y=labels)

        # log the loss and learning rate
        val_outputs = {
            "val_loss": loss,
            "val_segmentation_loss": segmentation_loss,
            "val_embedding_loss": embedding_loss,
            "val_number": len(images),
            "cosine_sim_pos": cosine_sim_pos,
            "cosine_sim_neg": cosine_sim_neg,
        }
        self.validation_step_outputs.append(val_outputs)

        return val_outputs

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        val_cosine_sim_pos, val_cosine_sim_neg = 0, 0
        val_embedding_loss = 0
        val_segmentation_loss = 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
            val_cosine_sim_pos += output["cosine_sim_pos"]
            val_cosine_sim_neg += output["cosine_sim_neg"]
            val_embedding_loss += output["val_embedding_loss"]
            val_segmentation_loss += output["val_segmentation_loss"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        mean_val_cosine_sim_pos = val_cosine_sim_pos / num_items
        mean_val_cosine_sim_neg = val_cosine_sim_neg / num_items
        mean_val_segmentation_loss = val_segmentation_loss / num_items
        mean_val_embedding_loss = val_embedding_loss / num_items

        # write the logs
        self.log("val_loss", mean_val_loss, batch_size=num_items)
        self.log(
            "val_cosine_sim_pos", mean_val_cosine_sim_pos, batch_size=num_items
        )
        self.log(
            "val_cosine_sim_neg", mean_val_cosine_sim_neg, batch_size=num_items
        )
        self.log(
            "val_embedding_loss", mean_val_embedding_loss, batch_size=num_items
        )
        self.log(
            "val_segmentation_loss",
            mean_val_segmentation_loss,
            batch_size=num_items,
        )

        # free memory
        self.validation_step_outputs.clear()
        self.val_metric.reset()
        return {"val_loss": mean_val_loss}

    def _compute_embedding_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        update_memory_bank: bool = False,
    ) -> Tuple[float, float, float]:
        """Compute the contrastive loss for the embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            (n, d) array containing the embeddings.
        labels : torch.Tensor
            (n,) array containing the label value for each embedding.
        update_memory_bank : bool
            If set to True, the embeddings will be added to the memory bank.
            Generally, this is set to True during trianing and False during
            validation. Default value is False.

        Returns
        -------
        float
            The computed loss.
        """
        # sample the embeddings and loss
        sampled_embeddings, sampled_labels = sample_random_features(
            features=embeddings,
            labels=labels,
            num_samples_per_class=self.hparams.n_samples_per_class,
        )
        cosine_sim_pos, cosine_sim_neg = cosine_similarities(
            embeddings=sampled_embeddings, labels=sampled_labels
        )

        if self.hparams.memory_banks:
            if self.iteration_count > self.hparams.n_memory_warmup:
                device = (
                    torch.device("cuda")
                    if embeddings.is_cuda
                    else torch.device("cpu")
                )
                (
                    stored_pixel_embeddings,
                    stored_pixel_labels,
                ) = self.pixel_memory_bank.get_embeddings()
                (
                    stored_label_embeddings,
                    stored_label_labels,
                ) = self.label_memory_bank.get_embeddings()
                contrastive_embeddings = torch.cat(
                    [stored_pixel_embeddings, stored_label_embeddings]
                ).to(device)
                contrastive_labels = torch.cat(
                    [stored_pixel_labels, stored_label_labels]
                ).to(device)

                loss = self.contrastive_loss(
                    predicted_embeddings=sampled_embeddings,
                    labels=sampled_labels,
                    contrastive_embeddings=contrastive_embeddings,
                    contrastive_labels=contrastive_labels,
                    mask_diagonal=False,
                )

            else:
                loss = self.contrastive_loss(
                    predicted_embeddings=sampled_embeddings,
                    labels=sampled_labels,
                    contrastive_embeddings=sampled_embeddings,
                    contrastive_labels=sampled_labels,
                    mask_diagonal=True,
                )

            if update_memory_bank:
                # update the memory bank
                self.pixel_memory_bank.set_embeddings(
                    embeddings=sampled_embeddings, labels=sampled_labels
                )
                self.label_memory_bank.set_embeddings(
                    embeddings=embeddings, labels=labels
                )

        else:
            loss = self.contrastive_loss(
                predicted_embeddings=sampled_embeddings,
                labels=sampled_labels,
                contrastive_embeddings=sampled_embeddings,
                contrastive_labels=sampled_labels,
                mask_diagonal=True,
            )

        # return the loss and cosine similarities
        return loss, cosine_sim_pos, cosine_sim_neg

    # def _compute_embedding_val_loss(
    #     self, embeddings: torch.Tensor, labels: torch.Tensor
    # ) -> Tuple[float, float, float]:
    #     """Compute the contrastive loss for the embeddings for
    #     the validation step.
    #
    #     Parameters
    #     ----------
    #     embeddings : torch.Tensor
    #         (n, d) array containing the embeddings.
    #     labels : torch.Tensor
    #         (n,) array containing the label value for each embedding.
    #
    #     Returns
    #     -------
    #     float
    #         The computed loss.
    #     """
    #     # sample the embeddings and loss
    #     sampled_embeddings, sampled_labels = sample_random_features(
    #         features=embeddings,
    #         labels=labels,
    #         num_samples_per_class=self.hparams.n_samples_per_class,
    #     )
    #     cosine_sim_pos, cosine_sim_neg = cosine_similarities(
    #         embeddings=sampled_embeddings, labels=sampled_labels
    #     )
    #
    #     loss = self.contrastive_loss(
    #         predicted_embeddings=sampled_embeddings,
    #         labels=sampled_labels,
    #         contrastive_embeddings=sampled_embeddings,
    #         contrastive_labels=sampled_labels,
    #         mask_diagonal=True,
    #     )
    #
    #     # return the loss and cosine similarities
    #     return loss, cosine_sim_pos, cosine_sim_neg

    @classmethod
    def from_pretrained_vit_weights(
        cls,
        weights_path: str,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        n_embedding_dims: int = 32,
        loss_temperature: float = 0.1,
        n_samples_per_class: int = 10,
        learning_rate: float = 1e-4,
        lr_scheduler_step: int = 1000,
        lr_reduction_factor: float = 0.2,
        lr_reduction_patience: int = 15,
        memory_banks: bool = False,
        label_values: Optional[List[int]] = None,
        n_pixel_embeddings_per_class: int = 50,
        n_pixel_embeddings_to_update: int = 5,
        n_label_embeddings_per_class: int = 10,
        n_memory_warmup: int = 1000,
    ):
        swin_unetr = cls(
            image_key=image_key,
            labels_key=labels_key,
            in_channels=in_channels,
            n_embedding_dims=n_embedding_dims,
            loss_temperature=loss_temperature,
            n_samples_per_class=n_samples_per_class,
            learning_rate=learning_rate,
            lr_scheduler_step=lr_scheduler_step,
            lr_reduction_factor=lr_reduction_factor,
            lr_reduction_patience=lr_reduction_patience,
            memory_banks=memory_banks,
            label_values=label_values,
            n_pixel_embeddings_per_class=n_pixel_embeddings_per_class,
            n_pixel_embeddings_to_update=n_pixel_embeddings_to_update,
            n_label_embeddings_per_class=n_label_embeddings_per_class,
            n_memory_warmup=n_memory_warmup,
        )

        # load the weights
        weights = torch.load(weights_path)

        # update the model
        swin_unetr._model.load_from(weights)

        return swin_unetr
