from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.logging.image import log_images


class SemanticSwinUNETR(pl.LightningModule):
    def __init__(
        self,
        pretrained_weights_path: Optional[str] = None,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        lr_scheduler_step: int = 1000,
        lr_reduction_factor: float = 0.2,
        lr_reduction_patience: int = 15,
        log_image_every_n: int = 200,
        distributed_training: bool = False,
    ):
        super().__init__()

        # store parameters
        self.save_hyperparameters()

        # make the model
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            drop_rate=dropout_rate,
        )

        if pretrained_weights_path is not None:
            # load the pre-trained encoder weights
            weights = torch.load(pretrained_weights_path, weights_only=True)
            model.load_from(weights=weights)

        self._model = model

        # post-processing transforms for validation
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
        self.post_label = AsDiscrete(to_onehot=out_channels)

        # metrics for training and validation
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.validation_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
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
        optimizer = torch.optim.AdamW(
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

        # make the forward pass and compute the loss
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)

        # log the loss and learning rate
        self.log(
            "training_loss",
            loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=self.hparams.distributed_training,
        )

        # log the images
        if (self.iteration_count % self.hparams.log_image_every_n) == 0:
            # only log every 200 iterations
            log_images(
                input=images,
                target=labels,
                prediction=logits,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
            )

        self.iteration_count += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        # get the images
        images = batch[self.hparams.image_key]
        labels = batch[self.hparams.labels_key]

        # infer on the whole patch with sliding window
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )

        # compute the loss
        loss = self.loss_function(outputs, labels)

        # store the val loss
        val_outputs = {
            "val_loss": loss,
            "val_number": len(images),
        }
        self.validation_step_outputs.append(val_outputs)

        # compute the val metric
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.validation_metric(y_pred=outputs, y=labels)

        return val_outputs

    def on_validation_epoch_end(self):
        """Implementation of the validation epoch end.

        This computes the mean validation loss across all
        validation steps.

        See the pytorch-lightning module documentation for details.
        """
        # compute the mean validation loss
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)
        self.log(
            "val_loss",
            mean_val_loss,
            batch_size=num_items,
            sync_dist=self.hparams.distributed_training,
        )
        self.validation_step_outputs.clear()  # free memory

        # compute the mean validation metric
        mean_val_dice = self.validation_metric.aggregate().item()
        self.validation_metric.reset()
        self.log(
            "val_dice",
            mean_val_dice,
            batch_size=num_items,
            sync_dist=self.hparams.distributed_training,
        )

        return {"val_loss": mean_val_loss, "val_dice": mean_val_dice}
