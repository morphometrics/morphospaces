from typing import Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.networks.utils import one_hot

from morphospaces.logging.image import log_images


class SemanticUnetr(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        image_shape: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        n_heads: int = 12,
        positional_embedding: str = "perceptron",
        feature_normalization: str = "instance",
        residual_block: bool = True,
        convolutional_block: bool = True,
        loss_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        image_key: str = "raw",
        label_key: str = "label",
    ):
        super().__init__()

        self._model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=image_shape,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=n_heads,
            pos_embed=positional_embedding,
            norm_name=feature_normalization,
            res_block=residual_block,
            conv_block=convolutional_block,
            dropout_rate=dropout_rate,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if loss_function is None:
            self.loss_function = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                include_background=False,
                ce_weight=torch.Tensor([0, 0.1, 1]),
            )
        else:
            self.loss_function = loss_function
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

        # keys for getting the images from the data dict
        self.image_key = image_key
        self.label_key = label_key

        # number or channels for converting to one-hot in val step
        self.n_classes = out_channels

        # counter for the number of iterations
        # used for choosing when to log images
        self.iteration_count = 0

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Implementation of one training step.
        See the pytorch-lightning module documentation for details.
        """
        images = batch[self.image_key]
        labels = batch[self.label_key]

        predictions = self._model(images)

        # compute the loss
        loss = self.loss_function(predictions, labels)

        self.log("training_loss", loss, batch_size=len(images))

        # log the images
        if (self.iteration_count % 100) == 0:
            # only log every 100 iterations
            log_images(
                input=images,
                target=labels,
                prediction=predictions,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="skeleton_",
                mask=None,
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
        labels = batch[self.label_key]

        predictions = self._model(images)

        # compute the loss
        loss = self.loss_function(predictions, labels)

        # compute the metric
        predicted_skeleton = torch.argmax(predictions, dim=1)
        predicted_one_hot = one_hot(
            torch.unsqueeze(predicted_skeleton, dim=1),
            num_classes=self.n_classes,
            dim=1,
        )
        self.dice_metric(
            y_pred=predicted_one_hot,
            y=one_hot(labels, num_classes=self.n_classes, dim=1),
        )

        return {"val_loss": loss, "val_number": len(images)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_loss = torch.tensor(val_loss / num_items)

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        self.log("val_loss", mean_val_loss, batch_size=num_items)
        self.log("val_metric", mean_val_dice, batch_size=num_items)
        return {"val_loss": mean_val_loss, "val_metric": mean_val_dice}
