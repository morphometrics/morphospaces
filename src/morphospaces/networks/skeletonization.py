from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DeepSupervisionLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet
from monai.networks.nets import UNet as MonaiUnet
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from morphospaces.logging.image import log_images
from morphospaces.losses.regression import MaskedSmoothL1Loss
from morphospaces.losses.skeletonization import (
    MaskedRegressionSoftSkeletonRecallLoss,
)
from morphospaces.losses.util import MaskedDeepSupervisionLoss


class SkeletonizationDynUNet(pl.LightningModule):
    def __init__(
        self,
        image_key: str = "raw",
        labels_key: str = "label",
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_size: tuple[int] = (3, 3, 3, 3, 3, 3),
        strides: tuple[int] = (1, 2, 2, 2, 2, 2),
        upsample_kernel_size: tuple[int] = (2, 2, 2, 2, 2),
        dropout_rate: float = 0.0,
        learning_rate: float = 1e-4,
        lr_scheduler_interval: str = "step",
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
        deep_supr_num = max(1, len(strides) - 2)
        model = DynUNet(
            spatial_dims=3,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            deep_supervision=True,
            deep_supr_num=deep_supr_num,
            dropout=dropout_rate,
        )

        self._model = model

        # post-processing transforms for validation
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
        self.post_label = AsDiscrete(to_onehot=out_channels)

        # metrics for training and validation
        self.loss_function = DeepSupervisionLoss(
            DiceCELoss(to_onehot_y=True, softmax=True)
        )
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
                "interval": self.hparams.lr_scheduler_interval,
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

        # for the deep supervision make each scale an element of the list
        logits = torch.unbind(logits, dim=1)
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


class SkeletonizationRegressionDynUNet(pl.LightningModule):
    def __init__(
        self,
        image_key: str = "raw",
        labels_key: str = "label",
        skeleton_key: str = "skeleton",
        segmentation_key: str = "segmentation",
        in_channels: int = 1,
        out_channels: int = 2,
        kernel_size: tuple[int] = (3, 3, 3, 3, 3, 3),
        strides: tuple[int] = (1, 2, 2, 2, 2, 2),
        upsample_kernel_size: tuple[int] = (2, 2, 2, 2, 2),
        dropout_rate: float = 0.0,
        skeleton_recall_factor: float = 0.01,
        learning_rate: float = 1e-4,
        lr_scheduler_interval: str = "step",
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
        deep_supr_num = max(1, len(strides) - 2)
        model = DynUNet(
            spatial_dims=3,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            deep_supervision=True,
            deep_supr_num=deep_supr_num,
            dropout=dropout_rate,
        )

        self._model = model

        # post-processing transforms for validation
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
        self.post_label = AsDiscrete(to_onehot=out_channels)

        # metrics for training and validation
        self.regression_loss = MaskedDeepSupervisionLoss(
            MaskedSmoothL1Loss(reduction="mean")
        )
        self.skeleton_recall_loss = MaskedDeepSupervisionLoss(
            MaskedRegressionSoftSkeletonRecallLoss(
                sigmoid_steepness=10, smooth=0.005
            )
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
                "interval": self.hparams.lr_scheduler_interval,
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
        skeleton = batch[self.hparams.skeleton_key]
        segmentation = batch[self.hparams.segmentation_key]

        # make the forward pass and compute the loss
        prediction = self.forward(images)

        # for the deep supervision make each scale an element of the list
        prediction = torch.unbind(prediction, dim=1)
        regression_loss = self.regression_loss(
            input=prediction, target=labels, mask=segmentation
        )

        skeleton_loss = self.skeleton_recall_loss(
            input=prediction,
            target=skeleton,
            mask=segmentation,
        )

        # sum the losses
        loss = (
            regression_loss
            + self.hparams.skeleton_recall_factor * skeleton_loss
        )

        # log the loss and learning rate
        self.log(
            "training_loss",
            loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=self.hparams.distributed_training,
        )
        self.log(
            "regression_loss",
            regression_loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=self.hparams.distributed_training,
        )
        self.log(
            "skeleton_loss",
            skeleton_loss,
            batch_size=len(images),
            prog_bar=True,
            sync_dist=self.hparams.distributed_training,
        )

        # log the images
        if (self.iteration_count % self.hparams.log_image_every_n) == 0:
            with torch.no_grad():
                masked_prediction = [p * segmentation for p in prediction]
            # only log every 200 iterations
            log_images(
                input=images,
                target=labels,
                prediction=masked_prediction,
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
        skeleton = batch[self.hparams.skeleton_key]
        segmentation = batch[self.hparams.segmentation_key]

        # infer on the whole patch with sliding window
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )

        # compute the loss
        regression_loss = self.regression_loss(
            input=outputs, target=labels, mask=segmentation
        )

        skeleton_loss = self.skeleton_recall_loss(
            input=outputs,
            target=skeleton,
            mask=segmentation,
        )

        # sum the losses
        loss = (
            regression_loss
            + self.hparams.skeleton_recall_factor * skeleton_loss
        )

        # store the val loss
        val_outputs = {
            "val_loss": loss,
            "val_number": len(images),
        }
        self.validation_step_outputs.append(val_outputs)

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

        return {"val_loss": mean_val_loss}


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
        self.log("lr", self.learning_rate, batch_size=len(images))

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


class SemanticSkeletonNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        n_vectors_channels: int = 3,
        out_channels: int = 3,
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
        self.n_classes = out_channels

        # make the model
        self._model = SkeletonizationModel(
            in_channels=in_channels,
            n_vector_channels=n_vectors_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

        # setup the loss functions
        self.vector_loss_function = MaskedSmoothL1Loss(reduction="mean")

        # do ignore the background (channel 0)
        skeleton_class_weights = torch.ones(
            (self.n_classes), dtype=torch.float
        )
        skeleton_class_weights[0] = 0
        self.skeleton_loss_function = CrossEntropyLoss(
            weight=skeleton_class_weights, reduction="mean"
        )

        # setup the validation metric
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )

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
        skeleton_target = batch[self.skeleton_gt_key]
        mask = batch[self.mask_key]

        # make the prediction
        skeleton_vectors, skeleton_predictions = self._model.training_forward(
            images
        )

        assert skeleton_vectors.isnan().sum() == 0
        assert skeleton_predictions.isnan().sum() == 0

        # compute the loss for the intermediate vector prediction
        vector_loss = self.vector_loss_function(
            skeleton_vectors, skeleton_vectors_target, mask=mask
        )

        # compute the loss for the semantic skeleton prediction
        # skeleton_target is NCZYX, must make it NZYX
        skeleton_loss = self.skeleton_loss_function(
            skeleton_predictions, skeleton_target[:, 0, ...].long()
        )

        # sum the losses
        # total_loss = vector_loss + skeleton_loss
        total_loss = vector_loss + skeleton_loss

        self.log("training_loss", total_loss, batch_size=len(images))

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
                mask=mask,
            )
            log_images(
                input=images,
                target=skeleton_target,
                prediction=skeleton_predictions,
                iteration_index=self.iteration_count,
                logger=self.logger.experiment,
                prefix="skeleton_",
                mask=mask,
            )

        self.iteration_count += 1
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        """Implementation of one validation step.
        This performs inference on the entire validation image
        using a sliding window. See the pytorch-lightning
        module documentation for details.
        """
        images = batch[self.image_key]
        skeleton_target = batch[self.skeleton_gt_key]

        _, skeleton_predictions = self._model.training_forward(images)

        # compute the loss
        # skeleton_target is NCZYX, must make it NZYX
        loss = self.skeleton_loss_function(
            skeleton_predictions, skeleton_target[:, 0, ...].long()
        )

        # compute the metric
        predicted_skeleton = torch.argmax(skeleton_predictions, dim=1)
        predicted_one_hot = one_hot(
            torch.unsqueeze(predicted_skeleton, dim=1),
            num_classes=self.n_classes,
            dim=1,
        )
        self.dice_metric(
            y_pred=predicted_one_hot,
            y=one_hot(skeleton_target, num_classes=self.n_classes, dim=1),
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


class SkeletonizationModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_vector_channels: int = 9,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: Tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
    ):
        super().__init__()

        self._skeleleton_vector_net = MonaiUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=n_vector_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )
        self._skeleton_net = MonaiUnet(
            spatial_dims=3,
            in_channels=n_vector_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )

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

    def forward(self, x) -> torch.Tensor:
        """Inference forward pass"""
        _, skeleton = self.training_forward(x)

        return skeleton
