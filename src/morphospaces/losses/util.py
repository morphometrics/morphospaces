from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MaskedDeepSupervisionLoss(_Loss):
    """
    Wrapper class around a masked function to accept
    a list of tensors returned from a deeply supervised network.
    The final loss is computed as the sum of weighted losses
    for each of deep supervision levels.
    """

    def __init__(
        self,
        loss: torch.nn.Module,
        weight_mode: str = "exp",
        weights: list[float] | None = None,
    ) -> None:
        """
        Args:
            loss: main loss instance, e.g DiceLoss().
            weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level.
                Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2:
                    1, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels:
                    1, 0.5, 0.5, 0.5, 0.5, etc
            weights: a list of weights to apply to each
                deeply supervised sub-loss, if provided,
                this will be used regardless of the weight_mode
        """
        super().__init__()
        self.loss = loss
        self.weight_mode = weight_mode
        self.weights = weights
        self.interp_mode = "nearest-exact"

    def get_weights(self, levels: int = 1) -> list[float]:
        """
        Calculates weights for a given number of scale levels
        """
        levels = max(1, levels)
        if self.weights is not None and len(self.weights) >= levels:
            weights = self.weights[:levels]
        elif self.weight_mode == "same":
            weights = [1.0] * levels
        elif self.weight_mode == "exp":
            weights = [
                max(0.5**l_index, 0.0625) for l_index in range(levels)
            ]
        elif self.weight_mode == "two":
            weights = [
                1.0 if l_index == 0 else 0.5 for l_index in range(levels)
            ]
        else:
            weights = [1.0] * levels

        return weights

    def get_loss(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.tensor
    ) -> torch.Tensor:
        """
        Calculates a loss output accounting for differences in shapes,
        and downsizing targets if necessary
        (using nearest neighbor interpolation)
        Generally downsizing occurs for all level,
        except for the first (level==0)
        """
        if input.shape[2:] != target.shape[2:]:
            target = F.interpolate(
                target, size=input.shape[2:], mode=self.interp_mode
            )
        return self.loss(input, target, mask)  # type: ignore[no-any-return]

    def forward(
        self,
        input: Union[None, torch.Tensor, list[torch.Tensor]],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            weights = self.get_weights(levels=len(input))
            loss = torch.tensor(0, dtype=torch.float, device=target.device)
            for l_index in range(len(input)):
                loss += weights[l_index] * self.get_loss(
                    input[l_index].float(), target, mask
                )
            return loss
        if input is None:
            raise ValueError("input shouldn't be None.")

        return self.loss(input.float(), target, mask)


def sample_random_features(features, labels, num_samples_per_class=10):
    """
    Sample random features from the input features and labels

    Args:
    - features: torch.Tensor, shape: [B, C, H, W, D]
    - labels: torch.Tensor, shape: [B, H, W, D]
    - num_samples_per_class: int, number of samples to sample per class
    """
    device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

    # transform features to shape [B*H*W*D, C]
    features = features.permute(0, 2, 3, 4, 1)
    features = features.reshape(-1, features.shape[-1])
    labels = labels.view(-1)

    # get the unique labels, not sure if this is too inefficient
    unique_labels = torch.unique(labels)

    # sample random features
    sampled_feats = []
    sampled_labels = []
    for label in unique_labels:
        label_feats = features[labels == label]
        if label_feats.size(0) > num_samples_per_class:
            idx = torch.randperm(label_feats.size(0))[:num_samples_per_class]
            sampled_feats.append(label_feats[idx])
            sampled_labels.append(
                label
                * torch.ones(
                    num_samples_per_class, device=device, dtype=torch.long
                )
            )
        else:
            sampled_feats.append(label_feats)
            sampled_labels.append(
                label
                * torch.ones(
                    label_feats.size(0), device=device, dtype=torch.long
                )
            )

    sampled_feats = torch.cat(sampled_feats, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    return sampled_feats, sampled_labels


def sample_fixed_points(features, labels):
    device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
    sample_points = torch.from_numpy(
        np.array(
            [
                [12, 10, 7],
                [12, 8, 14],
                [14, 10, 7],
                [14, 8, 15],
                [18, 10, 7],
                [35, 35, 35],
                [35, 35, 40],
                [28, 35, 38],
                [38, 35, 32],
                [35, 30, 40],
            ],
            dtype=int,
        ),
    ).to(device)
    sampled_features = features[
        :, :, sample_points[:, 0], sample_points[:, 1], sample_points[:, 2]
    ]
    sampled_labels = labels[
        :, :, sample_points[:, 0], sample_points[:, 1], sample_points[:, 2]
    ]
    sampled_features = sampled_features.permute(0, 2, 1)
    sampled_labels = sampled_labels.permute(0, 2, 1)
    reshaped_sampled_features = sampled_features.reshape(
        -1, sampled_features.shape[2]
    )
    reshaped_sampled_labels = sampled_labels.reshape(
        -1, sampled_labels.shape[2]
    )
    return reshaped_sampled_features, reshaped_sampled_labels


def cosine_similarities(embeddings: torch.Tensor, labels: torch.Tensor):
    """
    Compute the mean cosine similarity between the features of the same class
    and the mean cosine similarity between the features of different classes.

    Args:
    - embeddings: torch.Tensor, shape: [N, D]
    - labels: torch.Tensor, shape: [B]
    """
    # mask of positives without the diagonal
    positive_mask = torch.eq(
        labels.view(-1, 1), labels.contiguous().view(1, -1)
    )
    self_mask = torch.eye(embeddings.shape[0], device=embeddings.device)
    positive_mask.masked_fill(self_mask.bool(), 0)

    # mask of negatives without the diagonal
    negative_mask = torch.logical_not(positive_mask)

    # normalize the features and compute similarity
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarities = torch.mm(embeddings, embeddings.t())

    positive_similarities = similarities[positive_mask].mean()
    negative_similarities = similarities[negative_mask].mean()

    return positive_similarities, negative_similarities
