import numpy as np
import torch


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
