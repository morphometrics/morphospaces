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

    output = {}
    output["feats"] = sampled_feats
    output["labels"] = sampled_labels

    return output


def sample_fixed_points(features, labels):
    device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
    sample_points = torch.from_numpy(
        np.array(
            [
                [29, 53, 13],
                [29, 54, 40],
                [29, 55, 50],
                [29, 27, 39],
                [29, 18, 41],
                [29, 10, 40],
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
    return {
        "feats": sampled_features.reshape(-1, sampled_features.shape[2]),
        "labels": sampled_labels.reshape(-1, sampled_labels.shape[2]),
    }
