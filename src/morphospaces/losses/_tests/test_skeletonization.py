import torch

from morphospaces.losses.skeletonization import (
    MaskedRegressionSoftSkeletonRecallLoss,
)


def test_masked_soft_skeleton_recall():
    """Test the masked soft skeleton recall loss"""
    prediction = torch.zeros((1, 1, 10, 10, 10)).float()
    labels = torch.zeros((1, 1, 10, 10, 10)).float()
    mask = torch.zeros((1, 1, 10, 10, 10)).float()

    # valid region for the skeleton
    mask[:, :, 0:5, 0:5, 0:5] = 1.0

    # True positive (1, 1, 1)
    prediction[0, 0, 1, 1, 1] = 0.5
    labels[0, 0, 1, 1, 1] = 1.0

    # True positive (2, 2, 2)
    prediction[0, 0, 2, 2, 2] = 2.0
    labels[0, 0, 2, 2, 2] = 1.0

    # False negative (3, 3, 3)
    labels[0, 0, 3, 3, 3] = 1.0

    # Make the loss
    smooth_factor = 0.005
    loss_function = MaskedRegressionSoftSkeletonRecallLoss(
        sigmoid_steepness=10, smooth=smooth_factor
    )

    loss = loss_function(prediction, labels, mask)

    expected_loss = -(1.5 + smooth_factor) / (3 + smooth_factor)
    assert abs(loss - expected_loss) < 1e-2
