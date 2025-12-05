import torch


def unit_sigmoid(x: torch.Tensor, steepness: float = 6):
    """
    Sigmoid-like function that maps 0->0 and 1->1.

    Parameters
    ----------
    x : torch.Tensor
        The array to apply the function to.
    steepness : float
        Controls how steep the curve is (higher = steeper).
        Default value is 6.

    Returns
    -------
        Output tensor with same shape as input
    """
    # Shift input to center at 0.5, scale by steepness, apply sigmoid
    # Then scale and shift output to map [0,1] -> [0,1] with correct endpoints
    centered = steepness * (x - 0.5)
    sig = torch.sigmoid(centered)

    # Scale to map sigmoid(steepness * -0.5) -> 0
    # and sigmoid(steepness * 0.5) -> 1
    sig_min = torch.sigmoid(torch.tensor(-steepness / 2.0))
    sig_max = torch.sigmoid(torch.tensor(steepness / 2.0))

    return (sig - sig_min) / (sig_max - sig_min)


class MaskedRegressionSoftSkeletonRecallLoss(torch.nn.Module):
    def __init__(self, sigmoid_steepness: float = 10, smooth: float = 0.005):
        """ """
        super().__init__()

        self.sigmoid_steepness = sigmoid_steepness
        self.smooth = smooth

    def forward(self, prediction, labels, mask=None):

        # apply the sigmoid so all values are between 0 and 1
        normalized_prediction = unit_sigmoid(
            prediction,
            steepness=self.sigmoid_steepness,
        )

        with torch.no_grad():
            # count the number of skeleton voxels
            sum_gt = labels.sum() if mask is None else (labels * mask).sum()

        # compute the number of true positive
        n_true_positives = (
            (normalized_prediction * labels).sum()
            if mask is None
            else (normalized_prediction * labels * mask).sum()
        )

        # compute the recall
        recall = (n_true_positives + self.smooth) / (
            torch.clip(sum_gt + self.smooth, 1e-8)
        )

        recall = recall.mean()
        return -recall


class MaskedSegmentationSoftSkeletonRecallLoss(torch.nn.Module):
    def __init__(
        self,
        sigmoid_steepness: float = 10,
        smooth: float = 0.005,
        segmentation_channel: int = 1,
    ):
        """ """
        super().__init__()

        self.sigmoid_steepness = sigmoid_steepness
        self.smooth = smooth
        self.segmentation_channel = segmentation_channel

    def forward(self, prediction, labels, mask=None):

        # apply the sigmoid so all values are between 0 and 1
        normalized_prediction = torch.nn.functional.softmax(prediction, dim=1)

        # select the segmentation channel and make it have a channel dimension
        normalized_prediction = normalized_prediction[
            :, self.segmentation_channel, ...
        ]
        normalized_prediction = torch.unsqueeze(normalized_prediction, 1)

        with torch.no_grad():
            # count the number of skeleton voxels
            sum_gt = labels.sum() if mask is None else (labels * mask).sum()

        # compute the number of true positive
        n_true_positives = (
            (normalized_prediction * labels).sum()
            if mask is None
            else (normalized_prediction * labels * mask).sum()
        )

        # compute the recall
        recall = (n_true_positives + self.smooth) / (
            torch.clip(sum_gt + self.smooth, 1e-8)
        )

        recall = recall.mean()
        return -recall
