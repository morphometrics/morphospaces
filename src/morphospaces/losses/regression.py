import torch
from torch.nn import SmoothL1Loss


class MaskedSmoothL1Loss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.l1_loss = SmoothL1Loss(reduction=reduction)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # mask has to be repeated to match channels
        expanded_mask = mask.repeat((1, input.shape[1], 1, 1, 1)).bool()
        masked_input = input.masked_select(expanded_mask)
        masked_target = target.masked_select(expanded_mask)

        return self.l1_loss(masked_input, masked_target)
