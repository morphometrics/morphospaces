import torch


class PixelEmbedding(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def _sample_pixels(
        self, input: torch.Tensor, label_image: torch.Tensor
    ) -> torch.Tensor:
        """Sample embeddings for pixels in the input based on the label image.

        This method
        """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        label_image: torch.Tensor,
    ) -> torch.Tensor:
        pass
