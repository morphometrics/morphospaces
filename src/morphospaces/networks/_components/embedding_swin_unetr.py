import torch
from monai.networks.nets import SwinUNETR


class EmbeddingSwinUNETR(SwinUNETR):
    """Swin UNETR with a modified forward that returns the embedding."""

    def forward(self, x_in):
        """During inference, just return the embedding"""
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        return self.decoder1(dec0, enc0)

    def training_forward(self, x_in):
        """During training, return the embedding and segmentation logits.

        Returns
        -------
        pixel_embedding : torch.Tensor
            The pixel embedding.
        logits : torch.Tensor
            The segmentation logits.
        """
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        pixel_embedding = self.decoder1(dec0, enc0)
        logits = self.out(pixel_embedding)
        return pixel_embedding, logits
