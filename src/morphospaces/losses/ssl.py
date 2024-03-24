import torch
import torch.nn as nn
from torch.nn import functional as F


class Contrast(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature))
        self.register_buffer(
            "neg_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float(),
        )

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (
            2 * self.batch_size
        )


class SSLLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.rot_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.L1Loss()
        self.contrast_loss = Contrast(batch_size)
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(
        self,
        output_rot,
        target_rot,
        output_contrastive,
        target_contrastive,
        output_recons,
        target_recons,
    ):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(
            output_contrastive, target_contrastive
        )
        recon_loss = self.alpha3 * self.recon_loss(
            output_recons, target_recons
        )
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)
