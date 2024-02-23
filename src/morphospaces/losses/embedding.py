import torch
import torch.nn.functional as F


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


class MultiPosConLoss(torch.nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, outputs):
        feats = outputs["feats"]  # feats shape: [B, D]
        labels = outputs["labels"]  # labels shape: [B]

        device = torch.device("cuda") if feats.is_cuda else torch.device("cpu")

        feats = F.normalize(feats, dim=-1, p=2)

        # mask is 1 if the pair of samples are from the same class
        mask = (
            torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1))
            .float()
            .to(device)
        )
        # mask = torch.eq(labels.view(-1, 1),
        #                 labels.contiguous().view(1, -1)).float()

        # mask out the diagonal
        self.logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0,
        )
        # self.logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(mask.shape[0]).view(-1, 1),
        #     0
        # )
        self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, feats.T) / self.temperature

        # mask out the diagonal
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss
