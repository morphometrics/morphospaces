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

        # mask out the diagonal
        self.logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0,
        )
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


class NCELoss(torch.nn.Module):
    """NCE contrastive loss from Wang et al., 2021

    See Equation 3 in https://arxiv.org/pdf/2101.11939.pdf
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        labels: torch.Tensor,
        contrastive_embeddings: torch.Tensor,
        contrastive_labels: torch.Tensor,
        mask_diagonal: bool = False,
    ):
        """Compute the loss.

        Parameters
        ----------
        predicted_embeddings : torch.Tensor
            (N, D) array containing the embeddings predicted by the in
            the forward pass where N is the number of embeddings
            and D is the dimension of the embeddings.
        labels : torch.Tensor
            (N,) array containing the class labels associated with
            the predicted_embeddings.
        contrastive_embeddings : torch.Tensor
            (M, D) array containing the embeddings to compare the
            predicted embeddings to where M is the number of
            contrastive_embeddings and D is the dimension of
            the embeddings.
        contrastive_labels : torch.Tensor
            (M,) array containing the class labels associated with
            the contrastive_embeddings.

        Returns
        -------
        The computed loss.
        """
        device = (
            torch.device("cuda")
            if predicted_embeddings.is_cuda
            else torch.device("cpu")
        )
        # make the masks
        positive_mask = (
            torch.eq(
                labels.view(-1, 1), contrastive_labels.contiguous().view(1, -1)
            )
            .float()
            .to(device)
        )
        negative_mask = 1 - positive_mask

        # make a positive mask without the diagonal if requested
        # I don't think this is necessary when the contrastive_embeddings
        # are not the same as predicted_embeddings
        # (i.e., there aren't any "self" embedding comparison
        # therefore just copying as a placeholder
        if mask_diagonal:
            logits_mask = torch.scatter(
                torch.ones_like(positive_mask),
                1,
                torch.arange(predicted_embeddings.shape[0])
                .view(-1, 1)
                .to(device),
                0,
            )

        else:
            logits_mask = positive_mask

        # normalize the embeddings
        predicted_embeddings = F.normalize(predicted_embeddings, dim=-1, p=2)
        contrastive_embeddings = F.normalize(
            contrastive_embeddings, dim=-1, p=2
        )

        # compute the logits (numerator)
        logits = torch.div(
            torch.matmul(predicted_embeddings, contrastive_embeddings.T),
            self.temperature,
        )
        logits = stablize_logits(logits)

        # compute the denominator
        negative_logits = torch.exp(logits) * negative_mask
        negative_logits = negative_logits.sum(1, keepdim=True)
        exp_positive_logits = torch.exp(logits)
        denominator = torch.log(exp_positive_logits + negative_logits)

        # compute the loss
        log_probability = logits - denominator
        mean_log_prob_pos = (logits_mask * log_probability).sum(
            1
        ) / logits_mask.sum(1)
        loss = -1 * mean_log_prob_pos
        return loss.mean()
