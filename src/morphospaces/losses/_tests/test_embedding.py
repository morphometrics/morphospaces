import torch

from morphospaces.losses.embedding import MultiPosConLoss, NCELoss


def test_multi_pos_con_loss():
    labels = torch.tensor([0, 0, 2, 2])
    # two classes are clustered and separated
    good_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.8, 0.1, 0.3],
            [0.82, 0.11, 0.33],
        ]
    )

    loss_fn = MultiPosConLoss()

    good_loss = loss_fn({"feats": good_embedding, "labels": labels})

    # classes are similar, so the loss should be high
    bad_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.21, 0.21, 0.21],
            [0.22, 0.21, 0.23],
        ]
    )
    bad_loss = loss_fn({"feats": bad_embedding, "labels": labels})
    assert good_loss < bad_loss


def test_nce_loss_with_masking():
    labels = torch.tensor([0, 0, 2, 2])
    # two classes are clustered and separated
    good_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.8, 0.1, 0.3],
            [0.82, 0.11, 0.33],
        ]
    )

    loss_fn = NCELoss(temperature=0.07)

    good_loss = loss_fn(
        predicted_embeddings=good_embedding,
        labels=labels,
        contrastive_embeddings=good_embedding,
        contrastive_labels=labels,
        mask_diagonal=True,
    )

    # classes are similar, so the loss should be high
    bad_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.21, 0.21, 0.21],
            [0.22, 0.21, 0.23],
        ]
    )
    bad_loss = loss_fn(
        predicted_embeddings=bad_embedding,
        labels=labels,
        contrastive_embeddings=bad_embedding,
        contrastive_labels=labels,
        mask_diagonal=True,
    )
    assert good_loss < bad_loss


def test_nce_loss_without_masking():
    labels = torch.tensor([0, 0, 2, 2])

    # make the contrastive embeddings/labels
    contrastive_embeddings = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.44, 0.42, 0.43],
            [0.21, 0.21, 0.21],
            [0.81, 0.09, 0.31],
            [0.82, 0.11, 0.33],
            [0.22, 0.20, 0.88],
        ]
    )
    contrastive_labels = torch.tensor([0, 0, 0, 2, 2, 3])

    # two classes are clustered and separated
    good_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.8, 0.1, 0.3],
            [0.82, 0.11, 0.33],
        ]
    )

    loss_fn = NCELoss(temperature=0.07)

    good_loss = loss_fn(
        predicted_embeddings=good_embedding,
        labels=labels,
        contrastive_embeddings=contrastive_embeddings,
        contrastive_labels=contrastive_labels,
        mask_diagonal=False,
    )

    # classes are similar, so the loss should be high
    bad_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.21, 0.21, 0.21],
            [0.22, 0.21, 0.23],
        ]
    )
    bad_loss = loss_fn(
        predicted_embeddings=bad_embedding,
        labels=labels,
        contrastive_embeddings=contrastive_embeddings,
        contrastive_labels=contrastive_labels,
        mask_diagonal=False,
    )
    assert good_loss < bad_loss
