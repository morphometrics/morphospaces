import torch

from morphospaces.losses.embedding import MultiPosConLoss


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
