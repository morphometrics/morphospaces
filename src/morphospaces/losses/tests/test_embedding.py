import torch

from morphospaces.losses.embedding import MultiPosConLoss


def test_multi_pos_con_loss():
    labels = torch.tensor([0, 0, 2, 2])
    good_embedding = torch.tensor(
        [
            [0.2, 0.2, 0.2],
            [0.22, 0.21, 0.23],
            [0.8, 0.8, 0.8],
            [0.82, 0.81, 0.83],
        ]
    )

    loss_fn = MultiPosConLoss()

    good_loss = loss_fn({"feats": good_embedding, "labels": labels})

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
