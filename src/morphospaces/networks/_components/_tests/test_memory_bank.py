import torch
import torch.nn.functional as F

from morphospaces.networks._components.memory_bank import PixelMemoryBank


def test_pixel_memory_bank():
    n_embeddings_per_class = 10
    n_embeddings_to_update = 2
    n_dimensions = 3
    label_values = [0, 1, 4]
    n_label_values = len(label_values)
    memory_bank = PixelMemoryBank(
        n_embeddings_per_class=n_embeddings_per_class,
        n_embeddings_to_update=n_embeddings_to_update,
        n_dimensions=n_dimensions,
        label_values=label_values,
    )

    assert memory_bank.embeddings.shape == (
        n_label_values,
        n_embeddings_per_class,
        n_dimensions,
    )

    # should be initialized with all nans
    assert torch.all(torch.isnan(memory_bank.embeddings))

    # test setting some embeddings
    new_embeddings = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.3, 0.3],
            [0.2, 0.2, 0.4],
        ]
    )
    label_to_set = 4
    labels = torch.tensor([label_to_set] * new_embeddings.shape[0])
    memory_bank.set_embeddings(new_embeddings, labels)
    retrieved_embeddings, retrieved_labels = memory_bank.get_embeddings()

    # todo add tests for actual value
    assert retrieved_embeddings.shape == (n_embeddings_to_update, n_dimensions)
    assert torch.all(
        retrieved_labels == torch.tensor([label_to_set, label_to_set])
    )


def test_pixel_memory_bank_overflow():
    """test that when the memory bank is full,
    the buffer correctly wraps around."""
    n_embeddings_per_class = 3
    n_embeddings_to_update = 2
    n_dimensions = 2
    label_values = [0, 3]
    n_label_values = len(label_values)
    memory_bank = PixelMemoryBank(
        n_embeddings_per_class=n_embeddings_per_class,
        n_embeddings_to_update=n_embeddings_to_update,
        n_dimensions=n_dimensions,
        label_values=label_values,
    )

    # set the memory bank directly such that it is full
    # shape (n_labels, n_embeddings_per_class, n_dimensions)
    stored_embeddings = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[0.4, 0.5], [0.5, 0.5], [0.6, 0.7]],
        ]
    )
    memory_bank._embeddings = torch.tensor(stored_embeddings)
    assert memory_bank.embeddings.shape == (
        n_label_values,
        n_embeddings_per_class,
        n_dimensions,
    )

    # set the index such that the next set would overflow
    memory_bank._current_index = torch.tensor([1, 1])

    # add new embeddings
    new_embeddings = F.normalize(
        torch.tensor(
            [
                [0.5, 0.2],
                [0.3, 0.6],
                [0.7, 0.9],
                [0.5, 0.1],
            ]
        ),
        p=2,
        dim=1,
    )
    new_labels = torch.tensor([3, 3, 0, 0])
    memory_bank.set_embeddings(new_embeddings, new_labels)

    # check the embeddings were set correctly
    # (should be in the end of the queue)
    expected_embeddings = torch.zeros((6, n_dimensions))
    expected_embeddings[0, :] = stored_embeddings[0, 0, :]
    expected_embeddings[1:3, :] = new_embeddings[2:4, :]
    expected_embeddings[3, :] = stored_embeddings[1, 0, :]
    expected_embeddings[4:6, :] = new_embeddings[0:2, :]

    retrieved_embeddings, retrieved_labels = memory_bank.get_embeddings()

    # todo add a test that checks the values are correct
    assert retrieved_embeddings.shape == (
        2 * n_embeddings_per_class,
        n_dimensions,
    )

    # check the labels were returned correctly
    expected_labels = torch.tensor(3 * [0] + 3 * [3])
    assert torch.all(retrieved_labels == expected_labels)
