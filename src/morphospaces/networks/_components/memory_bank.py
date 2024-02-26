from typing import Dict, List

import torch


class PixelMemoryBank:
    """Store embeddings for later comparison across different images.

    Parameters
    ----------
    n_embeddings_per_class : int
        The total number of embeddings to store for each class.
    n_embeddings_to_update: int
        The number of embeddings to store per update.
    n_dimensions : int
        The number of dimensions for each embedding vector.
    """

    def __init__(
        self,
        n_embeddings_per_class: int,
        n_embeddings_to_update: int,
        n_dimensions: int,
        label_values: List[int],
    ):
        # store parameters
        self.n_embeddings_per_class = n_embeddings_per_class
        self.n_embeddings_to_update = n_embeddings_to_update
        self.n_dimensions = n_dimensions
        self._label_mapping = {
            value: index for index, value in enumerate(label_values)
        }
        self.n_labels = len(self.label_mapping)

        # current index to start sampling from
        self.current_index = torch.zeros((self.n_labels,))

        # pre-allocate the embeddings and initialize
        self._embeddings = torch.zeros(
            (self.n_labels, self.n_embeddings_per_class, self.n_dimensions)
        )
        self._initialize_memory_bank()

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings

    @property
    def label_mapping(self) -> Dict[int, int]:
        return self._label_mapping

    def _initialize_memory_bank(self):
        pass

    def set_embeddings(
        self, embeddings: torch.Tensor, labels: torch.Tensor, label_value: int
    ) -> None:
        # Get the index for the label value in the stored embeddings
        label_index = self.label_mapping[label_value]

        # Get the embeddings to be stored
        embeddings_in_label = embeddings[labels == label_value, :]
        n_embeddings = embeddings_in_label.shape[0]
        random_sampling_indices = torch.randperm(n_embeddings)
        n_embeddings_to_take = min(n_embeddings, self.n_embeddings_to_update)
        embeddings_to_store = embeddings_in_label[
            random_sampling_indices[:n_embeddings_to_take], :
        ]

        # Determine the start and end indices for storing
        # the embeddings in the memory banks
        starting_index = self.current_index[label_index]
        ending_index = starting_index + n_embeddings_to_take

        if ending_index >= self.n_embeddings_per_class:
            self.embeddings[
                label_index, -n_embeddings:, :
            ] = torch.nn.functional.normalize(embeddings_to_store, p=2, dim=1)
            self.current_index[label_index] = 0
        else:
            self.embeddings[
                label_index, starting_index:ending_index, :
            ] = torch.nn.functional.normalize(embeddings_to_store, p=2, dim=1)
            self.current_index[label_index] = (
                ending_index
            ) % self.n_embeddings_per_class

    def get_embeddings(self) -> torch.Tensor:
        pass
