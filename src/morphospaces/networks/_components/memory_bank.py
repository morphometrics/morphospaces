from typing import Dict, List, Tuple

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
        self._current_index = torch.zeros(
            (self.n_labels,), requires_grad=False
        )

        # pre-allocate the embeddings and initialize
        self._embeddings = torch.full(
            (self.n_labels, self.n_embeddings_per_class, self.n_dimensions),
            torch.nan,
            requires_grad=False,
        )

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings

    @property
    def label_mapping(self) -> Dict[int, int]:
        return self._label_mapping

    @property
    def current_index(self) -> torch.Tensor:
        return self._current_index

    def set_embeddings(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        unique_label_values = torch.unique(labels)

        for unique_label in unique_label_values:
            # Get the index for the label value in the stored embeddings
            label_value = int(unique_label)
            label_index = self.label_mapping[label_value]

            # Get the embeddings to be stored
            embeddings_in_label = embeddings[labels == label_value, :]
            n_embeddings = embeddings_in_label.shape[0]
            random_sampling_indices = torch.randperm(n_embeddings)
            n_embeddings_to_take = min(
                n_embeddings, self.n_embeddings_to_update
            )
            embeddings_to_store = embeddings_in_label[
                random_sampling_indices[:n_embeddings_to_take], :
            ]

            # Determine the start and end indices for storing
            # the embeddings in the memory banks
            starting_index = int(self.current_index[label_index])
            ending_index = starting_index + n_embeddings_to_take

            if ending_index >= self.n_embeddings_per_class:
                self._embeddings[
                    label_index, -n_embeddings_to_take:, :
                ] = torch.nn.functional.normalize(
                    embeddings_to_store, p=2, dim=1
                ).detach()
                self.current_index[label_index] = 0
            else:
                self._embeddings[
                    label_index, starting_index:ending_index, :
                ] = torch.nn.functional.normalize(
                    embeddings_to_store, p=2, dim=1
                ).detach()
                self.current_index[label_index] = (
                    ending_index
                ) % self.n_embeddings_per_class

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.tensor]:
        embeddings = []
        labels = []
        for label_value, label_index in self.label_mapping.items():
            # get the embeddings that are not nan (i.e., not initialized)
            label_embedding = self.embeddings[label_index, :, :]
            nan_rows = torch.any(label_embedding.isnan(), dim=1)
            label_embedding = label_embedding[~nan_rows]
            embeddings.append(label_embedding)

            # get the label values
            n_label_embeddings = label_embedding.shape[0]
            labels.append(
                label_value
                * torch.ones((n_label_embeddings,), dtype=torch.int32)
            )

        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


class LabelMemoryBank:
    """Memory bank to store the mean embedding for each label."""

    def __init__(
        self,
        n_embeddings_per_class: int,
        n_dimensions: int,
        label_values: List[int],
    ):
        # store parameters
        self.n_embeddings_per_class = n_embeddings_per_class
        self.n_dimensions = n_dimensions
        self._label_mapping = {
            value: index for index, value in enumerate(label_values)
        }
        self.n_labels = len(self.label_mapping)

        # current index to start sampling from
        self._current_index = torch.zeros(
            (self.n_labels,), requires_grad=False
        )

        # pre-allocate the embeddings and initialize
        self._embeddings = torch.full(
            (self.n_labels, self.n_embeddings_per_class, self.n_dimensions),
            torch.nan,
            requires_grad=False,
        )

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings

    @property
    def label_mapping(self) -> Dict[int, int]:
        return self._label_mapping

    @property
    def current_index(self) -> torch.Tensor:
        return self._current_index

    def set_embeddings(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """Add new embeddings to the memory bank.

        Parameters
        ----------
        embeddings : torch.Tensor
            (D, z, y, x) array containing the embeddings to store.
            This should be the output of the embeddings prediction.

        labels : torch.Tensor
            (D, z, y, x) array containing the labels
        """
        unique_label_values = torch.unique(labels)

        for unique_label in unique_label_values:
            # Get the index for the label value in the stored embeddings
            label_value = int(unique_label)
            label_index = self.label_mapping[label_value]

            # Get the embeddings to be stored
            # todo generalize to batch size > 1
            assert embeddings.shape[0] == 1, "batch size must be 1"
            embeddings_in_label = torch.squeeze(embeddings)[
                :, torch.squeeze(labels == label_value)
            ].moveaxis(0, -1)
            normalized_embeddings = torch.nn.functional.normalize(
                embeddings_in_label, p=2, dim=1
            )
            mean_embedding = torch.mean(normalized_embeddings, dim=0)

            # Determine the start and end indices for storing
            # the embeddings in the memory banks
            memory_bank_index = int(self.current_index[label_index])

            self._embeddings[
                label_index, memory_bank_index, :
            ] = torch.nn.functional.normalize(
                mean_embedding, p=2, dim=0
            ).detach()
            self.current_index[label_index] = (
                memory_bank_index + 1
            ) % self.n_embeddings_per_class

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.tensor]:
        """Get all embeddings stored in the memory bank.

        Returns
        -------
        embeddings : torch.Tensor
            (n, d) array containing all embeddings.
        labels : torch.Tensor
            (n,) array containing the corresponding label values.
        """
        embeddings = []
        labels = []
        for label_value, label_index in self.label_mapping.items():
            # get the embeddings that are not nan (i.e., not initialized)
            label_embedding = self.embeddings[label_index, :, :]
            nan_rows = torch.any(label_embedding.isnan(), dim=1)
            label_embedding = label_embedding[~nan_rows]
            embeddings.append(label_embedding)

            # get the label values
            n_label_embeddings = label_embedding.shape[0]
            labels.append(
                label_value
                * torch.ones((n_label_embeddings,), dtype=torch.int32)
            )

        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
