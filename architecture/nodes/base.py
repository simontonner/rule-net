from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from architecture.utils import truth_table_indices, truth_table_patterns


class BinaryNode(ABC):
    def __init__(self, node_name: str, input_names: list[str]):
        """
        Base class for all nodes in the network.

        :param node_name: The name of this node.
        :param input_names: The names of the input values (features).

        Note: The node_name and input_names are used for wiring up the network graph.
        """
        self.name = node_name
        self.input_names = input_names

        self.node_predictions = None



    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        """
        Returns predictions for the given input vales.

        :param input_values: The input values, shape (N, num_bits)
        :return: The predictions, shape (N,)
        """
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")
        if input_values.shape[1] != len(self.input_names):
            raise ValueError(f"Node {self.name} accepts only inputs of length {len(self.input_names)}")
        return self.node_predictions[truth_table_indices(input_values)]

    def get_truth_table(self) -> tuple[np.ndarray, list[str]]:
        """
        Return the truth table of this node.

        :return:
            table: Complete truth table of this node in the typical format, shape (2**num_bits, num_bits + 1)
            column_names: Column headers to be used when displaying the table, shape (num_bits + 1,)

        Note: Predictions must be precomputed and stored in `self.node_predictions` by the node implementation.
        """
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")
        patterns = truth_table_patterns(len(self.input_names))
        table = np.column_stack((patterns, self.node_predictions))
        column_names = self.input_names + [f"{self.name} (output)"]
        return table, column_names

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Returns metadata specific to this node type.

        :return: A dictionary containing node-specific metadata (e.g., node_type, seed, expression)
        """
        ...