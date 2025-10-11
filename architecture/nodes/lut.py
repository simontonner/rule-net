from __future__ import annotations
import numpy as np

from architecture.nodes.base import BinaryNode
from architecture.utils import truth_table_indices


class LutNode(BinaryNode):
    def __init__(
            self,
            node_name: str,
            input_names: list[str],
            input_values: np.ndarray[np.bool_],
            target_values: np.ndarray[np.bool_],
            seed: int,
    ):
        """
        Simple Lookup Table (LUT) node using majority voting on training data.

        Unseen patterns and voting ties are resolved randomly but reproducibly using `seed`.

        :param node_name: The name of this node.
        :param input_names: The names of the input values in the column order of input_values.
        :param input_values: The input values for this node, shape (N, num_bits)
        :param target_values: The target values for this node, shape (N,)
        :param seed: A random seed for reproducibility.
        """
        super().__init__(node_name, input_names)
        self.seed = seed
        rng = np.random.default_rng(seed)

        if input_values.dtype != bool or target_values.dtype != bool:
            raise TypeError("input_values and target_values must be boolean arrays")

        # +1 vote for True, -1 for False
        target_plus_minus = target_values.astype(np.int8) * 2 - 1
        pattern_indices = truth_table_indices(input_values)

        votes = np.bincount(pattern_indices, weights=target_plus_minus, minlength=2 ** input_values.shape[1])

        zero_mask = votes == 0
        if zero_mask.any():
            votes[zero_mask] = rng.integers(0, 2, size=zero_mask.sum()) * 2 - 1

        self.node_predictions = votes > 0

    def get_metadata(self) -> dict:
        return {"type": "lut", "seed": self.seed}


def make_lut_node(
        node_name: str,
        input_names: list[str],
        input_values: np.ndarray[np.bool_],
        target_values: np.ndarray[np.bool_],
        seed: int,
) -> LutNode:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return LutNode(node_name, input_names, input_values, target_values, seed)