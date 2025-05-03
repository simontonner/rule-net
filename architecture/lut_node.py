from __future__ import annotations
import numpy as np
from .deep_binary_classifier import BaseNode
from .utils import truth_table_patterns, truth_table_indices


class LutNode(BaseNode):
    def __init__(self, X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int):
        """
        Simple Lookup Table (LUT) node using majority voting on training data.

        Uses random predictions for unseen bit-patterns or voting ties.

        :param X_cols: The columns of the input data that this node uses, shape (num_bits,)
        :param X_node: The input data for this node, shape (N, num_bits)
        :param y_node: The target data for this node, shape (N,)
        :param seed: Random seed for reproducibility
        """
        super().__init__(X_cols)

        self.seed = seed
        rng = np.random.default_rng(seed)

        # the voting counts +1 for True and -1 for False
        y_plus_minus = y_node.astype(np.int8) * 2 - 1
        pattern_indices = truth_table_indices(X_node)
        votes = np.bincount(pattern_indices, weights=y_plus_minus, minlength=2**X_node.shape[1])

        # ties and missing patterns result in 0 and are set to a random choice
        mask = votes == 0
        votes[mask] = rng.choice([-1, 1], size=mask.sum())

        # since each node is small, we store the prediction for each possible bit-pattern
        self.pred_node = votes > 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        idxs = truth_table_indices(X[:, self.X_cols])
        return self.pred_node[idxs]

    def get_truth_table(self) -> np.ndarray:
        """
        Generates the full truth-table on-demand since storing it in each node would be wasteful.

        :return: Boolean array, shape (2**num_bits, num_bits + 1)
        """
        patterns = truth_table_patterns(len(self.X_cols))
        return np.column_stack((patterns, self.pred_node))


def make_lut_node(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> LutNode:
    """
    Simple node factory to be hand over to the `DeepBinaryClassifier`.
    """
    return LutNode(X_cols, X_node, y_node, seed)
