from __future__ import annotations
import numpy as np
import pandas as pd
import wittgenstein as lw
from .deep_binary_classifier import BaseNode
from .utils import truth_table_indices, truth_table_patterns

class RipperNode(BaseNode):
    """
    RIPPER-based lookup node.

    Trains a RIPPER rule set on the node's subfeatures, then
    materializes a full truth table for predictions.

    :param X_cols: Indices of the input features used by this node
    :param X_node: Training inputs for this node, shape (N, num_bits)
    :param y_node: Binary training targets, shape (N,)
    :param seed: RNG seed for reproducibility
    """
    def __init__(
            self,
            X_cols: np.ndarray,
            X_node: np.ndarray,
            y_node: np.ndarray,
            seed: int
    ):
        super().__init__(X_cols)

        self.seed = seed
        num_bits = X_node.shape[1]

        # wittgenstein requires DataFrames
        names = [f"x_{i}" for i in X_cols]
        X_df = pd.DataFrame(X_node.astype(bool), columns=names)
        y_df = pd.DataFrame({"y": y_node})

        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df)

        # store the ruleset for later use
        self.ripper = ripper

        # predict full truth table
        patterns = truth_table_patterns(num_bits)
        pat_df = pd.DataFrame(patterns, columns=names)
        pred = ripper.predict(pat_df)

        # since each node is small, we store the prediction for each possible bit-pattern instead of invoking RIPPER
        self.pred_node = np.asarray(pred, dtype=bool)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        idxs = truth_table_indices(X[:, self.X_cols])
        return self.pred_node[idxs]


def make_ripper_node(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> RipperNode:
    """
    Simple node factory to be hand over to the `DeepBinaryClassifier`.
    """
    return RipperNode(X_cols, X_node, y_node, seed)
