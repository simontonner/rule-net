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

        # wittgenstein requires DataFrames
        names = [f"x_{i}" for i in X_cols]
        X_df = pd.DataFrame(X_node.astype(bool), columns=names)
        y_df = pd.DataFrame({"y": y_node})

        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df, pos_class=True)

        # store the ruleset for later use
        self.ruleset = ripper.ruleset_
        self.ripper = ripper

        # the new rule might rely on fewer columns
        names = ripper.selected_features_
        self.X_cols = np.array([int(n.split('_')[1]) for n in names], dtype=int)

        # predict full truth table (over the reduced bit‐width)
        patterns = truth_table_patterns(len(self.X_cols))
        pat_df = pd.DataFrame(patterns, columns=names)
        pred = ripper.predict(pat_df)

        # since each node is small, we store the prediction for each possible bit-pattern instead of invoking RIPPER
        self.pred_node = np.asarray(pred, dtype=bool)

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

    def get_ruleset(self, disjunction_str = 'V') -> str:
        """
        Returns the ruleset in DNF format.

        :param disjunction_str: Allows to specify a different disjunction string for nicer formatting.
        :return: String representation of the ruleset
        """
        if len(self.ruleset) == 0:
            return "No rules found! Defaulting to False for every input."

        rule_str = disjunction_str.join(str(rule) for rule in self.ruleset)
        return rule_str



def make_ripper_node(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> RipperNode:
    """
    Simple node factory to be hand over to the `DeepBinaryClassifier`.
    """
    return RipperNode(X_cols, X_node, y_node, seed)
