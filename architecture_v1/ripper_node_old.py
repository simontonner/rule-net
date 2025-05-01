"""
Rule-based node.
In lieu of the wittgenstein RIPPER package we approximate with a
depth-unlimited decision-tree and then freeze its predictions into a LUT.
"""

from __future__ import annotations
import itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .deep_binary_classifier import BaseNode
from .lut_node import truth_table_indices


# ----------------------------------------------------------------------
class RipperNode(BaseNode):
    def __init__(
            self,
            X_bits: np.ndarray,
            y_pm1: np.ndarray,           # we convert back to {0,1}
            bits: int,
            cols: np.ndarray,
            *_,
    ):
        super().__init__(bits, cols)
        y_bool = y_pm1 > 0

        # train a tiny decision tree as stand-in for RIPPER
        clf = DecisionTreeClassifier(max_depth=None, random_state=0)
        clf.fit(X_bits.astype(int), y_bool)
        self.clf = clf

        # materialise its truth-table once for O(1) look-ups later
        patterns = np.array(
            list(itertools.product([False, True], repeat=bits)), dtype=bool
        )
        self.lut = clf.predict(patterns.astype(int)).astype(bool)

    # ------------------------------------------------------------------
    def __call__(self, X: np.ndarray) -> np.ndarray:
        sub = X[:, self.cols]
        return self.lut[truth_table_indices(sub)]


# factory helper --------------------------------------------------------
def make_ripper_node(X_bits, y_pm1, bits, cols, *_unused):
    return RipperNode(X_bits, y_pm1, bits, cols)
