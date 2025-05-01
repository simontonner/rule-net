"""
Fast majority-vote LUT node with optional random tie-breaking.
"""

from __future__ import annotations
import numpy as np
from numpy.random import Generator
from .deep_binary_classifier import BaseNode


# --------------------------------------------------------------------- #
def truth_table_indices(bits_bool: np.ndarray) -> np.ndarray:
    """Boolean patterns ➜ integer indices (big-endian)."""
    weights = 1 << np.arange(bits_bool.shape[1] - 1, -1, -1, dtype=np.uint32)
    return (bits_bool.astype(np.uint32) * weights).sum(axis=1).astype(np.int64)


class LutNode(BaseNode):
    def __init__(
            self,
            X_bits:   np.ndarray,
            y_pm1:    np.ndarray,
            bits:     int,
            cols:     np.ndarray,
            rng:      Generator,
            tie_break:str = "random",
    ):
        super().__init__(bits, cols)

        idxs  = truth_table_indices(X_bits)
        votes = np.bincount(idxs, weights=y_pm1, minlength=2**bits)

        if tie_break == "random":
            ties       = votes == 0
            votes[ties] = rng.choice([-1, 1], size=ties.sum())
        elif tie_break != "zero":
            raise ValueError("tie_break must be 'random' or 'zero'")

        self.lut = votes > 0            # bool LUT

    # --------------------------------------------------------------- #
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.cols])]


# ------------------------------------------------------------------ #
def make_lut_node(
        X_bits, y_pm1, bits, cols, rng, tie_break, **unused
) -> LutNode:
    """Factory helper matching the required signature."""
    return LutNode(X_bits, y_pm1, bits, cols, rng, tie_break)
