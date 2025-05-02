# lut_node.py
"""
Fast majority-vote LUT node with optional random tie-breaking.
"""

from __future__ import annotations
import numpy as np
from .deep_binary_classifier import BaseNode


def truth_table_indices(bits_bool: np.ndarray) -> np.Fndarray:
    """Boolean patterns ➜ integer indices (big-endian)."""
    weights = 1 << np.arange(bits_bool.shape[1] - 1, -1, -1, dtype=np.uint32)
    return (bits_bool.astype(np.uint32) * weights).sum(axis=1).astype(np.int64)


class LutNode(BaseNode):
    def __init__(self, X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int):
        super().__init__(X_cols)

        rng = np.random.default_rng(seed)

        y_pm1     = y_node.astype(np.int8) * 2 - 1

        idxs  = truth_table_indices(X_node)
        num_bits = X_node.shape[1]
        votes = np.bincount(idxs, weights=y_pm1, minlength=2**num_bits)

        ties        = votes == 0
        votes[ties] = rng.choice([-1, 1], size=ties.sum())

        self.lut = votes > 0  # bool LUT

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.X_cols])]


def make_lut_node(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> LutNode:
    return LutNode(X_cols, X_node, y_node, seed)
