# lut_ripper_node.py
"""
RIPPER-based node that exactly emulates the old ruleset behaviour,
yet plugs into the new multiprocessing DeepBinaryClassifier.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import wittgenstein as lw
from .deep_binary_classifier import BaseNode


def truth_table_indices(bits_bool: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(bits_bool.shape[1] - 1, -1, -1, dtype=np.uint32)
    return (bits_bool.astype(np.uint32) * weights).sum(axis=1).astype(np.int64)


class RipperNode(BaseNode):
    def __init__(self, X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int):
        super().__init__(X_cols)

        rng = np.random.default_rng(seed)
        num_bits = X_node.shape[1]

        y_pm1     = y_node.astype(np.int8) * 2 - 1

        # 1) Train RIPPER using the same seed
        names = [f"bit{i}" for i in range(num_bits)]
        X_df  = pd.DataFrame(X_node.astype(bool), columns=names)
        y_df  = pd.DataFrame({"y": (y_pm1 > 0)})

        # RIPPER might draw the exact same random number as the rng above (which shouldn't matter)
        rip = lw.RIPPER(random_state=seed)
        rip.fit(X_df, y_df)

        # 2) Materialize full truth table and predict
        patterns = np.array(list(itertools.product([False, True], repeat=num_bits)), bool)
        pat_df   = pd.DataFrame(patterns, columns=names)
        preds    = rip.predict(pat_df)
        preds = preds.to_numpy(dtype=bool) if hasattr(preds, "to_numpy") else np.asarray(preds, bool)

        # 3) Tie‐break or fill unseen with fresh RNG draws
        seen_idx    = truth_table_indices(X_node)
        unseen_mask = np.ones(2**num_bits, bool)
        unseen_mask[seen_idx] = False
        preds[unseen_mask] = rng.integers(0, 2, size=unseen_mask.sum(), dtype=np.uint8)

        self.lut = preds

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.X_cols])]


def make_ripper_node(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> RipperNode:
    return RipperNode(X_cols, X_node, y_node, seed)
