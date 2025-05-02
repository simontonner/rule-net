

"""
RIPPER-based node that exactly emulates the old ruleset behaviour,
yet plugs into the new multiprocessing `DeepBinaryClassifierMP`.
"""

from __future__ import annotations
import itertools
import warnings
import numpy as np
import pandas as pd
import wittgenstein as lw            # RIPPER implementation
from numpy.random import Generator
from .deep_binary_classifier import BaseNode

# --------------------------------------------------------------------- #
def truth_table_indices(bits_bool: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(bits_bool.shape[1] - 1, -1, -1, dtype=np.uint32)
    return (bits_bool.astype(np.uint32) * weights).sum(axis=1).astype(np.int64)


class RipperNode(BaseNode):
    """
    Parameters
    ----------
    X_bits, y_pm1, bits, cols : like `LutNode`
    rng        : NumPy Generator used solely for reproducibility and tie-breaks
    tie_break  : "random" | "zero"
    """

    def __init__(
            self,
            X_bits: np.ndarray,
            y_pm1: np.ndarray,
            bits: int,
            cols: np.ndarray,
            rng: Generator,
            tie_break: str = "random",
    ):
        super().__init__(bits, cols)

        # Prepare DataFrame
        names = [f"bit{i}" for i in range(bits)]
        X_df = pd.DataFrame(X_bits.astype(bool), columns=names)
        y_df = pd.DataFrame({"y": (y_pm1 > 0)})

        # Seed RIPPER for reproducibility
        seed = int(rng.integers(0, 2**31 - 1))
        rip = lw.RIPPER(random_state=seed)

        # Suppress single-class and empty-ruleset warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r".*No negative samples.*|.*Ruleset is empty.*"
            )
            rip.fit(X_df, y_df)

        # Materialize full truth-table predictions
        patterns = np.array(list(itertools.product([False, True], repeat=bits)), bool)
        pat_df = pd.DataFrame(patterns, columns=names)
        preds_ser = rip.predict(pat_df)
        preds = preds_ser.to_numpy(dtype=bool) if hasattr(preds_ser, 'to_numpy') else np.asarray(preds_ser, bool)

        # Handle unseen patterns with rng
        if tie_break == 'random':
            seen_idx = truth_table_indices(X_bits)
            unseen_mask = np.ones(2**bits, bool)
            unseen_mask[seen_idx] = False
            preds[unseen_mask] = rng.integers(0, 2, size=unseen_mask.sum(), dtype=np.uint8)

        self.lut = preds
        # Store learned rules for introspection
        self.rules = getattr(rip, 'rule_list_', [])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.cols])]


# ------------------------------------------------------------------ #
def make_ripper_node(
        X_bits, y_pm1, bits, cols, rng, tie_break
) -> RipperNode:
    """Factory helper – no extra kwargs, defaults suffice."""
    return RipperNode(X_bits, y_pm1, bits, cols, rng, tie_break)