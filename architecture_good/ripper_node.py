"""
RIPPER-based node that exactly emulates the old ruleset behaviour,
yet plugs into the new multiprocessing `DeepBinaryClassifierMP`.
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import wittgenstein as lw            # ↳ RIPPER implementation
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
    rng        : NumPy Generator used solely for random tie-breaks
    tie_break  : "random" | "zero"
    ripper_kwargs : forwarded to `lw.RIPPER(...)`
    """

    def __init__(
            self,
            X_bits: np.ndarray,
            y_pm1:  np.ndarray,
            bits:   int,
            cols:   np.ndarray,
            rng:    Generator,
            tie_break: str = "random",
            *,
            ripper_kwargs: dict | None = None,
    ):
        super().__init__(bits, cols)

        # ---- 1. Train classic RIPPER on a boolean DataFrame ----------
        names  = [f"bit{i}" for i in range(bits)]
        X_df   = pd.DataFrame(X_bits.astype(bool), columns=names)
        y_df   = pd.DataFrame({"y": (y_pm1 > 0)})

        rip = lw.RIPPER(**(ripper_kwargs or {}))
        rip.fit(X_df, y_df)

        # ---- 2. Materialise full truth table -------------------------
        patterns = np.array(list(itertools.product([False, True], repeat=bits)), bool)
        pat_df   = pd.DataFrame(patterns, columns=names)
        preds    = rip.predict(pat_df)

        # robust → ndarray[bool]
        preds = preds.to_numpy(dtype=bool) if hasattr(preds, "to_numpy") else np.asarray(preds, bool)

        # ---- 3. Deal with unseen patterns ----------------------------
        if tie_break == "random":
            seen_idx     = truth_table_indices(X_bits)
            unseen_mask  = np.ones(2**bits, bool)
            unseen_mask[seen_idx] = False
            preds[unseen_mask] = rng.integers(0, 2, size=unseen_mask.sum(), dtype=np.uint8)

        self.lut = preds.astype(bool)

    # ---------------------------------------------------------------- #
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.cols])]


# ------------------------------------------------------------------ #
def make_ripper_node(
        X_bits, y_pm1, bits, cols, rng, tie_break, **kw
) -> RipperNode:
    """Factory helper – accepts extra RIPPER kwargs via **kw."""
    return RipperNode(X_bits, y_pm1, bits, cols, rng, tie_break, **kw)
