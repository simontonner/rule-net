from __future__ import annotations

import warnings
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
    def __init__(self, X_bits, y_pm1, bits, cols, rng,
                 tie_break="random", *, ripper_kwargs=None):
        super().__init__(bits, cols)

        # --- 0. single-class shortcut ---------------------------------
        if (y_pm1 > 0).all() or (y_pm1 <= 0).all():
            const_true = (y_pm1 > 0).all()
            self.rules = [f"RULE: always_{'true' if const_true else 'false'}"]
            self.lut   = np.full(2**bits, const_true, dtype=bool)
            return

        # --- 1. fit RIPPER (warnings silenced) ------------------------
        names = [f"bit{i}" for i in range(bits)]
        X_df  = pd.DataFrame(X_bits.astype(bool), columns=names)
        y_df  = pd.DataFrame({"y": (y_pm1 > 0)})

        rip = lw.RIPPER(**(ripper_kwargs or {}))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning,
                message=r".*No negative samples.*|.*Ruleset is empty.*")
            rip.fit(X_df, y_df)

        # --- 2. store rules (may be empty) ----------------------------
        self.rules = getattr(rip, "rule_list_", [])

        # --- 3. build LUT from RIPPER’s own predict -------------------
        patterns = np.array(list(itertools.product([False, True], repeat=bits)), bool)
        pat_df   = pd.DataFrame(patterns, columns=names)
        preds    = rip.predict(pat_df).to_numpy(dtype=bool)

        preds = preds.to_numpy(dtype=bool) if hasattr(preds, "to_numpy") else np.asarray(preds, bool)

    # --- 4. optional random tie-break for unseen patterns ---------
        if tie_break == "random":
            seen_idx    = truth_table_indices(X_bits)
            unseen_mask = np.ones(2**bits, bool)
            unseen_mask[seen_idx] = False
            preds[unseen_mask] = rng.integers(0, 2, size=unseen_mask.sum(), dtype=np.uint8)

        self.lut = preds


    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.lut[truth_table_indices(X[:, self.cols])]


# ------------------------------------------------------------------ #
def make_ripper_node(
        X_bits, y_pm1, bits, cols, rng, tie_break, **kw
) -> RipperNode:
    """Factory helper – accepts extra RIPPER kwargs via **kw."""
    return RipperNode(X_bits, y_pm1, bits, cols, rng, tie_break, **kw)
