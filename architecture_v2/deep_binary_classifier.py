# architecture/deep_binary_classifier.py
"""
Feed-forward Boolean network – refactor with two critical fixes:

1.  *reuse_prev_width* now defaults to **False** so every layer can still
    “see” all original input bits in addition to the previous layer’s outputs.
2.  The tie-break strategy requested by the nodes is forwarded untouched.
"""

from __future__ import annotations
from typing   import Sequence, Callable, List
from abc      import ABC, abstractmethod
import numpy as np


class BaseNode(ABC):
    def __init__(self, bits: int, cols: np.ndarray):
        self.bits, self.cols = bits, cols

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
# ---------------------------------------------------------------------


class DeepBinaryClassifier:
    def __init__(
            self,
            nodes_per_layer: Sequence[int],
            bits_per_node:   Sequence[int],
            *,
            node_factory:      Callable[..., BaseNode],
            tie_break:         str = "random",
            reuse_prev_width:  bool = False,     # ← fix #1
            rng:               int | None = None,
    ):
        if len(bits_per_node) != len(nodes_per_layer) + 1:
            raise ValueError("bits_per_node must be one longer than nodes_per_layer")

        self.nodes_per_layer = list(nodes_per_layer)
        self.bits_per_node   = list(bits_per_node)
        self.node_factory    = node_factory
        self.tie_break       = tie_break
        self.reuse_prev_width= reuse_prev_width
        self.rng             = np.random.default_rng(rng)
        self.layers: List[List[BaseNode]] = []

    # -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        y_pm1     = y.astype(np.int8) * 2 - 1        # {-1,+1}
        layer_out = X                                # start with raw bits

        for layer_idx, (width, bits) in enumerate(
                zip(self.nodes_per_layer, self.bits_per_node[:-1])
        ):
            # ---- candidate-pool: raw inputs ∪ previous layer -----------
            pool = (
                layer_out                             # previous layer
                if (self.reuse_prev_width or layer_idx == 0)
                else np.column_stack((X, layer_out))  # ← raw + prev
            )
            cols_arr = self.rng.choice(
                pool.shape[1], size=(width, bits), replace=True
            )

            nodes = [
                self.node_factory(
                    pool[:, cols], y_pm1, bits, cols,
                    self.rng, self.tie_break           # ← forward tie_break
                )
                for cols in cols_arr
            ]
            self.layers.append(nodes)
            layer_out = np.column_stack([n(pool) for n in nodes])

        # ---- final single node ----------------------------------------
        bits_last = self.bits_per_node[-1]
        cols_fin  = self.rng.choice(layer_out.shape[1], size=bits_last, replace=True)
        fin_node  = self.node_factory(
            layer_out[:, cols_fin], y_pm1, bits_last, cols_fin,
            self.rng, self.tie_break
        )
        self.layers.append([fin_node])
        return self

    # -----------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        raw       = X               # keep original bits around
        layer_out = X               # start with raw as “previous layer”

        for layer_idx, layer in enumerate(self.layers):
            # exactly the same pool logic as in fit()
            if self.reuse_prev_width or layer_idx == 0:
                pool = layer_out
            else:
                pool = np.column_stack((raw, layer_out))

            # call each node on that pool
            layer_out = np.column_stack([n(pool) for n in layer])

        # final network output is the first (and only) column
        return layer_out[:, 0]
