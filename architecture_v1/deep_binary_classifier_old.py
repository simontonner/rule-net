"""
Core feed-forward Boolean network.
Keeps zero hard-dependencies on the concrete node types.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Callable, List
import numpy as np


# ------------------------------------------------------------------------
class BaseNode(ABC):
    """Interface every node must implement."""

    def __init__(self, bits: int, cols: np.ndarray):
        self.bits = bits
        self.cols = cols  # column indices used as this node’s inputs

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return the Boolean output for each row in X."""
        ...


# ------------------------------------------------------------------------
class DeepBinaryClassifier:
    """
    Generic Boolean network. Concrete behaviour comes from *node_factory*.

    Parameters
    ----------
    nodes_per_layer : e.g. [16, 8]
    bits_per_node   : e.g. [6, 4, 3]  (must be len(nodes_per_layer)+1)
    node_factory    : callable -> BaseNode
                      signature: (X_bits, y_pm1, bits, cols, rng, tie_break)
    tie_break       : forwarded to node_factory (LUT nodes use it)
    reuse_prev_width: if True each layer chooses inputs only from previous layer
    rng             : int or None – shared random seed
    """

    def __init__(
            self,
            nodes_per_layer: Sequence[int],
            bits_per_node: Sequence[int],
            *,
            node_factory: Callable[..., BaseNode],
            tie_break: str = "random",
            reuse_prev_width: bool = True,
            rng: int | None = None,
    ):
        if len(bits_per_node) != len(nodes_per_layer) + 1:
            raise ValueError("bits_per_node must be one longer than nodes_per_layer")
        self.nodes_per_layer = list(nodes_per_layer)
        self.bits_per_node = list(bits_per_node)
        self.node_factory = node_factory
        self.tie_break = tie_break
        self.reuse_prev_width = reuse_prev_width
        self.rng = np.random.default_rng(rng)
        self.layers: List[List[BaseNode]] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")
        y_pm1 = y.astype(np.int8) * 2 - 1  # {-1, +1} for LUT voting
        layer_out = X

        # hidden layers -------------------------------------------------
        for layer_idx, (width, bits) in enumerate(
                zip(self.nodes_per_layer, self.bits_per_node[:-1])
        ):
            pool_size = (
                layer_out.shape[1]
                if (not self.reuse_prev_width or layer_idx == 0)
                else self.nodes_per_layer[layer_idx - 1]
            )
            cols_arr = self.rng.choice(pool_size, size=(width, bits), replace=True)

            nodes = [
                self.node_factory(
                    layer_out[:, cols], y_pm1, bits, cols, self.rng, self.tie_break
                )
                for cols in cols_arr
            ]
            self.layers.append(nodes)
            layer_out = np.column_stack([n(layer_out) for n in nodes])

        # final single node ---------------------------------------------
        bits_last = self.bits_per_node[-1]
        cols_fin = self.rng.choice(layer_out.shape[1], size=bits_last, replace=True)
        fin_node = self.node_factory(
            layer_out[:, cols_fin], y_pm1, bits_last, cols_fin, self.rng, self.tie_break
        )
        self.layers.append([fin_node])
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = np.column_stack([n(out) for n in layer])
        return out[:, 0]
