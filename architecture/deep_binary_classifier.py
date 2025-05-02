# deep_binary_classifier.py
"""
Feed-forward Boolean network with deterministic, multiprocessing-friendly training.

Key points
----------
* Parallelises node construction via `concurrent.futures.ProcessPoolExecutor`.
* All randomness flows through **one** NumPy Generator (`self._rng`); child nodes receive
  private seeds so results are reproducible no matter how many worker processes you launch.
* Works with any `node_factory` that follows the signature
  `(X_cols, X_node, y_node, seed)`.
"""

from __future__ import annotations
from typing import Sequence, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class BaseNode(ABC):
    def __init__(self, X_cols: np.ndarray):
        self.X_cols = X_cols

    # the node chooses the columns on its own in the forward pass
    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        ...


class DeepBinaryClassifier:
    """
    Parameters
    ----------
    nodes_per_layer : list[int]
    bits_per_node   : list[int]   (must be `len(nodes_per_layer)+1`)
    node_factory    : callable    -> returns a `BaseNode` instance
    rng             : None | int
    n_jobs          : None | int  (None or 1 → no multiprocessing)
    """

    def __init__(
            self,
            nodes_per_layer: Sequence[int],
            bits_per_node:   Sequence[int],
            *,
            node_factory: Callable[..., BaseNode],
            rng: int | None = None,   ####not sure whether this seed or rng
            n_jobs: int | None = None,
    ):
        if len(bits_per_node) != len(nodes_per_layer) + 1:
            raise ValueError("bits_per_node must be one longer than nodes_per_layer")

        self.nodes_per_layer = list(nodes_per_layer)
        self.bits_per_node   = list(bits_per_node)

        self.node_factory = node_factory
        self._rng         = np.random.default_rng(rng)
        self.n_jobs       = n_jobs
        self.layers: List[List[BaseNode]] = []

    def _build_layer(
            self,
            X: np.ndarray,
            y: np.ndarray,
            width: int,
            bits: int,
            use_multiprocessing: bool,
    ) -> List[BaseNode]:
        cols_arr = self._rng.choice(X.shape[1], size=(width, bits), replace=True)
        seeds    = self._rng.integers(0, 2**32 - 1, size=width, dtype=np.uint64)

        if not use_multiprocessing:
            nodes: List[BaseNode] = []
            for X_cols, seed in zip(cols_arr, seeds):
                node = self.node_factory( X_cols, X[:, X_cols], y, int(seed))
                nodes.append(node)
            return nodes

        with ProcessPoolExecutor(self.n_jobs) as ex:
            futures = []
            for X_cols, seed in zip(cols_arr, seeds):
                future = ex.submit(self.node_factory, X_cols, X[:, X_cols], y, int(seed))
                futures.append(future)

            # first we start all the workers and then we await the result of each task
            nodes = [future.result() for future in futures]
            return nodes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        layer_out = X
        use_mp    = self.n_jobs is not None and self.n_jobs > 1

        # build all layers (last layer has width=1)
        for width, bits in zip(self.nodes_per_layer + [1], self.bits_per_node):
            nodes = self._build_layer(layer_out, y, width, bits, use_mp)
            self.layers.append(nodes)
            layer_out = np.column_stack([n(layer_out) for n in nodes])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        out = X
        for layer in self.layers:
            out = np.column_stack([n(out) for n in layer])
        return out[:, 0]