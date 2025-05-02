# deep_binary_classifier.py
"""
Feed-forward Boolean network with deterministic, multiprocessing-friendly training.

Key points
----------
* Parallelises node construction via `concurrent.futures.ProcessPoolExecutor`.
* All randomness flows through **one** NumPy Generator (`self._rng`); child nodes receive
  private seeds so results are reproducible no matter how many worker processes you launch.
* Works with any `node_factory` that follows the signature
  `(X_bits, y_pm1, bits, cols, rng, tie_break)`.
"""

from __future__ import annotations
import os
from typing import Sequence, Callable, List
from abc    import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future


# --------------------------------------------------------------------- #
class BaseNode(ABC):
    def __init__(self, bits: int, cols: np.ndarray):
        self.bits, self.cols = bits, cols

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
# --------------------------------------------------------------------- #


class DeepBinaryClassifier:
    """
    Parameters
    ----------
    nodes_per_layer : list[int]
    bits_per_node   : list[int]   (must be `len(nodes_per_layer)+1`)
    node_factory    : callable    -> returns a `BaseNode` instance
    rng             : None | int
    n_jobs          : None | int  (worker processes; None → all cores)
    """

    def __init__(
            self,
            nodes_per_layer: Sequence[int],
            bits_per_node:   Sequence[int],
            *,
            node_factory: Callable[..., BaseNode],
            rng:            int | None = None,
            n_jobs:         int | None = None,
    ):
        if len(bits_per_node) != len(nodes_per_layer) + 1:
            raise ValueError("bits_per_node must be one longer than nodes_per_layer")

        self.W = list(nodes_per_layer)
        self.B = list(bits_per_node)

        self.node_factory = node_factory
        self._rng         = np.random.default_rng(rng)
        self.n_jobs       = n_jobs or (os.cpu_count() or 1)

        self.layers: List[List[BaseNode]] = []

    # ----------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        y_pm1     = y.astype(np.int8) * 2 - 1
        layer_out = X

        with ProcessPoolExecutor(self.n_jobs) as ex:
            for width, bits in zip(self.W, self.B[:-1]):
                pool     = layer_out
                cols_arr = self._rng.choice(
                    pool.shape[1], size=(width, bits), replace=True
                )
                seeds    = self._rng.integers(0, 2**32 - 1, size=width, dtype=np.uint64)

                # submit one future per node
                futures: List[Future[BaseNode]] = []
                for cols, seed in zip(cols_arr, seeds):
                    rng_node = np.random.default_rng(int(seed))
                    Xb = pool[:, cols]
                    futures.append(
                        ex.submit(
                            self.node_factory,
                            Xb,
                            y_pm1,
                            bits,
                            cols,
                            rng_node,
                        )
                    )

                nodes = [f.result() for f in futures]
                self.layers.append(nodes)
                layer_out = np.column_stack([n(pool) for n in nodes])

            # final node (serial – only one)
            bits_last = self.B[-1]
            cols_fin  = self._rng.choice(layer_out.shape[1], size=bits_last, replace=True)
            seed_fin  = int(self._rng.integers(0, 2**32 - 1))
            rng_fin   = np.random.default_rng(seed_fin)

            fin_node = self.node_factory(
                layer_out[:, cols_fin],
                y_pm1,
                bits_last,
                cols_fin,
                rng_fin,
            )
            self.layers.append([fin_node])

        return self

    # ----------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        out = X
        for layer in self.layers:
            out = np.column_stack([n(out) for n in layer])
        return out[:, 0]
