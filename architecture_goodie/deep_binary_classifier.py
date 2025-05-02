# deep_binary_classifier.py
"""
Feed-forward Boolean network with deterministic, multiprocessing-friendly training.

Key points
----------
* `DeepBinaryClassifierMP` parallelises node construction via
  `concurrent.futures.ProcessPoolExecutor`.
* All randomness flows through **one** NumPy Generator (`self._rng`);
  child nodes receive private seeds so results are reproducible
  no matter how many worker processes you launch.
* Works with any `node_factory` that follows the signature
  `(X_bits, y_pm1, bits, cols, rng, tie_break, **extra)`.
"""

from __future__ import annotations
import os
from typing import Sequence, Callable, List
from abc    import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# --------------------------------------------------------------------- #
class BaseNode(ABC):
    def __init__(self, bits: int, cols: np.ndarray):
        self.bits, self.cols = bits, cols

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
# --------------------------------------------------------------------- #


def _build_node(args):
    node_factory, X_bits, y_pm1, bits, cols, seed, tie_break, extra_kw = args
    rng = np.random.default_rng(int(seed))
    # no more ripper_kwargs handling here
    return node_factory(X_bits, y_pm1, bits, cols, rng, tie_break, **extra_kw)



class DeepBinaryClassifier:
    """
    Parameters
    ----------
    nodes_per_layer : list[int]
    bits_per_node   : list[int]   (must be `len(nodes_per_layer)+1`)
    node_factory    : callable    -> returns a `BaseNode` instance
    tie_break       : "random" | "zero"
    reuse_prev_width: see paper – when False each layer sees *raw+prev*
    rng             : None | int
    n_jobs          : None | int  (worker processes; None → all cores)
    extra_node_kw   : dict        (passed verbatim to every node_factory)
    """

    def __init__(
            self,
            nodes_per_layer: Sequence[int],
            bits_per_node:   Sequence[int],
            *,
            node_factory:     Callable[..., BaseNode],
            tie_break:        str  = "random",
            reuse_prev_width: bool = False,
            rng:              int | None = None,
            n_jobs:           int | None = None,
            extra_node_kw:    dict | None = None,
    ):
        if len(bits_per_node) != len(nodes_per_layer) + 1:
            raise ValueError("bits_per_node must be one longer than nodes_per_layer")

        self.W = list(nodes_per_layer)
        self.B = list(bits_per_node)

        self.node_factory     = node_factory
        self.tie_break        = tie_break
        self.reuse_prev_width = reuse_prev_width
        self._rng             = np.random.default_rng(rng)
        self.n_jobs           = n_jobs or (os.cpu_count() or 1)
        self.extra_node_kw    = extra_node_kw or {}

        self.layers: List[List[BaseNode]] = []

    # ----------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        y_pm1     = y.astype(np.int8) * 2 - 1
        layer_out = X

        with ProcessPoolExecutor(self.n_jobs) as ex:
            for layer_idx, (width, bits) in enumerate(zip(self.W, self.B[:-1])):
                # --- candidate bit-pool --------------------------------
                pool = (
                    layer_out
                    if (self.reuse_prev_width or layer_idx == 0)
                    else np.column_stack((X, layer_out))
                )

                # --- draw column sets & per-node seeds -----------------
                cols_arr = self._rng.choice(
                    pool.shape[1], size=(width, bits), replace=True
                )
                seeds = self._rng.integers(0, 2**32 - 1, size=width, dtype=np.uint64)

                # --- build nodes in parallel ---------------------------
                args = [
                    (
                        self.node_factory,
                        pool[:, cols],
                        y_pm1,
                        bits,
                        cols,
                        seed,
                        self.tie_break,
                        self.extra_node_kw,
                    )
                    for cols, seed in zip(cols_arr, seeds)
                ]
                nodes = list(ex.map(_build_node, args))
                self.layers.append(nodes)

                layer_out = np.column_stack([n(pool) for n in nodes])

            # ---- final node (serial – only one) ----------------------
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
                self.tie_break,
                **self.extra_node_kw,
            )
            self.layers.append([fin_node])

        return self

    # ----------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        raw, out = X, X
        for layer_idx, layer in enumerate(self.layers):
            pool = out if (self.reuse_prev_width or layer_idx == 0) else np.column_stack((raw, out))
            out  = np.column_stack([n(pool) for n in layer])
        return out[:, 0]
