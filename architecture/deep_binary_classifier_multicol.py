from __future__ import annotations
from typing import Sequence, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class BaseNode(ABC):
    def __init__(self, X_cols: np.ndarray):
        self.X_cols = X_cols
        """
        Base class for all nodes in the network providing a minimal interface for the `DeepBinaryClassifier`.
        
        :param X_cols: The columns of the input data that this node uses, shape (num_bits,)
        """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Chooses the right columns from the input data and returns the predictions.

        :param X: The input data, shape (N, num_bits)
        :return: The predictions, shape (N,)
        """
        ...


class DeepBinaryClassifier:
    def __init__(
            self,
            layer_node_counts: Sequence[int],
            layer_bit_counts: Sequence[int],
            node_factory: Callable[..., BaseNode],
            seed: int | None = None,
            jobs: int | None = None,
    ):
        """
        Initializes a feed-forward Boolean network composed of layers of binary nodes.

        Each node is constructed from randomly selected input bits and trained independently.
        Layers are trained sequentially, and multiprocessing is optionally used for node creation within each layer.

        :param layer_node_counts: Number of nodes in each layer, shape (num_layers,)
        :param layer_bit_counts: Number of input bits each node receives per layer, shape (num_layers,)
        :param node_factory: Callable that builds a node from (X_cols, X_node, y_node, seed)
        :param seed: Master seed for RNG to ensure reproducibility
        :param jobs: Number of worker processes to use; if None or 1, runs sequentially
        """
        if len(layer_node_counts) != len(layer_bit_counts):
            raise ValueError(f"Both layer_node_counts and layer_bit_counts must specify one value per layer")

        for i in range(1, len(layer_node_counts)):
            bits = layer_bit_counts[i]
            prev = layer_node_counts[i - 1]
            if bits > prev:
                raise ValueError(
                    f"Layer {i} is trying to choose {bits} bits but only {prev} outputs available from layer {i-1}"
                )

        self.layer_node_counts = list(layer_node_counts)
        self.layer_bit_counts  = list(layer_bit_counts)

        self.node_factory = node_factory
        self._rng         = np.random.default_rng(seed)
        self.jobs       = jobs
        self.layers: List[List[BaseNode]] = []

    def _build_layer(
            self,
            X: np.ndarray,
            y: np.ndarray,
            layer_node_count: int,
            layer_bit_count: int,
            jobs: int | None
    ) -> List[BaseNode]:
        cols_arr = self._rng.choice(X.shape[1], size=(layer_node_count, layer_bit_count), replace=True)
        seeds    = self._rng.integers(0, 2 ** 32 - 1, size=layer_node_count, dtype=np.uint64)

        if jobs in (None, 1):
            nodes: List[BaseNode] = []
            for X_cols, seed in zip(cols_arr, seeds):
                node = self.node_factory( X_cols, X[:, X_cols], y, int(seed))
                nodes.append(node)
            return nodes

        with ProcessPoolExecutor(self.jobs) as ex:
            futures = []
            for X_cols, seed in zip(cols_arr, seeds):
                future = ex.submit(self.node_factory, X_cols, X[:, X_cols], y, int(seed))
                futures.append(future)

                # first we start all the workers and then we await the result of each task
                nodes = [future.result() for future in futures]
                return nodes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        """
        Trains the network layer-by-layer on the provided Boolean data.

        Each node is trained independently on a random subset of input bits, and the outputs of one layer are used as
        inputs to the next.

        :param X: Boolean input data, shape (N, input_dim)
        :param y: Boolean target labels, shape (N,)
        :return: The fitted classifier
        """
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        # from the individual nodes we can never reliably reconstruct the original input dim
        self.input_dim = X.shape[1]

        # build and evaluate all layers
        X_layer = X
        for layer_node_count, layer_bit_count in zip(self.layer_node_counts, self.layer_bit_counts):
            nodes = self._build_layer(X_layer, y, layer_node_count, layer_bit_count, self.jobs)
            self.layers.append(nodes)

            # after building all nodes of the layer, we evaluate them and build the data for the next layer
            X_layer = np.column_stack([n(X_layer) for n in nodes])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a full forward pass through the network. All nodes are processed sequentially this time.

        :param X: Boolean input data, shape (N, input_dim)
        :return: Output of the final layer, flattened, shape (N,)
        """
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        X_layer = X
        for nodes in self.layers:
            X_layer = np.column_stack([n(X_layer) for n in nodes])

        return X_layer.flatten()