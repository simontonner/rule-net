from __future__ import annotations
from typing import Sequence, Callable, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# ---------- Top-level worker (must be picklable) ----------

def _build_node_worker(
        node_factory: Callable[..., "BaseNode"],
        node_name: str,
        feature_names: list[str],
        feature_values: np.ndarray,   # (N, bits) already sliced
        target_values: np.ndarray,    # (N,)
        seed: int,
        x_cols: np.ndarray,           # wiring indices into previous layer
) -> Tuple["BaseNode", np.ndarray]:
    node = node_factory(node_name, feature_names, feature_values, target_values, seed)
    return node, np.asarray(x_cols, dtype=int)


# ---------- BaseNode (unchanged) ----------

class BaseNode(ABC):
    def __init__(self, X_cols: np.ndarray, name: str):
        self.X_cols = X_cols
        self.name = name
        """
        Base class for all nodes in the network providing a minimal interface for the `DeepBinaryClassifier`.

        :param X_cols: The columns of the input data that this node uses, shape (num_bits,)
        :param name: Unique identifier of this node (e.g., "L2N3")
        """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predictions for the given input data.

        :param X: The input data, shape (N, num_bits)
        :return: The predictions, shape (N,)
        """
        ...


# ---------- DeepBinaryClassifier ----------

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
        A feed-forward Boolean network composed of layers of binary nodes.

        Each node is constructed from randomly selected input bits and trained independently.
        Layers are trained sequentially, and multiprocessing is optionally used for node creation within each layer.

        NOTE: node_factory must be:
            node_factory(
                node_name: str,
                feature_names: list[str],
                feature_values: np.ndarray,  # (N, num_bits)
                target_values: np.ndarray,   # (N,)
                seed: int
            ) -> BaseNode

        :param layer_node_counts: Number of nodes in each layer, shape (num_layers,)
        :param layer_bit_counts: Number of input bits each node receives per layer, shape (num_layers,)
        :param node_factory: Callable described above
        :param seed: Master seed for RNG to ensure reproducibility
        :param jobs: Number of worker processes to use; if None or 1, runs sequentially
        """
        if len(layer_node_counts) != len(layer_bit_counts):
            raise ValueError("Both layer_node_counts and layer_bit_counts must specify one value per layer")

        for i in range(1, len(layer_node_counts)):
            bits = layer_bit_counts[i]
            prev = layer_node_counts[i - 1]
            if bits > prev:
                raise ValueError(
                    f"Layer {i} is trying to choose {bits} bits but only {prev} outputs available from layer {i-1}"
                )

        self.layer_node_counts = list(layer_node_counts)
        self.layer_bit_counts = list(layer_bit_counts)

        self.node_factory = node_factory
        self._rng = np.random.default_rng(seed)
        self.jobs = jobs

        # trained artifacts
        self.layers: List[List[BaseNode]] = []          # nodes per layer
        self.wiring: List[List[np.ndarray]] = []        # for each node: indices into previous layer outputs
        self.layer_feature_names: List[List[str]] = []  # names per layer (layer 0 are "L0N{i}")

    # ---------- internal: build a layer ----------

    def _build_layer(
            self,
            X_layer: np.ndarray,
            y: np.ndarray,
            layer_idx: int,
            layer_node_count: int,
            layer_bit_count: int,
            jobs: int | None,
    ) -> tuple[List[BaseNode], List[np.ndarray]]:
        """
        Build a single layer of nodes given the current layer input matrix X_layer.

        Returns:
            nodes: list of constructed nodes
            wiring: list of np.ndarray, each with the original chosen indices into X_layer
        """
        node_seeds = self._rng.integers(0, 2**32 - 1, size=layer_node_count, dtype=np.uint64)
        prev_names = self.layer_feature_names[layer_idx]

        if jobs in (None, 1):
            nodes: List[BaseNode] = []
            wiring: List[np.ndarray] = []
            for node_idx, node_seed in enumerate(node_seeds):
                X_cols = self._rng.choice(X_layer.shape[1], size=layer_bit_count, replace=False)
                node_name = f"L{layer_idx+1}N{node_idx}"
                feature_names = [prev_names[i] for i in X_cols]
                feature_values = X_layer[:, X_cols]
                node = self.node_factory(node_name, feature_names, feature_values, y, int(node_seed))
                nodes.append(node)
                wiring.append(np.asarray(X_cols, dtype=int))
            return nodes, wiring

        # multiprocessing path: use top-level worker (picklable)
        with ProcessPoolExecutor(self.jobs) as ex:
            futures = []
            for node_idx, node_seed in enumerate(node_seeds):
                X_cols = self._rng.choice(X_layer.shape[1], size=layer_bit_count, replace=False)
                node_name = f"L{layer_idx+1}N{node_idx}"
                feature_names = [prev_names[i] for i in X_cols]
                feature_values = X_layer[:, X_cols]  # slice in parent; send only needed data
                futures.append(
                    ex.submit(
                        _build_node_worker,
                        self.node_factory,
                        node_name,
                        feature_names,
                        feature_values,
                        y,
                        int(node_seed),
                        np.asarray(X_cols, dtype=int),
                    )
                )
            results = [f.result() for f in futures]
            nodes = [r[0] for r in results]
            wiring = [r[1] for r in results]
            return nodes, wiring

    # ---------- public: fit / predict ----------

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

        self.input_dim = X.shape[1]
        self.layers.clear()
        self.wiring.clear()
        self.layer_feature_names = [[f"L0N{i}" for i in range(self.input_dim)]]

        X_layer = X
        for layer_idx, (layer_node_count, layer_bit_count) in enumerate(
                zip(self.layer_node_counts, self.layer_bit_counts)
        ):
            nodes, wiring = self._build_layer(X_layer, y, layer_idx, layer_node_count, layer_bit_count, self.jobs)
            self.layers.append(nodes)
            self.wiring.append(wiring)

            # evaluate this layer
            outs = []
            for j, node in enumerate(nodes):
                X_local = X_layer[:, wiring[j]]
                outs.append(node(X_local))
            X_layer = np.column_stack(outs)

            # names for next layer are the node names we just created
            self.layer_feature_names.append([node.name for node in nodes])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a full forward pass through the network. All nodes are processed sequentially.

        :param X: Boolean input data, shape (N, input_dim)
        :return: Output of the final layer, flattened, shape (N,)
        """
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        X_layer = X
        for L, nodes in enumerate(self.layers):
            outs = []
            for j, node in enumerate(nodes):
                X_local = X_layer[:, self.wiring[L][j]]
                outs.append(node(X_local))
            X_layer = np.column_stack(outs)

        return X_layer.flatten()

    # ---------- optional: simple in-place prune ----------

    def prune(self, verbose: bool = True) -> "DeepBinaryClassifier":
        """
        In-place pruning:
          - Keeps all existing outputs of the final layer.
          - Walks dependencies end→start using current wiring to decide parents to keep.
          - Drops unneeded nodes and reindexes wiring accordingly.
        """
        if not self.layers:
            raise RuntimeError("Cannot prune: the network has no layers. Did you call fit()?")

        n_layers = len(self.layers)
        if verbose:
            before = [len(L) for L in self.layers]
            print(f"Before pruning: {before}")

        # 1) seed with all last-layer outputs
        keep: List[set[int]] = [set() for _ in range(n_layers)]
        keep[-1] = set(range(len(self.layers[-1])))

        # 2) backward dependency walk
        for L in range(n_layers - 1, 0, -1):
            prev_size = len(self.layers[L - 1])
            for j in keep[L]:
                cols = self.wiring[L][j]
                for pi in cols:
                    pi = int(pi)
                    if pi < 0 or pi >= prev_size:
                        raise IndexError(
                            f"Layer {L} node {j} references invalid prev index {pi} (prev layer size {prev_size})"
                        )
                    keep[L - 1].add(pi)

        # 3) prune layers & build old->new maps
        index_maps: List[dict[int, int]] = []
        for L in range(n_layers):
            survivors = sorted(keep[L])
            if not survivors:
                raise RuntimeError(f"Pruning resulted in empty layer {L}.")
            mapper = {old: new for new, old in enumerate(survivors)}
            index_maps.append(mapper)

            # prune nodes/wiring
            self.layers[L] = [self.layers[L][old] for old in survivors]
            self.wiring[L] = [self.wiring[L][old] for old in survivors]
            # update feature names for next layer (L+1), if exists
            if L + 1 < len(self.layer_feature_names):
                self.layer_feature_names[L + 1] = [self.layer_feature_names[L + 1][old] for old in survivors]

        # 4) reindex wiring for downstream layers
        for L in range(1, n_layers):
            prev_map = index_maps[L - 1]
            reindexed = []
            for cols in self.wiring[L]:
                new_cols = np.array([prev_map[int(c)] for c in cols], dtype=int)
                reindexed.append(new_cols)
            self.wiring[L] = reindexed

        # 5) update counts
        self.layer_node_counts = [len(L) for L in self.layers]

        if verbose:
            after = [len(L) for L in self.layers]
            print(f"After pruning:  {after}")

        return self
