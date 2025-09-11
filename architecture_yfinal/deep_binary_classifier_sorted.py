from __future__ import annotations
from typing import Sequence, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from architecture_yfinal.utils import truth_table_indices, truth_table_patterns


# ---------- Base Node ----------
class BinaryNode(ABC):
    def __init__(self, node_name: str, input_names: list[str]):
        self.node_name = node_name
        self.input_names = input_names
        self.node_predictions: np.ndarray | None = None

    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        """
        Return predictions for the given inputs.

        :param input_values: (N, num_bits) boolean array
        :return: (N,) boolean array
        """
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.node_name} must set self.node_predictions during initialization.")
        if input_values.dtype != bool:
            raise TypeError(f"Node {self.node_name} expects boolean inputs")
        k = len(self.input_names)
        if input_values.shape[1] != k:
            raise ValueError(f"Node {self.node_name} expects {k} inputs, got {input_values.shape[1]}")
        if self.node_predictions.shape[0] != (1 << k):
            raise ValueError(
                f"Node {self.node_name}: node_predictions length {self.node_predictions.shape[0]} "
                f"does not match 2**{k}={1<<k}"
            )
        idx = truth_table_indices(input_values)
        return self.node_predictions[idx]

    def get_truth_table(self):
        """Return full truth table (patterns + predictions)."""
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.node_name} must set self.node_predictions during initialization.")
        k = len(self.input_names)
        patterns = truth_table_patterns(k)
        table = np.column_stack((patterns, self.node_predictions))
        column_names = self.input_names + [f"{self.node_name} (output)"]
        return table, column_names

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return metadata specific to the node type."""
        ...


# ---------- Top-level worker (picklable) ----------
def _build_node_worker(
        node_factory: Callable[..., "BinaryNode"],
        node_name: str,
        feature_names: list[str],
        feature_values: np.ndarray,   # (N, bits) ordered by feature_names
        target_values: np.ndarray,    # (N,)
        seed: int,
) -> "BinaryNode":
    return node_factory(node_name, feature_names, feature_values, target_values, seed)


# ---------- DeepBinaryClassifier ----------
class DeepBinaryClassifier:
    def __init__(
            self,
            layer_node_counts: Sequence[int],
            layer_bit_counts: Sequence[int],
            node_factory: Callable[..., BinaryNode],
            seed: int | None = None,
            jobs: int | None = None,
    ):
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
        self.layers: List[List[BinaryNode]] = []         # layers[li] is list of nodes in layer li (0-based)
        self.layer_feature_names: List[List[str]] = []   # boundary names
        self.wiring_names: List[List[List[str]]] = []    # wiring per boundary
        self.wiring_indices: List[List[np.ndarray]] = [] # cached int indices
        self.input_names: list[str] = []                 # freeze L0 inputs

    # ---------- internals ----------
    def _name_to_idx(self, names: List[str]) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(names)}

    def _rebuild_indices_for_boundary(self, bi: int) -> None:
        if bi == 0:
            return
        prev_names = self.layer_feature_names[bi - 1]
        prev_map = self._name_to_idx(prev_names)
        idxs: List[np.ndarray] = []
        for fnames in self.wiring_names[bi]:
            idxs.append(np.array([prev_map[n] for n in fnames], dtype=int))
        self.wiring_indices[bi] = idxs

    def _build_layer(
            self,
            X_layer: np.ndarray,
            y: np.ndarray,
            layer_idx: int,
            layer_node_count: int,
            layer_bit_count: int,
            jobs: int | None,
    ) -> tuple[List[BinaryNode], List[List[str]]]:
        node_seeds = self._rng.integers(0, 2**32 - 1, size=layer_node_count, dtype=np.uint64)
        prev_names = self.layer_feature_names[layer_idx]

        chosen_parent_indices = [
            np.sort(self._rng.choice(len(prev_names), size=layer_bit_count, replace=False))
            for _ in range(layer_node_count)
        ]

        if jobs in (None, 1):
            nodes: List[BinaryNode] = []
            wiring_names_bi: List[List[str]] = []
            for node_idx, node_seed in enumerate(node_seeds):
                cols = chosen_parent_indices[node_idx]
                node_name = f"L{layer_idx+1}N{node_idx}"
                parent_names = [prev_names[i] for i in cols]
                feature_values = X_layer[:, cols]
                node = self.node_factory(node_name, parent_names, feature_values, y, int(node_seed))

                node_deps = list(node.input_names)
                missing = [n for n in node_deps if n not in parent_names]
                if missing:
                    raise ValueError(
                        f"{node_name}: node.input_names contains names not in parent set: {missing}. "
                        f"Parent set: {parent_names}"
                    )
                wiring_names_bi.append(node_deps)
                nodes.append(node)
            return nodes, wiring_names_bi

        with ProcessPoolExecutor(self.jobs) as ex:
            futures = []
            parent_name_choices = []
            for node_idx, node_seed in enumerate(node_seeds):
                cols = chosen_parent_indices[node_idx]
                node_name = f"L{layer_idx+1}N{node_idx}"
                parent_names = [prev_names[i] for i in cols]
                parent_name_choices.append(parent_names)
                feature_values = X_layer[:, cols]
                futures.append(
                    ex.submit(
                        _build_node_worker,
                        self.node_factory,
                        node_name,
                        parent_names,
                        feature_values,
                        y,
                        int(node_seed),
                    )
                )
            nodes = [f.result() for f in futures]
            wiring_names_bi = []
            for node, parent_names in zip(nodes, parent_name_choices):
                node_deps = list(node.input_names)
                missing = [n for n in node_deps if n not in parent_names]
                if missing:
                    raise ValueError(
                        f"{node.node_name}: node.input_names contains names not in parent set: {missing}. "
                        f"Parent set: {parent_names}"
                    )
                wiring_names_bi.append(node_deps)
            return nodes, wiring_names_bi

    # ---------- public ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        self.input_dim = X.shape[1]
        self.layers.clear()
        self.wiring_names.clear()
        self.wiring_indices.clear()

        # boundary 0 = inputs
        self.layer_feature_names = [[f"L0N{i}" for i in range(self.input_dim)]]
        self.input_names = list(self.layer_feature_names[0])
        self.wiring_names.append([])
        self.wiring_indices.append([])

        X_layer = X
        for li, (layer_node_count, layer_bit_count) in enumerate(
                zip(self.layer_node_counts, self.layer_bit_counts)
        ):
            nodes, wiring_names_bi = self._build_layer(
                X_layer, y, li, layer_node_count, layer_bit_count, self.jobs
            )
            self.layers.append(nodes)
            self.wiring_names.append(wiring_names_bi)

            prev_map = self._name_to_idx(self.layer_feature_names[li])
            idxs = [
                np.array([prev_map[n] for n in nodes[j].input_names], dtype=int)
                for j in range(len(nodes))
            ]
            self.wiring_indices.append(idxs)

            outs = [nodes[j](X_layer[:, idxs[j]]) for j in range(len(nodes))]
            X_layer = np.column_stack(outs)
            self.layer_feature_names.append([node.node_name for node in nodes])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        X_layer = X
        for bi in range(1, len(self.layer_feature_names)):
            li = bi - 1
            nodes = self.layers[li]
            idxs_list = self.wiring_indices[bi]
            outs = [nodes[j](X_layer[:, idxs_list[j]]) for j in range(len(nodes))]
            X_layer = np.column_stack(outs)

        # enforce single output
        if X_layer.ndim != 2 or X_layer.shape[1] != 1:
            raise ValueError(
                f"Final layer produced {X_layer.shape[1]} outputs; expected exactly 1. "
                "Check layer_node_counts (last should be 1)."
            )
        return X_layer[:, 0]

    def refresh_wiring_from_nodes(self) -> None:
        """Resync wiring_names to each node's current feature_names and rebuild indices."""
        for bi in range(1, len(self.layer_feature_names)):
            li = bi - 1
            if li >= len(self.layers):
                break
            prev_set = set(self.layer_feature_names[bi - 1])
            nodes = self.layers[li]
            deps_list = []
            for n in nodes:
                deps = list(n.input_names)
                missing = [nm for nm in deps if nm not in prev_set]
                if missing:
                    raise ValueError(
                        f"{n.node_name}: input_names not in previous boundary: {missing}. "
                        f"Prev boundary: {sorted(prev_set)}"
                    )
                deps_list.append(deps)
            self.wiring_names[bi] = deps_list
            self._rebuild_indices_for_boundary(bi)
