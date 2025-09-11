# deep_binary_classifier.py
# Name-based wiring with strict L0 invariants and robust pruning.
# Conventions:
# - Inputs are named L0N0, L0N1, ..., L0N{D-1}
# - Layer li ∈ [0..n_layers-1] contains nodes named L{li+1}N{j}
# - Boundaries bi ∈ [0..n_layers]: layer_feature_names[bi]
#     bi=0 -> input names (L0N*)
#     bi=k -> names of nodes in layer k (1-based)
# - wiring_names[bi] holds deps (by names) for nodes in layer li=bi-1
# - wiring_indices[bi] caches integer indices for fast slicing; derived from names

from __future__ import annotations
from typing import Sequence, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from architecture_vis.utils import truth_table_indices, truth_table_patterns


class BinaryNode(ABC):
    def __init__(self, node_name: str, input_names: list[str]):
        self.name = node_name
        self.input_names = input_names

        self.node_predictions = None

        """
        Base class for all nodes in the network.
              
        :param node_name: The name of this node.
        :param input_names: The names of the input values (features).        
        
        :remarks:
            The node_name and input_names are used for wiring up the network graph.
        """

    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        """
        Returns predictions for the given input vales.

        :param input_values: The input values, shape (N, num_bits)
        :return: The predictions, shape (N,)
        """
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")

        if input_values.shape[1] != len(self.input_names):
            raise ValueError(f"Node {self.name} accepts only inputs of length {len(self.input_names)}")

        return self.node_predictions[truth_table_indices(input_values)]

    def get_truth_table(self):
        """Return full truth table (patterns + predictions)."""
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")

        patterns = truth_table_patterns(len(self.input_names))
        table = np.column_stack((patterns, self.node_predictions))
        column_names = self.input_names + [f"{self.name} (output)"]
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
        # boundary names (length = n_layers + 1)
        self.layer_feature_names: List[List[str]] = []  # layer_feature_names[0] -> L0N*, [k] -> names in layer k (1-based)
        # wiring per boundary (length mirrors layer_feature_names)
        self.wiring_names: List[List[List[str]]] = []   # wiring_names[bi] -> deps for nodes in layer li=bi-1
        self.wiring_indices: List[List[np.ndarray]] = []# cached int indices for fast slicing

        # freeze L0 (inputs must never change)
        self.input_names: list[str] = []

    # ---------- internals ----------
    def _name_to_idx(self, names: List[str]) -> dict[str, int]:
        return {nm: i for i, nm in enumerate(names)}

    def _rebuild_indices_for_boundary(self, bi: int) -> None:
        """
        Recompute wiring_indices[bi] from wiring_names[bi] using names at the previous boundary (bi-1).
        bi: boundary index ∈ [0..n_layers]; bi==0 has no wiring.
        """
        if bi == 0:
            return
        prev_names = self.layer_feature_names[bi - 1]
        prev_map = self._name_to_idx(prev_names)
        idxs: List[np.ndarray] = []
        for fnames in self.wiring_names[bi]:
            # validate all deps exist in previous boundary
            idxs.append(np.array([prev_map[n] for n in fnames], dtype=int))
        self.wiring_indices[bi] = idxs

    def _build_layer(
            self,
            X_layer: np.ndarray,
            y: np.ndarray,
            layer_idx: int,          # li
            layer_node_count: int,
            layer_bit_count: int,
            jobs: int | None,
    ) -> tuple[List[BinaryNode], List[List[str]]]:
        """
        Returns:
          nodes: nodes built for layer li
          wiring_names_bi: deps by names for boundary bi = li+1 (one list per node)
        """
        node_seeds = self._rng.integers(0, 2**32 - 1, size=layer_node_count, dtype=np.uint64)
        prev_names = self.layer_feature_names[layer_idx]  # boundary bi = li, names of previous layer nodes

        chosen_parent_indices = [
            self._rng.choice(len(prev_names), size=layer_bit_count, replace=False)
            for _ in range(layer_node_count)
        ]

        if jobs in (None, 1):
            nodes: List[BinaryNode] = []
            wiring_names_bi: List[List[str]] = []
            for node_idx, node_seed in enumerate(node_seeds):
                cols = chosen_parent_indices[node_idx]
                node_name = f"L{layer_idx+1}N{node_idx}"
                parent_names = [prev_names[i] for i in cols]  # always names from previous boundary
                feature_values = X_layer[:, cols]
                node = self.node_factory(node_name, parent_names, feature_values, y, int(node_seed))

                # Node may have reduced/reordered its deps; ensure they are subset of parent_names
                node_deps = list(node.input_names)
                missing = [n for n in node_deps if n not in parent_names]
                if missing:
                    raise ValueError(
                        f"{node_name}: node.feature_names contains names not in parent set: {missing}. "
                        f"Parent set: {parent_names}"
                    )
                wiring_names_bi.append(node_deps)
                nodes.append(node)
            return nodes, wiring_names_bi

        # multiprocessing path
        with ProcessPoolExecutor(self.jobs) as ex:
            futures = []
            parent_name_choices = []  # keep chosen names to validate later if needed
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
                        f"{node.name}: node.feature_names contains names not in parent set: {missing}. "
                        f"Parent set: {parent_names}"
                    )
                wiring_names_bi.append(node_deps)
            return nodes, wiring_names_bi

    # ---------- public: fit / predict ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")

        self.input_dim = X.shape[1]
        self.layers.clear()
        self.wiring_names.clear()
        self.wiring_indices.clear()

        # Boundary 0: inputs
        self.layer_feature_names = [[f"L0N{i}" for i in range(self.input_dim)]]
        self.input_names = list(self.layer_feature_names[0])  # freeze L0
        self.wiring_names.append([])     # bi=0: no wiring
        self.wiring_indices.append([])

        X_layer = X
        for li, (layer_node_count, layer_bit_count) in enumerate(
                zip(self.layer_node_counts, self.layer_bit_counts)
        ):
            nodes, wiring_names_bi = self._build_layer(
                X_layer, y, li, layer_node_count, layer_bit_count, self.jobs
            )
            self.layers.append(nodes)
            # attach wiring for boundary bi = li+1
            self.wiring_names.append(wiring_names_bi)

            # Build indices from CURRENT node deps (subset/order after reduction)
            prev_map = self._name_to_idx(self.layer_feature_names[li])  # previous boundary
            idxs = [
                np.array([prev_map[n] for n in nodes[j].input_names], dtype=int)
                for j in range(len(nodes))
            ]
            self.wiring_indices.append(idxs)

            # Forward this layer
            outs = [nodes[j](X_layer[:, idxs[j]]) for j in range(len(nodes))]
            X_layer = np.column_stack(outs)

            # Next boundary names are these node names
            self.layer_feature_names.append([node.node_name for node in nodes])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")

        X_layer = X
        # boundaries bi=1..n_layers correspond to layers li=bi-1
        for bi in range(1, len(self.layer_feature_names)):
            li = bi - 1
            nodes = self.layers[li]
            idxs_list = self.wiring_indices[bi]
            outs = [nodes[j](X_layer[:, idxs_list[j]]) for j in range(len(nodes))]
            X_layer = np.column_stack(outs)
        return X_layer.flatten()

    # ---------- maintenance ----------
    def refresh_wiring_from_nodes(self) -> None:
        """
        Resync wiring_names to each node's *current* feature_names and rebuild indices.
        Use this after any node-side expression reduction.
        """
        for bi in range(1, len(self.layer_feature_names)):
            li = bi - 1
            if li >= len(self.layers):
                break
            nodes = self.layers[li]
            # Node deps must be names from previous boundary
            prev_set = set(self.layer_feature_names[bi - 1])
            deps_list = []
            for n in nodes:
                deps = list(n.input_names)
                missing = [nm for nm in deps if nm not in prev_set]
                if missing:
                    raise ValueError(
                        f"{n.name}: feature_names contains names not in previous boundary: {missing}. "
                        f"Prev boundary: {sorted(prev_set)}"
                    )
                deps_list.append(deps)
            self.wiring_names[bi] = deps_list
            self._rebuild_indices_for_boundary(bi)

    def prune(self, verbose: bool = True) -> "DeepBinaryClassifier":
        """
        Name-based pruning:
          - Keep all outputs of the final layer.
          - Walk deps (by names) back to L0.
          - Drop unneeded nodes and rebuild indices.
          - L0 (inputs) is immutable and never pruned.
        """
        if not self.layers:
            raise RuntimeError("Cannot prune: the network has no layers. Did you call fit()?")

        # Enforce L0 invariance before any mutation
        if self.layer_feature_names and self.layer_feature_names[0] != self.input_names:
            raise RuntimeError("L0 boundary changed — inputs must be immutable.")

        n_layers = len(self.layers)
        if verbose:
            before = [len(L) for L in self.layers]
            print(f"Before pruning: {before}")

        # Ensure wiring names reflect CURRENT node deps
        self.refresh_wiring_from_nodes()

        # 1) seed with all last-layer nodes
        keep: List[set[int]] = [set() for _ in range(n_layers)]
        keep[-1] = set(range(len(self.layers[-1])))

        # 2) backward dependency walk
        for li in range(n_layers - 1, 0, -1):
            bi = li + 1
            deps_list = self.wiring_names[bi]                  # deps for layer li nodes
            prev_name_to_idx = self._name_to_idx(self.layer_feature_names[li])  # previous boundary
            for j in keep[li]:
                for nm in deps_list[j]:
                    keep[li - 1].add(prev_name_to_idx[nm])

        # 3) prune each layer and related structures
        for li in range(n_layers):
            survivors = sorted(keep[li])
            if not survivors:
                raise RuntimeError(f"Pruning resulted in empty layer {li}.")
            # prune nodes
            self.layers[li] = [self.layers[li][old] for old in survivors]
            # prune names at next boundary (bi = li+1)
            self.layer_feature_names[li + 1] = [self.layer_feature_names[li + 1][old] for old in survivors]
            # prune wiring for this layer (stored at boundary bi = li+1)
            if li + 1 < len(self.wiring_names):
                self.wiring_names[li + 1] = [self.wiring_names[li + 1][old] for old in survivors]

        # 4) rebuild indices for all boundaries
        # Keep wiring_indices length consistent
        if len(self.wiring_indices) != len(self.wiring_names):
            self.wiring_indices = [[] for _ in range(len(self.wiring_names))]
        for bi in range(1, len(self.layer_feature_names)):
            if bi >= len(self.wiring_indices):
                self.wiring_indices.append([])
            self._rebuild_indices_for_boundary(bi)

        # 5) update counts
        self.layer_node_counts = [len(L) for L in self.layers]

        # Re-enforce L0 invariance after mutations
        if self.layer_feature_names[0] != self.input_names:
            raise RuntimeError("L0 boundary changed during pruning — this is a bug.")

        if verbose:
            after = [len(L) for L in self.layers]
            print(f"After pruning:  {after}")

        return self
