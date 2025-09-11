# deep_binary_classifier.py
# Name-based wiring with strict L0 invariants and robust pruning.
# Internally we store only INDICES for wiring; names exist for BinaryNode ctor & display.

from __future__ import annotations
from typing import Sequence, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from architecture_yfinal.utils import truth_table_indices, truth_table_patterns


class BinaryNode(ABC):
    def __init__(self, node_name: str, input_names: list[str]):
        self.name = node_name
        self.input_names = input_names
        self.node_predictions = None
        """
        Base class for all nodes in the network.
        """

    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")
        if input_values.shape[1] != len(self.input_names):
            raise ValueError(f"Node {self.name} accepts only inputs of length {len(self.input_names)}")
        return self.node_predictions[truth_table_indices(input_values)]

    def get_truth_table(self):
        if self.node_predictions is None:
            raise AttributeError(f"Node {self.name} needs to populate self.node_predictions during initialization.")
        patterns = truth_table_patterns(len(self.input_names))
        table = np.column_stack((patterns, self.node_predictions))
        column_names = self.input_names + [f"{self.name} (output)"]
        return table, column_names

    @abstractmethod
    def get_metadata(self) -> dict:
        ...


class DeepBinaryClassifier:
    """
    Boolean network of BinaryNodes.
    - Train nodes on sampled parent subsets.
    - After node builds (and possibly simplifies), derive wiring INDICES from node.input_names.
    - Use those indices for forward passes and pruning.
    """

    def __init__(
            self,
            layer_node_counts: Sequence[int],
            layer_bit_counts: Sequence[int],
            node_factory: Callable[..., "BinaryNode"],  # (name, input_names, X_subset, y, seed) -> BinaryNode
            seed: int | None = None,
            jobs: int | None = None,
    ):
        if len(layer_node_counts) != len(layer_bit_counts):
            raise ValueError("layer_node_counts and layer_bit_counts must have equal length")

        for i in range(1, len(layer_node_counts)):
            if layer_bit_counts[i] > layer_node_counts[i - 1]:
                raise ValueError(
                    f"Layer {i}: needs {layer_bit_counts[i]} bits but prev layer has {layer_node_counts[i-1]}"
                )

        self.layer_node_counts = list(layer_node_counts)
        self.layer_bit_counts = list(layer_bit_counts)
        self.node_factory = node_factory
        self._rng = np.random.default_rng(seed)
        self.jobs = jobs

        # trained artifacts
        self.layers: List[List["BinaryNode"]] = []
        self.wiring_indices: List[List[np.ndarray]] = []   # parents per node at boundary bi = li+1
        self.layer_feature_names: List[List[str]] = []     # names per boundary (for ctor/display)
        self.input_dim: int | None = None
        self.input_names: List[str] = []

    # ---------- internals ----------
    def _build_layer(
            self,
            X_layer: np.ndarray,
            y: np.ndarray,
            prev_names: List[str],
            layer_idx: int,
            node_count: int,
            bit_count: int,
            jobs: int | None,
    ) -> tuple[List["BinaryNode"], List[np.ndarray], List[str]]:
        seeds = self._rng.integers(0, 2**32 - 1, size=node_count, dtype=np.uint64)

        # Sample parents for training data (sorted for deterministic column order)
        sampled_cols_list = [
            np.sort(self._rng.choice(len(prev_names), size=bit_count, replace=False))
            for _ in range(node_count)
        ]

        # Build nodes (no nested callables to avoid pickle issues)
        if jobs in (None, 1):
            nodes: List["BinaryNode"] = []
            for node_idx, (cols, s) in enumerate(zip(sampled_cols_list, seeds)):
                node_name = f"L{layer_idx+1}N{node_idx}"
                parent_names = [prev_names[i] for i in cols]
                X_subset = X_layer[:, cols]
                nodes.append(self.node_factory(node_name, parent_names, X_subset, y, int(s)))
        else:
            with ProcessPoolExecutor(jobs) as ex:
                futures = []
                for node_idx, (cols, s) in enumerate(zip(sampled_cols_list, seeds)):
                    node_name = f"L{layer_idx+1}N{node_idx}"
                    parent_names = [prev_names[i] for i in cols]
                    X_subset = X_layer[:, cols]
                    futures.append(ex.submit(self.node_factory, node_name, parent_names, X_subset, y, int(s)))
                nodes = [f.result() for f in futures]

        # After node may simplify/reorder deps, build ACTUAL wiring indices from node.input_names.
        prev_map = {nm: i for i, nm in enumerate(prev_names)}
        idxs_list: List[np.ndarray] = []
        for node in nodes:
            deps = list(node.input_names)
            missing = [nm for nm in deps if nm not in prev_map]
            if missing:
                raise ValueError(
                    f"{node.name}: input_names not in previous boundary: {missing}. "
                    f"Prev boundary: {prev_names}"
                )
            idxs = np.array([prev_map[nm] for nm in deps], dtype=int)
            idxs_list.append(idxs)

        new_names = [n.name for n in nodes]
        return nodes, idxs_list, new_names

    # ---------- public ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepBinaryClassifier":
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be boolean arrays")
        self.input_dim = X.shape[1]

        self.layers.clear()
        self.wiring_indices.clear()
        self.layer_feature_names.clear()

        # boundary 0 = input names
        self.input_names = [f"L0N{i}" for i in range(self.input_dim)]
        self.layer_feature_names.append(self.input_names)
        self.wiring_indices.append([])  # no wiring at L0

        X_layer = X
        for li, (node_cnt, bit_cnt) in enumerate(zip(self.layer_node_counts, self.layer_bit_counts)):
            prev_names = self.layer_feature_names[li]
            nodes, idxs_list, new_names = self._build_layer(
                X_layer, y, prev_names, li, node_cnt, bit_cnt, self.jobs
            )
            self.layers.append(nodes)
            self.wiring_indices.append(idxs_list)
            self.layer_feature_names.append(new_names)

            # forward using ACTUAL deps
            outs = [n(X_layer[:, idxs]) for n, idxs in zip(nodes, idxs_list)]
            X_layer = np.column_stack(outs) if len(outs) > 1 else outs[0].reshape(-1, 1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be a boolean array")
        if not self.layers:
            raise RuntimeError("Model not fitted")

        X_layer = X
        for li, nodes in enumerate(self.layers):
            idxs_list = self.wiring_indices[li + 1]
            outs = [n(X_layer[:, idxs]) for n, idxs in zip(nodes, idxs_list)]
            X_layer = np.column_stack(outs) if len(outs) > 1 else outs[0].reshape(-1, 1)

        if X_layer.shape[1] != 1:
            raise ValueError(
                f"Final layer produced {X_layer.shape[1]} outputs; expected 1. "
                "Set last layer_node_counts[-1] = 1."
            )
        return X_layer[:, 0]

    def refresh_wiring_from_nodes(self) -> None:
        """Resync wiring_indices from node.input_names (e.g., after internal simplification)."""
        for bi in range(1, len(self.layer_feature_names)):
            prev_names = self.layer_feature_names[bi - 1]
            prev_map = {nm: i for i, nm in enumerate(prev_names)}
            nodes = self.layers[bi - 1]
            idxs_list: List[np.ndarray] = []
            for n in nodes:
                deps = list(n.input_names)
                missing = [nm for nm in deps if nm not in prev_map]
                if missing:
                    raise ValueError(
                        f"{n.name}: input_names not in previous boundary: {missing}. "
                        f"Prev boundary: {prev_names}"
                    )
                idxs = np.array([prev_map[nm] for nm in deps], dtype=int)
                idxs_list.append(idxs)
            self.wiring_indices[bi] = idxs_list

    def prune(self, verbose: bool = True) -> "DeepBinaryClassifier":
        """
        Index-based pruning with strict L0 invariants.
        - Seed with all final-layer nodes.
        - Walk wiring_indices backwards.
        - Slice layers, names, wiring accordingly.
        - Enforce that L0 boundary remains identical before/after.
        """
        if not self.layers:
            raise RuntimeError("Cannot prune an unfitted model")

        # Enforce L0 invariance before any mutation
        if self.layer_feature_names and self.layer_feature_names[0] != self.input_names:
            raise RuntimeError("L0 boundary changed — inputs must be immutable.")

        n_layers = len(self.layers)
        if verbose:
            print("Before pruning:", [len(L) for L in self.layers])

        # Ensure wiring reflects CURRENT node deps
        self.refresh_wiring_from_nodes()

        keep: List[set[int]] = [set() for _ in range(n_layers)]
        keep[-1] = set(range(len(self.layers[-1])))

        # backward reachability via indices
        for li in range(n_layers - 1, 0, -1):
            for j in keep[li]:
                for p in self.wiring_indices[li + 1][j]:
                    keep[li - 1].add(int(p))

        # slice in lockstep
        for li in range(n_layers):
            survivors = sorted(keep[li])
            if not survivors:
                raise RuntimeError(f"Pruning resulted in empty layer {li}")
            self.layers[li] = [self.layers[li][s] for s in survivors]
            self.layer_feature_names[li + 1] = [self.layer_feature_names[li + 1][s] for s in survivors]
            if li + 1 < len(self.wiring_indices):
                self.wiring_indices[li + 1] = [self.wiring_indices[li + 1][s] for s in survivors]

        # update counts
        self.layer_node_counts = [len(L) for L in self.layers]

        # Re-enforce L0 invariance after mutations
        if self.layer_feature_names[0] != self.input_names:
            raise RuntimeError("L0 boundary changed during pruning — this is a bug.")

        if verbose:
            print("After pruning: ", [len(L) for L in self.layers])
        return self
