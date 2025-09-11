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
    def __init__(
            self,
            layer_node_counts: Sequence[int],
            layer_bit_counts: Sequence[int],
            node_factory: Callable[..., "BinaryNode"],
            seed: int | None = None,
            jobs: int | None = None,
    ):
        if len(layer_node_counts) != len(layer_bit_counts):
            raise ValueError("layer_node_counts and layer_bit_counts must have equal length")

        for i in range(1, len(layer_node_counts)):
            bits = layer_bit_counts[i]
            prev = layer_node_counts[i - 1]
            if bits > prev:
                raise ValueError(f"Nodes in layer {i} expect {bits} inputs, but previous layer has only {prev} nodes")

        self.layer_node_counts = list(layer_node_counts)
        self.layer_bit_counts = list(layer_bit_counts)
        self.node_factory = node_factory
        self._rng = np.random.default_rng(seed)
        self.jobs = jobs

        # parallel bookkeeping of layers with nodes and indices
        self.layers: List[List["BinaryNode"]] = []
        self.wiring_indices: List[List[np.ndarray]] = []
        self.layer_node_names: List[Sequence[str]] = []

    # ---------- internals ----------
    def _compute_wiring(self, prev_names: Sequence[str], nodes: Sequence["BinaryNode"]) -> List[np.ndarray]:
        """Map each node.input_names to column indices in prev_names."""
        prev_map = {nm: i for i, nm in enumerate(prev_names)}
        idxs_list: List[np.ndarray] = []
        for n in nodes:
            missing = [nm for nm in n.input_names if nm not in prev_map]
            if missing:
                raise ValueError(
                    f"{n.name}: input_names not in previous boundary: {missing}. "
                    f"Prev boundary: {prev_names}"
                )
            idxs_list.append(np.fromiter((prev_map[nm] for nm in n.input_names), dtype=int))
        return idxs_list

    def _rebuild_all_wiring(self) -> None:
        """Recompute wiring_indices for all boundaries from node.input_names."""
        self.wiring_indices = [[]]  # L0 has no wiring
        for li, nodes in enumerate(self.layers):
            prev_names = self.layer_node_names[li]
            self.wiring_indices.append(self._compute_wiring(prev_names, nodes))

    def _build_layer(
            self,
            layer_inputs: np.ndarray,
            target_values: np.ndarray,
            prev_names: Sequence[str],
            layer_idx: int,
            node_count: int,
            bit_count: int,
            jobs: int | None,
    ) -> tuple[List["BinaryNode"], List[np.ndarray], List[str]]:
        seeds = self._rng.integers(0, 2**32 - 1, size=node_count, dtype=np.uint64)

        # Sample parents (sorted for deterministic column order)
        sampled_cols_list = [
            np.sort(self._rng.choice(len(prev_names), size=bit_count, replace=False))
            for _ in range(node_count)
        ]

        # Build nodes
        if jobs in (None, 1):
            nodes: List["BinaryNode"] = []
            for node_idx, (cols, s) in enumerate(zip(sampled_cols_list, seeds)):
                node_name = f"L{layer_idx+1}N{node_idx}"
                parent_names = [prev_names[i] for i in cols]
                node_input_values = layer_inputs[:, cols]
                nodes.append(self.node_factory(node_name, parent_names, node_input_values, target_values, int(s)))
        else:
            with ProcessPoolExecutor(jobs) as ex:
                futures = []
                for node_idx, (cols, s) in enumerate(zip(sampled_cols_list, seeds)):
                    node_name = f"L{layer_idx+1}N{node_idx}"
                    parent_names = [prev_names[i] for i in cols]
                    node_input_values = layer_inputs[:, cols]
                    futures.append(ex.submit(
                        self.node_factory, node_name, parent_names, node_input_values, target_values, int(s)
                    ))
                nodes = [f.result() for f in futures]

        # Single source of truth for wiring
        idxs_list = self._compute_wiring(prev_names, nodes)
        new_names = [n.name for n in nodes]
        return nodes, idxs_list, new_names

    # ---------- public ----------
    def fit(self, input_values: np.ndarray, target_values: np.ndarray) -> "DeepBinaryClassifier":
        if input_values.dtype != bool or target_values.dtype != bool:
            raise TypeError("input_values and target_values must be boolean arrays")

        self.layers.clear()
        self.wiring_indices.clear()
        self.layer_node_names.clear()

        # boundary 0 = input names (immutable)
        l0_names = tuple(f"L0N{i}" for i in range(input_values.shape[1]))
        self.layer_node_names.append(l0_names)
        self.wiring_indices.append([])  # no wiring at L0

        layer_inputs = input_values
        for li, (node_cnt, bit_cnt) in enumerate(zip(self.layer_node_counts, self.layer_bit_counts)):
            prev_names = self.layer_node_names[li]
            nodes, idxs_list, new_names = self._build_layer(
                layer_inputs, target_values, prev_names, li, node_cnt, bit_cnt, self.jobs
            )
            self.layers.append(nodes)
            self.wiring_indices.append(idxs_list)
            self.layer_node_names.append(new_names)

            # forward using ACTUAL deps
            outs = [n(layer_inputs[:, idxs]) for n, idxs in zip(nodes, idxs_list)]
            layer_inputs = np.column_stack(outs) if len(outs) > 1 else outs[0].reshape(-1, 1)

        return self

    def predict(self, input_values: np.ndarray) -> np.ndarray:
        if input_values.dtype != bool:
            raise TypeError("input_values must be a boolean array")
        if not self.layers:
            raise RuntimeError("Model not fitted")

        expected = len(self.layer_node_names[0])
        if input_values.shape[1] != expected:
            raise ValueError(f"input_values has {input_values.shape[1]} columns - model expects {expected} L0 inputs")

        layer_inputs = input_values
        for li, nodes in enumerate(self.layers):
            idxs_list = self.wiring_indices[li + 1]
            outs = [n(layer_inputs[:, idxs]) for n, idxs in zip(nodes, idxs_list)]
            layer_inputs = np.column_stack(outs) if len(outs) > 1 else outs[0].reshape(-1, 1)

        if layer_inputs.shape[1] != 1:
            raise ValueError(
                f"Final layer produced {layer_inputs.shape[1]} outputs; expected 1. "
                "Set last layer_node_counts[-1] = 1."
            )
        return layer_inputs[:, 0]

    def prune(self) -> "DeepBinaryClassifier":
        """
        Index-based pruning; then rebuild all wiring in one pass.
        L0 boundary is immutable by construction (tuple).
        """
        if not self.layers:
            raise RuntimeError("Cannot prune an unfitted model")

        # Ensure wiring reflects current node.input_names
        self._rebuild_all_wiring()

        n_layers = len(self.layers)

        # Backward reachability: which nodes to keep per layer
        keep = [set() for _ in range(n_layers)]
        keep[-1] = set(range(len(self.layers[-1])))
        for li in range(n_layers - 1, 0, -1):
            for j in keep[li]:
                for p in self.wiring_indices[li + 1][j]:
                    keep[li - 1].add(int(p))

        # Slice each layer and its outgoing boundary names
        for li in range(n_layers):
            survivors = sorted(keep[li])
            if not survivors:
                raise RuntimeError(f"Pruning resulted in empty layer {li}")

            # prune nodes
            self.layers[li] = [self.layers[li][s] for s in survivors]

            # prune boundary names after this layer
            bi = li + 1
            self.layer_node_names[bi] = [self.layer_node_names[bi][s] for s in survivors]

        # Rebuild wiring once from names -> indices
        self._rebuild_all_wiring()

        # Update counts
        self.layer_node_counts = [len(L) for L in self.layers]
        return self