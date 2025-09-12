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

        self.layers: List[List["BinaryNode"]] = []
        self.backlinks: List[List[List[int]]] = []
        self.node_names: List[Sequence[str]] = []

    @staticmethod
    def _get_backlinks(prev_node_names: Sequence[str], layer_nodes: Sequence["BinaryNode"]) -> List[List[int]]:
        prev_backlink_map = {prev_node_name: idx for idx, prev_node_name in enumerate(prev_node_names)}

        layer_backlinks: List[List[int]] = []
        for node in layer_nodes:
            node_backlinks = [prev_backlink_map[prev_node_name] for prev_node_name in node.input_names]
            layer_backlinks.append(node_backlinks)

        return layer_backlinks

    def _rewire_net(self) -> None:
        self.backlinks = [[]]  # the input layer has no backlinks
        for layer_idx, layer_nodes in enumerate(self.layers):
            layer_node_names = self.node_names[layer_idx]
            layer_backlinks = self._get_backlinks(layer_node_names, layer_nodes)
            self.backlinks.append(layer_backlinks)

    def _build_layer(
            self,
            layer_input_values: np.ndarray,
            target_values: np.ndarray,
            prev_node_names: Sequence[str],
            layer_idx: int,
            node_count: int,
            bit_count: int,
            jobs: int | None,
    ) -> tuple[List["BinaryNode"], List[List[int]], List[str]]:
        node_seeds = self._rng.integers(0, 2**32 - 1, size=node_count, dtype=np.int64).tolist()

        if jobs in (None, 1):
            nodes: List["BinaryNode"] = []
            for node_idx, node_seed in enumerate(node_seeds):
                node_backlinks = np.sort(self._rng.choice(len(prev_node_names), size=bit_count, replace=False))
                input_names = [prev_node_names[b] for b in node_backlinks]
                input_values = layer_input_values[:, node_backlinks]
                node_name = f"L{layer_idx+1}N{node_idx}"
                node = self.node_factory(node_name, input_names, input_values, target_values, node_seed)
                nodes.append(node)

            backlinks = self._get_backlinks(prev_node_names, nodes)
            node_names = [n.name for n in nodes]
            return nodes, backlinks, node_names

        with ProcessPoolExecutor(jobs) as ex:
            futures = []
            for node_idx, node_seed in enumerate(node_seeds):
                node_backlinks = np.sort(self._rng.choice(len(prev_node_names), size=bit_count, replace=False))
                input_names = [prev_node_names[b] for b in node_backlinks]
                input_values = layer_input_values[:, node_backlinks]
                node_name = f"L{layer_idx+1}N{node_idx}"
                futures.append(ex.submit(self.node_factory, node_name, input_names, input_values, target_values, node_seed))

            nodes = [f.result() for f in futures]

        backlinks = self._get_backlinks(prev_node_names, nodes)
        node_names = [n.name for n in nodes]
        return nodes, backlinks, node_names


    def fit(self, input_values: np.ndarray, target_values: np.ndarray) -> "DeepBinaryClassifier":
        if input_values.dtype != bool or target_values.dtype != bool:
            raise TypeError("input_values and target_values must be boolean arrays")

        self.layers.clear()
        self.backlinks.clear()
        self.node_names.clear()

        # we wanna have the original input names as immutable (tuple)
        input_names = tuple(f"L0N{i}" for i in range(input_values.shape[1]))
        self.node_names.append(input_names)
        self.backlinks.append([])

        layer_input_values = input_values
        for layer_idx, (node_cnt, bit_cnt) in enumerate(zip(self.layer_node_counts, self.layer_bit_counts)):
            layer_node_names = self.node_names[layer_idx]

            build_layer_args = (layer_input_values, target_values, layer_node_names, layer_idx, node_cnt, bit_cnt, self.jobs)
            layer_nodes, layer_backlinks, layer_node_names = self._build_layer(*build_layer_args)

            self.layers.append(layer_nodes)
            self.backlinks.append(layer_backlinks)
            self.node_names.append(layer_node_names)

            # evaluate each node in the layer with its specific inputs
            layer_output_values = []
            for node, node_backlinks in zip(layer_nodes, layer_backlinks):
                node_output = node(layer_input_values[:, node_backlinks])
                layer_output_values.append(node_output)

            if len(layer_output_values) == 1: # special treatment for last layer with single node
                layer_input_values = layer_output_values[0].reshape(-1, 1)
                continue

            layer_input_values = np.column_stack(layer_output_values)

        return self

    def predict(self, input_values: np.ndarray) -> np.ndarray:
        if input_values.dtype != bool:
            raise TypeError("input_values must be a boolean array")
        if not self.layers:
            raise RuntimeError("Model not fitted")

        expected = len(self.node_names[0])
        if input_values.shape[1] != expected:
            raise ValueError(f"input_values has {input_values.shape[1]} columns - model expects {expected} L0 inputs")

        layer_inputs = input_values
        for li, nodes in enumerate(self.layers):
            idxs_list = self.backlinks[li + 1]
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
        self._rewire_net()

        n_layers = len(self.layers)

        # Backward reachability: which nodes to keep per layer
        keep = [set() for _ in range(n_layers)]
        keep[-1] = set(range(len(self.layers[-1])))
        for li in range(n_layers - 1, 0, -1):
            for j in keep[li]:
                for p in self.backlinks[li + 1][j]:
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
            self.node_names[bi] = [self.node_names[bi][s] for s in survivors]

        # Rebuild wiring once from names -> indices
        self._rewire_net()

        return self