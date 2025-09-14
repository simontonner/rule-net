from typing import Any, Dict
import numpy as np


def truth_table_patterns(num_bits: int) -> np.ndarray[np.bool_]:
    """
    Generates bit-patterns like the inputs of a typical truth table.

    :param num_bits: Number of bits in the input
    :return: Boolean array with every possible bit-string, shape (2**num_bits, num_bits)
    """
    row_indices = np.arange(2 ** num_bits - 1, -1, -1, dtype=np.uint32).reshape(-1, 1)      # descending
    bit_weights = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)                     # big-endian
    return (row_indices & bit_weights) > 0


def truth_table_indices(bit_patterns: np.ndarray) -> np.ndarray[np.int64]:
    """
    Converts bit patterns to their corresponding row indices in a typical truth table.

    :param bit_patterns: Boolean array of bit patterns, shape (N, num_bits)
    :return: Integer array of row indices, shape (N,)
    """
    num_bits = bit_patterns.shape[1]

    bit_weights = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
    ascending_indices = (bit_patterns.astype(np.uint32) * bit_weights).sum(axis=1)

    max_index = (1 << num_bits) - 1
    descending_indices = max_index - ascending_indices

    return descending_indices.astype(np.int64)


def describe_architecture(net) -> Dict[str, Any]:
    """
    Describes the architecture of a fitted "DeepBinaryClassifier" network as a nested dictionary.

    :param net: The fitted network to describe
    :return: A dictionary representing the architecture
    """
    if not getattr(net, "layers", None):
        raise RuntimeError("Network not fitted. Nothing to describe.")

    num_layers = len(net.layers)

    input_nodes = [{"name": n} for n in net.node_names[0]]
    layers = [{"name": "INPUT", "nodes": input_nodes}]

    for layer_idx in range(num_layers):
        prev_layer_node_names = net.node_names[layer_idx]
        layer_node_names = net.node_names[layer_idx + 1]
        layer_backlinks = net.backlinks[layer_idx + 1]

        layer_nodes = []
        for node_idx, node in enumerate(net.layers[layer_idx]):
            prev_node_names = [prev_layer_node_names[bl] for bl in layer_backlinks[node_idx]]

            layer_nodes.append({"name": layer_node_names[node_idx], "backlinks": prev_node_names, "metadata": node.get_metadata()})

        layer_name = "OUTPUT" if layer_idx == num_layers - 1 else f"L{layer_idx+1}"
        layers.append({"name": layer_name, "nodes": layer_nodes})

    return {"layers": layers}