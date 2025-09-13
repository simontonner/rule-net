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