import numpy as np


def truth_table_patterns(num_bits: int) -> np.ndarray:
    """
    Generates bit-patterns like the inputs of a typical truth table.

    :param num_bits: Number of bits in the input
    :return: Boolean array with every possible bit-string, shape (2**num_bits, num_bits)
    """
    row_indices = np.arange(2 ** num_bits, dtype=np.uint32).reshape(-1, 1)
    bit_weights = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)  # big-endian
    bit_patterns = (row_indices & bit_weights) > 0
    return bit_patterns


def truth_table_indices(bit_patterns: np.ndarray) -> np.ndarray:
    """
    Converts bit patterns to their corresponding row indices in a truth table.

    :param bit_patterns: Boolean array of bit patterns, shape (N, num_bits)
    :return: Integer array of row indices, shape (N,)
    """
    bit_weights = 1 << np.arange(bit_patterns.shape[1] - 1, -1, -1, dtype=np.uint32)
    row_indices = (bit_patterns.astype(np.uint32) * bit_weights).sum(axis=1)
    return row_indices.astype(np.int64)