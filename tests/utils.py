import numpy as np

from architecture.utils import truth_table_patterns, truth_table_indices


rng = np.random.default_rng(42)


def test_truth_table_patterns_three_bits():
    """
    truth_table_patterns(3) should return the standard schema of a truth table with 3 bits.
    """
    reference_patterns = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    test_patterns = truth_table_patterns(3)
    assert np.array_equal(reference_patterns, test_patterns), "truth_table_patterns mismatch for 3 bits"


def test_truth_table_indices_inverse_mapping():
    """
    truth_table_indices should correctly map arbitrary bit patterns back into the reference truth table.
    """
    num_bits = 4
    reference_patterns = truth_table_patterns(num_bits)

    random_patterns = rng.integers(0, 2, size=(100, num_bits), dtype=np.uint8).astype(bool)

    indices = truth_table_indices(random_patterns)
    looked_up = reference_patterns[indices]

    assert np.array_equal(random_patterns, looked_up), "truth_table_indices did not map patterns back to reference"
