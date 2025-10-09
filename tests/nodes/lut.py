import numpy as np

from architecture.nodes.lut import make_lut_node
from architecture.utils import truth_table_patterns


def test_learns_simple_and():
    """
    LUT should perfectly reconstruct a simple AND function.
    """
    input_names = ["x_1", "x_2"]
    input_values = truth_table_patterns(2)
    target_values = (input_values[:, 0] & input_values[:, 1]).astype(bool)

    node = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=0)
    predictions = node(input_values)

    assert np.array_equal(predictions, target_values), "LUT failed to learn simple AND"


def test_majority_voting_under_noise():
    """
    With noisy labels, LUT should still follow the majority vote for each pattern.
    """
    rng = np.random.default_rng(0)
    input_names = ["x_1", "x_2"]
    input_values = np.tile(truth_table_patterns(2), (50, 1))
    target_values = (input_values[:, 0] ^ input_values[:, 1]).astype(bool)  # XOR

    noise = rng.random(len(target_values)) < 0.1
    target_values = np.where(noise, ~target_values, target_values)

    node = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=123)

    truth_patterns = truth_table_patterns(2)
    predictions = node(truth_patterns)

    true_xor = truth_patterns[:, 0] ^ truth_patterns[:, 1]
    assert (predictions == true_xor).sum() >= 3, "LUT failed to preserve majority rule under noise"


def test_unseen_patterns_filled_randomly():
    """
    Unseen or tie patterns should be resolved reproducibly using the seed.
    """
    rng = np.random.default_rng(0)
    input_names = ["x_1", "x_2", "x_3"]

    input_values = rng.integers(0, 2, size=(10, 3)).astype(bool)
    target_values = rng.integers(0, 2, size=10).astype(bool)

    node1 = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=42)
    node2 = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=42)

    assert np.array_equal(node1.node_predictions, node2.node_predictions), "Seed should control tie-breaking"

    node3 = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=43)
    assert not np.array_equal(node1.node_predictions, node3.node_predictions), "Different seed should alter tie-breaking"


def test_collapses_on_constant_targets():
    """
    If all target values are the same, LUT should predict a constant function.
    """
    input_names = ["x_1", "x_2"]
    input_values = truth_table_patterns(2)
    target_values = np.ones(input_values.shape[0], dtype=bool)

    node = make_lut_node("TEST_NODE", input_names, input_values, target_values, seed=0)
    predictions = node(input_values)

    assert predictions.all(), "Expected constant-True predictions"
    assert node.node_predictions.all(), "Expected LUT truth table to be constant True"
