import warnings

import numpy as np
import pandas as pd

from sympy import symbols, lambdify

from architecture.nodes.ripper import make_ripper_node
from architecture.utils import truth_table_patterns


rng = np.random.default_rng(0)


def test_finds_rules_for_parity_targets():
    """
    Even though the target values come from a parity function, the node will try to find some spurious rule.
    """
    input_names = [f"x_{i}" for i in range(1, 4)]
    input_values = truth_table_patterns(len(input_names))
    target_values = (input_values.sum(axis=1) % 2).astype(bool)

    node = make_ripper_node("TEST_NODE", input_names, input_values, target_values, seed=42)

    aligned_inputs = rng.integers(0, 2, size=(128, len(node.input_names))).astype(bool)

    predictions = node(aligned_inputs)
    assert predictions.any() and (~predictions).any(), "Parity should not collapse to constant"
    assert len(node.input_names) > 0, "Expected at least one input retained"


def test_finds_rules_for_noisy_targets():
    """
    Even though the target values are pure noise, the node will try to find some spurious rule.
    """
    rng = np.random.default_rng(0)
    input_names = [f"bit_{i}" for i in range(1, 5)]
    input_values = rng.integers(0, 2, size=(1024, len(input_names))).astype(bool)
    target_values = rng.integers(0, 2, size=1024).astype(bool)

    node = make_ripper_node("TEST_NODE", input_names, input_values, target_values, seed=0)

    aligned_inputs = rng.integers(0, 2, size=(256, len(node.input_names))).astype(bool)

    predictions = node(aligned_inputs)
    assert predictions.any() and (~predictions).any(), "Noise should not produce constant predictions"
    assert len(node.input_names) > 0, "Expected RIPPER to keep some inputs"


def test_collapses_on_homogeneous_targets():
    """
    If all target_values are True, the node collapses to constant False and removes all backlinks.
    """
    input_names = [f"x_{i}" for i in range(1, 5)]
    input_values = truth_table_patterns(len(input_names))
    target_values = np.ones(input_values.shape[0], dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        node = make_ripper_node("TEST_NODE", input_names, input_values, target_values, seed=42)

    assert node.input_names == [], f"Expected no backlinks, got {node.input_names}"
    assert not node.node_predictions.any(), "Expected constant-False predictions"


def test_cuts_dependency_to_homogeneous_inputs():
    """
    If one input bit is constant over the whole dataset, the node should drop its dependency on it.
    """
    rng = np.random.default_rng(0)
    input_names = [f"bit_{i}" for i in range(1, 5)]
    input_values = rng.integers(0, 2, size=(1024, len(input_names))).astype(bool)
    input_values[:, 1] = True

    target_values = (input_values[:, 0] & input_values[:, 1]) | (input_values[:, 2] & ~input_values[:, 3])
    flips = rng.random(1024) < 0.01
    target_values = np.where(flips, ~target_values, target_values).astype(bool)

    node = make_ripper_node("TEST_NODE", input_names, input_values, target_values, seed=0)

    predictions = node(input_values[:, [input_names.index(n) for n in node.input_names]])
    assert predictions.any() and (~predictions).any(), "Should not collapse to constant"
    assert "bit_2" not in node.input_names, f"Expected constant bit_2 to be dropped, got {node.input_names}"


def test_reduced_expression_matches_previous_predictions():
    """
    After reduction, the new node expression should produce the same predictions as before.
    """
    dataset_df = pd.read_csv("../test_dataset.csv")
    input_names = [c for c in dataset_df.columns if c != "target"]
    input_values = dataset_df[input_names].to_numpy(bool)
    target_values = dataset_df["target"].to_numpy(bool)

    node = make_ripper_node("L1N0", input_names, input_values, target_values, seed=42)
    node.reduce_expression()

    truth_table, column_names = node.get_truth_table()
    truth_table_inputs = [truth_table[:, column_names.index(n)] for n in node.input_names]
    tt_output = truth_table[:, -1].astype(bool)

    expr_symbols = [symbols(n) for n in node.input_names]
    expr_func = lambdify(expr_symbols, node.get_expression(), "numpy")
    expr_output = expr_func(*truth_table_inputs).astype(bool)

    assert np.array_equal(expr_output, tt_output), "Reduced expression does not reproduce truth table outputs"
