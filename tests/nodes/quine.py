import numpy as np
import sympy as sp

from architecture.nodes.quine import make_quine_node
from architecture.utils import truth_table_patterns


rng = np.random.default_rng(0)


# --- helpers (mirror QuineNode's internal indexing & votes) ---

def _pattern_indices_little_endian(bits_bool: np.ndarray) -> np.ndarray:
    """
    Map boolean rows (N, n_bits) to integers using the same convention as QuineNode:
    np.packbits(..., bitorder='little') -> LSB-first (first column = 2**0).
    """
    packed = np.packbits(bits_bool, axis=1, bitorder="little")
    # Assumes n_bits <= 8 in these tests (single byte); safe for our cases.
    return packed[:, 0].astype(np.int64)


def _vote_vector(input_values: np.ndarray, target_values: np.ndarray, n_bits: int) -> np.ndarray:
    pm = target_values.astype(np.int8) * 2 - 1
    idx = _pattern_indices_little_endian(input_values)
    return np.bincount(idx, weights=pm, minlength=2**n_bits)


# --- tests ---

def test_prunes_to_single_relevant_input_when_target_is_single_bit():
    """
    If target == first input bit, QuineNode should minimize to a single symbol
    and drop all other inputs. We don't enforce *which* symbol (SymPy may pick
    an equivalent one under don't-cares).
    """
    input_names = ["a", "b", "c"]
    X = rng.integers(0, 2, size=(512, 3)).astype(bool)
    y = X[:, 0]  # depends only on 'a'

    node = make_quine_node("Q1", input_names, X, y, seed=0)

    # Should keep only one input
    assert len(node.input_names) == 1, f"Expected 1 input, got {node.input_names}"

    # And that one input should reproduce the target relation
    kept = node.input_names[0]
    kept_idx = input_names.index(kept)
    assert np.array_equal(node.node_predictions, truth_table_patterns(1)[:, 0]), \
        f"Kept input {kept} does not behave like a direct passthrough"



def test_tie_indices_match_expected_seen_and_unseen():
    """
    Construct a dataset with explicit ties (equal pos/neg) and some unseen patterns.
    QuineNode.tie_indices should match all of them.
    """
    n_bits = 3
    input_names = [f"x{i}" for i in range(n_bits)]

    # Build counts per pattern (LSB-first indexing)
    # 000 -> 3x negative
    # 001 -> tie (2 pos, 2 neg)
    # 010 -> tie (4 pos, 4 neg)
    # 011 -> 3x positive
    # 100 -> tie (1 pos, 1 neg)
    # 101,110,111 -> unseen (ties)
    rows = []
    labels = []

    def add(idx, pos, neg):
        bits = ((np.array([(idx >> k) & 1 for k in range(n_bits)], dtype=bool))).astype(bool)
        bits = np.tile(bits, (pos + neg, 1))
        y = np.array([True] * pos + [False] * neg, dtype=bool)
        rows.append(bits)
        labels.append(y)

    add(0b000, pos=0, neg=3)
    add(0b001, pos=2, neg=2)
    add(0b010, pos=4, neg=4)
    add(0b011, pos=3, neg=0)
    add(0b100, pos=1, neg=1)

    X = np.vstack(rows)
    y = np.concatenate(labels)

    node = make_quine_node("Q2", input_names, X, y, seed=0)

    votes = _vote_vector(X, y, n_bits)
    expected_ties = set(np.flatnonzero(votes == 0).tolist())
    actual_ties = set(node.tie_indices.tolist())

    assert actual_ties == expected_ties, f"Tie indices mismatch: {actual_ties} vs {expected_ties}"


def test_predictions_equal_expression_over_truth_domain():
    """
    node.node_predictions must equal evaluating node.expression across the full truth table
    (for the *reduced* input set).
    """
    input_names = ["a", "b", "c"]
    X = truth_table_patterns(3)  # full coverage, no ties
    y = (X[:, 0] ^ X[:, 1] ^ X[:, 2]).astype(bool)  # 3-bit parity

    node = make_quine_node("Q3", input_names, X, y, seed=0)

    # Evaluate expression across the pruned domain
    expr_syms = [sp.Symbol(nm) for nm in node.input_names]
    tt_cols = truth_table_patterns(len(node.input_names)).T
    f = sp.lambdify(expr_syms, node.get_expression(), "numpy")
    expr_out = np.asarray(f(*tt_cols), dtype=bool)

    assert np.array_equal(expr_out, node.node_predictions), "Expression and stored truth table differ"


def test_majority_constraints_hold_on_seen_patterns():
    """
    For patterns that appear with a non-zero majority, the resulting LUT must agree with that majority.
    """
    n_bits = 2
    input_names = ["a", "b"]

    # Build:
    # 10 -> majority True (5 pos, 1 neg)
    # 01 -> majority False (1 pos, 5 neg)
    # Others left as don't-cares
    rows = []
    labels = []

    def mk_row(bits, pos, neg):
        arr = np.tile(np.array(bits, dtype=bool), (pos + neg, 1))
        y = np.array([True] * pos + [False] * neg, dtype=bool)
        return arr, y

    r, l = mk_row([True, False], 5, 1)   # 10
    rows.append(r); labels.append(l)
    r, l = mk_row([False, True], 1, 5)   # 01
    rows.append(r); labels.append(l)

    X = np.vstack(rows)
    y = np.concatenate(labels)

    node = make_quine_node("Q4", input_names, X, y, seed=0)

    # Evaluate the LUT on the *original* two-bit grid restricted to the kept inputs
    # Then read off the values for the two specific patterns we constrained.
    # Build truth table for current (possibly pruned) inputs and recompute pattern map.
    kept = node.input_names
    kept_idx = [input_names.index(nm) for nm in kept]

    # These two queries are expressed in the kept variable order:
    q_10 = np.array([[True, False]])[:, kept_idx[:len(kept)]]
    q_01 = np.array([[False, True]])[:, kept_idx[:len(kept)]]

    # If some variable was pruned away, adjust queries accordingly by slicing 'kept_idx'.
    q_10 = q_10[:, :len(kept)]
    q_01 = q_01[:, :len(kept)]

    pred_10 = node(q_10).item()
    pred_01 = node(q_01).item()

    assert pred_10 is True, "Pattern 10 should map to True by majority"
    assert pred_01 is False, "Pattern 01 should map to False by majority"


def test_reduce_expression_preserves_predictions():
    """
    Calling reduce_expression (canonicalization to DNF in current impl) must not change the LUT.
    """
    input_names = ["b0", "b1", "b2", "b3"]
    X = rng.integers(0, 2, size=(1024, 4)).astype(bool)
    # Target uses a structured rule so minimization has something to chew on.
    y = (X[:, 0] & ~X[:, 1]) | (X[:, 2] & X[:, 3])

    node = make_quine_node("Q5", input_names, X, y, seed=0)
    before = node.node_predictions.copy()

    node.reduce_expression()
    after = node.node_predictions

    assert np.array_equal(before, after), "Canonicalization changed frozen LUT predictions"


def test_dtype_and_shape_validation():
    input_names = ["a", "b"]
    X_bool = rng.integers(0, 2, size=(32, 2)).astype(bool)
    y_bool = rng.integers(0, 2, size=32).astype(bool)

    # dtype check
    try:
        make_quine_node("Q6", input_names, X_bool.astype(int), y_bool, seed=0)
        assert False, "Expected TypeError for non-boolean input_values"
    except TypeError:
        pass

    # shape/name mismatch
    try:
        make_quine_node("Q7", ["a", "b", "c"], X_bool, y_bool, seed=0)
        assert False, "Expected ValueError for len(input_names) != num_bits"
    except ValueError:
        pass
