from __future__ import annotations
import numpy as np
import sympy as sp

from architecture.nodes.base import BinaryNode
from architecture.utils import truth_table_patterns


class QuineNode(BinaryNode):
    def __init__(
            self,
            node_name: str,
            input_names: list[str],
            input_values: np.ndarray[np.bool_],
            target_values: np.ndarray[np.bool_],
            seed: int = 0,
    ):
        """
        Quine–McCluskey-based binary node.

        Builds a LUT via majority voting and resolves ties deterministically
        using Boolean minimization (SOP/POS).

        :param node_name: The name of this node.
        :param input_names: The names of the input values in the column order of input_values.
        :param input_values: The input values for this node, shape (N, num_bits).
        :param target_values: The target values for this node, shape (N,).
        :param seed: Random seed for API symmetry (not used).
        """
        super().__init__(node_name, input_names)
        self.seed = seed

        if input_values.dtype != bool or target_values.dtype != bool:
            raise TypeError("input_values and target_values must be boolean arrays")

        num_bits = input_values.shape[1]
        if num_bits != len(input_names):
            raise ValueError("len(input_names) must equal number of columns in input_values")

        # majority voting
        target_pm = target_values.astype(np.int8) * 2 - 1
        ints = np.packbits(input_values, axis=1, bitorder="little").ravel()
        votes = np.bincount(ints, weights=target_pm, minlength=2**num_bits)

        on_idx  = np.flatnonzero(votes > 0)
        off_idx = np.flatnonzero(votes < 0)
        dc_idx  = np.flatnonzero(votes == 0)

        # Boolean minimization
        vars_ = [sp.Symbol(n) for n in self.input_names]
        expr_sop = sp.SOPform(vars_, on_idx.tolist(), dc_idx.tolist())
        expr_pos = sp.POSform(vars_, off_idx.tolist(), dc_idx.tolist())

        def _score(e: sp.BooleanFunction) -> int:
            try:
                return sp.count_ops(e, visual=False)
            except Exception:
                return len(str(e))

        expr = expr_sop if _score(expr_sop) <= _score(expr_pos) else expr_pos
        self.expression = sp.simplify_logic(expr, force=True)

        self.tie_indices = dc_idx
        self._update_node()

    def get_metadata(self):
        return {
            "type": "quine",
            "seed": self.seed,
            "tie_indices": self.tie_indices.tolist(),
            "expression": str(self.expression),
        }

    def get_expression(self):
        return self.expression

    def reduce_expression(self):
        if not self.expression:
            return
        self.expression = sp.simplify_logic(self.expression, form="dnf")
        self._update_node()

    def _update_node(self):
        used_input_names = {str(s) for s in getattr(self.expression, "free_symbols", set())}
        self.input_names = [nm for nm in self.input_names if nm in used_input_names]

        truth_table_columns = truth_table_patterns(len(self.input_names)).T
        expression_symbols = [sp.Symbol(nm) for nm in self.input_names]
        expression_function = sp.lambdify(expression_symbols, self.expression, "numpy")
        expression_prediction = expression_function(*truth_table_columns)

        self.node_predictions = np.asarray(expression_prediction, dtype=bool)


def make_quine_node(
        node_name: str,
        input_names: list[str],
        input_values: np.ndarray[np.bool_],
        target_values: np.ndarray[np.bool_],
        seed: int = 0,
) -> QuineNode:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return QuineNode(node_name, input_names, input_values, target_values, seed)