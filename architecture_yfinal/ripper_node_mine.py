from __future__ import annotations
import numpy as np

import wittgenstein as lw
from sympy import symbols, lambdify
from sympy.logic.boolalg import Or, And, Not, simplify_logic

from .deep_binary_classifier import BinaryNode
from .utils import truth_table_indices, truth_table_patterns


class RipperNode(BinaryNode):
    def __init__(
            self,
            node_name: str,
            input_names: list[str],
            input_values: np.ndarray[np.bool_],
            target_values: np.ndarray[np.bool_],
            seed: int,
    ):
        """
        RIPPER-based binary node.

        Trains a RIPPER rule set on the given input data. Allows for expression reduction via sympy.

        :param node_name: The name of this node.
        :param input_names: The names of the input values in the column order of input_values.
        :param input_values: The input values for this node, shape (N, num_bits)
        :param target_values: The target values for this node, shape (N,)
        :param seed: A random seed for reproducibility.
        """
        super().__init__(node_name, input_names)
        self.seed = seed

        # Keep the original parent order to preserve stable ordering after simplification
        self._parent_order = list(input_names)

        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(input_values, target_values, feature_names=self.input_names, pos_class=True)

        self.expression = self.ruleset_to_expression(ripper.ruleset_)

        # Precompute LUT (and possibly shrink input_names)
        self._update_from_expr()

    @staticmethod
    def ruleset_to_expression(ruleset):
        assert ruleset, "ruleset_to_expression: got an empty/falsey ruleset."

        conjunctions = []
        for rule in ruleset:
            conjuncts = []
            for cond in rule.conds:
                feat = symbols(cond.feature)
                literal = feat if cond.val else Not(feat)
                conjuncts.append(literal)
            conjunctions.append(And(*conjuncts))  # a single rule is a conjunction of its literals

        disjunction = Or(*conjunctions)  # the whole ruleset is a disjunction of those conjunctions
        return disjunction

    def __call__(self, input_values: np.ndarray) -> np.ndarray:
        """
        input_values columns must match current self.input_names.
        """
        if len(self.input_names) == 0:
            return np.full(input_values.shape[0], bool(self.expr), dtype=bool)

        # Fail fast on misalignment
        if input_values.shape[1] != len(self.input_names):
            raise ValueError(
                f"{self.name}: got {input_values.shape[1]} cols, expected {len(self.input_names)} "
                f"({self.input_names})"
            )

        idxs = truth_table_indices(input_values)
        return self.pred_node[idxs]

    def get_truth_table(self):
        patterns = truth_table_patterns(len(self.input_names))[:, ::-1]
        patterns = patterns[::-1]   # flip rows only here
        table = np.column_stack((patterns, self.pred_node[::-1]))  # flip output too
        col_names = self.input_names + [f"{self.name} (output)"]
        return table, col_names

    def get_expr(self):
        return self.expr

    def reduce_expr(self):
        if not self.expr:
            return
        self.expr = simplify_logic(self.expr, form="dnf")
        self._update_from_expr()

    def _update_from_expr_old(self):
        if not self.expr:
            # Constant False (or no rules): zero-arity LUT
            self.input_names = []
            self.pred_node = np.array([False], dtype=bool)
            return

        # Determine used features; keep original parent order (no alphabetical resorting)
        used = {str(s) for s in getattr(self.expr, "free_symbols", set())}
        new_names = [nm for nm in self._parent_order if nm in used]
        self.input_names = new_names

        # Build LUT in that exact order
        if self.input_names:
            syms = [symbols(nm) for nm in self.input_names]
            patterns = truth_table_patterns(len(syms))[:, ::-1]   # no row flip here
            f = lambdify(syms, self.expr, "numpy")
            pred = f(*patterns.T)                          # boolean vector length 2^k
            self.pred_node = np.asarray(pred, dtype=bool)
        else:
            # Constant True
            self.pred_node = np.array([bool(self.expr)], dtype=bool)

    def _update_from_expr(self):
        if not self.expr:
            self.input_names = []
            self.pred_node = np.array([False], dtype=bool)
            return

        # All used features
        used = {str(s) for s in getattr(self.expr, "free_symbols", set())}

        # Keep stable parent order but only keep used ones
        self.input_names = [nm for nm in self._parent_order if nm in used]

        if self.input_names:
            syms = [symbols(nm) for nm in self.input_names]
            patterns = truth_table_patterns(len(syms))[:, ::-1]
            f = lambdify(syms, self.expr, "numpy")
            pred = f(*patterns.T)
            self.pred_node = np.asarray(pred, dtype=bool)
        else:
            self.pred_node = np.array([bool(self.expr)], dtype=bool)

def make_ripper_node(
        node_name: str,
        input_names: list[str],
        input_values: np.ndarray[np.bool_],
        target_values: np.ndarray[np.bool_],
        seed: int,
) -> RipperNode:
    return RipperNode(node_name, input_names, input_values, target_values, seed)
