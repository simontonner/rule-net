from __future__ import annotations
import numpy as np

import wittgenstein as lw
from sympy import symbols, lambdify
from sympy.logic.boolalg import Or, And, Not, false, simplify_logic

from architecture.nodes.base import BinaryNode
from architecture.utils import truth_table_patterns


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

        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(input_values, target_values, feature_names=self.input_names, pos_class=True)

        self.expression = self._ruleset_to_expression(ripper.ruleset_)
        self._update_node()

    def get_metadata(self):
        return { "type": "ripper", "seed": self.seed, "expression": str(self.expression) }

    def get_expression(self):
        return self.expression

    def reduce_expression(self):
        if not self.expression:
            return
        self.expression = simplify_logic(self.expression, form="dnf")
        self._update_node()

    def _update_node(self):
        # remove inputs that are not used in the expression
        used_input_names = {str(s) for s in getattr(self.expression, "free_symbols", set())}
        self.input_names = [nm for nm in self.input_names if nm in used_input_names]

        # the sympy expression is applied over the columns of a truth table
        truth_table_columns = truth_table_patterns(len(self.input_names)).T

        # evaluate expression to get new truth table outputs
        expression_symbols = [symbols(nm) for nm in self.input_names]
        expression_function = lambdify(expression_symbols, self.expression, "numpy")
        expression_prediction = expression_function(*truth_table_columns)

        self.node_predictions = np.asarray(expression_prediction, dtype=bool)

    @staticmethod
    def _ruleset_to_expression(ruleset):
        if not ruleset:
            return false    # return sympy false to match rippers default behaviour on empty rulesets

        conjunctions = []
        for rule in ruleset:
            literals = []
            for cond in rule.conds:
                feature = symbols(cond.feature)
                literal = feature if cond.val else Not(feature)
                literals.append(literal)

            conjunction = And(*literals)
            conjunctions.append(conjunction)

        disjunction = Or(*conjunctions)
        return disjunction


def make_ripper_node(
        node_name: str,
        input_names: list[str],
        input_values: np.ndarray[np.bool_],
        target_values: np.ndarray[np.bool_],
        seed: int,
) -> RipperNode:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return RipperNode(node_name, input_names, input_values, target_values, seed)
