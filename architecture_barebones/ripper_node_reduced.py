from __future__ import annotations
import numpy as np
import pandas as pd
import wittgenstein as lw

from sympy import symbols, lambdify
from sympy.logic.boolalg import Or, And, Not, simplify_logic

from .deep_binary_classifier import BaseNode
from .utils import truth_table_indices, truth_table_patterns


class RipperNodeReduced(BaseNode):
    def __init__(self, X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int):
        """
        RIPPER-based lookup node (reduced).

        Trains a RIPPER rule set on the given input data, converts it to a
        Sympy Boolean expression, and allows reducing the expression to
        remove backlinks and unused features.

        :param X_cols: The indices referencing the original columns of the dataset, shape (num_bits,)
        :param X_node: The input data for this node, shape (N, num_bits)
        :param y_node: The target labels for this node, shape (N,)
        :param seed: Random seed for reproducibility
        """
        super().__init__(X_cols)

        self.seed = seed

        # wittgenstein requires DataFrames
        features = [f"x_{i}" for i in X_cols]
        X_df = pd.DataFrame(X_node.astype(bool), columns=features)
        y_df = pd.DataFrame({"y": y_node})

        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df, pos_class=True)

        self.expr = self.ruleset_to_expr(ripper.ruleset_)
        self._update_from_expr()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        idxs = truth_table_indices(X[:, self.X_cols])
        return self.pred_node[idxs]

    def get_truth_table(self) -> np.ndarray:
        """
        Generates the full truth-table on-demand since storing it in each node would be wasteful.

        :return: Boolean array, shape (2**num_bits, num_bits + 1)
        """
        patterns = truth_table_patterns(len(self.X_cols))
        return np.column_stack((patterns, self.pred_node))

    def get_expr(self):
        """
        Returns the current Sympy Boolean expression of the ruleset.

        :return: Sympy Boolean expression
        """
        return self.expr

    def reduce_expr(self):
        """
        Simplifies the stored Boolean expression into DNF using Sympy's simplify_logic.
        Also updates feature indices and prediction table accordingly.
        """
        if not self.expr:
            return
        self.expr = simplify_logic(self.expr, form="dnf")
        self._update_from_expr()

    def _update_from_expr(self):
        """
        Updates X_cols and the truth-table predictions from the current expression.
        """
        if not self.expr:
            self.X_cols = np.array([], dtype=int)
            self.pred_node = np.array([], dtype=bool)
            return

        # active features
        used_syms = sorted(self.expr.free_symbols, key=lambda s: int(str(s).split("_")[1]))
        self.X_cols = np.array([int(str(s).split("_")[1]) for s in used_syms], dtype=int)

        # precompute truth table predictions with lambdify
        patterns = truth_table_patterns(len(self.X_cols))
        f = lambdify(used_syms, self.expr, "numpy")  # returns a vectorized numpy function
        pred = f(*patterns.T)  # unpack columns as arguments
        self.pred_node = np.asarray(pred, dtype=bool)

    @staticmethod
    def ruleset_to_expr(ruleset):
        """
        Converts a Wittgenstein ruleset into a Sympy Boolean expression.

        :param ruleset: Wittgenstein ruleset
        :return: Sympy Boolean expression
        """
        if not ruleset:
            return False

        exprs = []
        for rule in ruleset:
            terms = []
            for cond in rule.conds:
                feat = symbols(cond.feature)
                terms.append(feat if cond.val else Not(feat))
            exprs.append(And(*terms))
        return Or(*exprs)


def make_ripper_node_reduced(X_cols: np.ndarray, X_node: np.ndarray, y_node: np.ndarray, seed: int) -> RipperNodeReduced:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return RipperNodeReduced(X_cols, X_node, y_node, seed)