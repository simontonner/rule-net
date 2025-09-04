from __future__ import annotations
import numpy as np
import pandas as pd
import wittgenstein as lw

from sympy import symbols, lambdify
from sympy.logic.boolalg import Or, And, Not, simplify_logic

from .deep_binary_classifier import BaseNode
from .utils import truth_table_indices, truth_table_patterns


class RipperNodeReduced(BaseNode):
    def __init__(
            self,
            X_cols: np.ndarray,
            X_node: np.ndarray,
            y_node: np.ndarray,
            seed: int,
            layer_idx: int,
            node_idx: int,
    ):
        """
        RIPPER-based lookup node (reduced).

        Trains a RIPPER rule set on the given input data, converts it to a
        Sympy Boolean expression, and allows reducing the expression to
        remove backlinks and unused features.

        :param X_cols: Indices of input features for this node (relative to previous layer)
        :param X_node: Input data for this node, shape (N, num_bits)
        :param y_node: Target labels for this node, shape (N,)
        :param seed: Random seed for reproducibility
        :param layer_idx: Index of the current layer
        :param node_idx: Index of the node within this layer
        """
        self.seed = seed
        self.layer_idx = layer_idx
        self.node_idx = node_idx
        self.name = f"L{layer_idx}N{node_idx}"

        # initialize BaseNode with both X_cols and name
        super().__init__(X_cols, self.name)

        # feature names from previous layer
        self.input_names = [f"L{layer_idx-1}N{j}" for j in X_cols]

        # train RIPPER
        X_df = pd.DataFrame(X_node.astype(bool), columns=self.input_names)
        y_df = pd.DataFrame({"y": y_node})
        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df, pos_class=True)

        # store expression in Sympy form
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

        used_syms = sorted(self.expr.free_symbols, key=str)
        # Extract back the input indices from names like "L0N24" or "L2N3"
        self.X_cols = np.array(
            [int(str(s).split("N")[1]) for s in used_syms],
            dtype=int,
        )

        # Precompute truth table predictions
        patterns = truth_table_patterns(len(self.X_cols))
        f = lambdify(used_syms, self.expr, "numpy")
        pred = f(*patterns.T)
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
                feat = symbols(cond.feature)  # should already be like "L{layer-1}N{j}"
                terms.append(feat if cond.val else Not(feat))
            exprs.append(And(*terms))
        return Or(*exprs)


def make_ripper_node_reduced(
        X_cols: np.ndarray,
        X_node: np.ndarray,
        y_node: np.ndarray,
        seed: int,
        layer_idx: int,
        node_idx: int,
) -> RipperNodeReduced:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return RipperNodeReduced(X_cols, X_node, y_node, seed, layer_idx, node_idx)
