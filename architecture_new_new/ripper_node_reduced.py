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
            node_name: str,
            feature_names: list[str],
            feature_values: np.ndarray,
            target_values: np.ndarray,
            seed: int,
    ):
        """
        RIPPER-based lookup node (reduced).

        Trains a RIPPER rule set on the given input data, converts it to a
        Sympy Boolean expression, and allows reducing the expression to
        remove backlinks and unused features.

        :param node_name: Unique identifier for this node (e.g., "L2N3")
        :param feature_names: Names of the input features, shape (num_bits,)
        :param feature_values: Boolean input data for this node, shape (N, num_bits)
        :param target_values: Boolean target labels for this node, shape (N,)
        :param seed: Random seed for reproducibility
        """
        super().__init__(np.arange(len(feature_names)), node_name)

        self.seed = seed
        self.node_name = node_name
        self.feature_names = list(feature_names)

        # Train RIPPER
        X_df = pd.DataFrame(feature_values.astype(bool), columns=self.feature_names)
        y_df = pd.DataFrame({"y": target_values})
        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df, pos_class=True)

        # Convert ruleset → Sympy expression
        self.expr = self.ruleset_to_expr(ripper.ruleset_)

        # Precompute LUT
        self._update_from_expr()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predictions for the given input data using the
        precomputed lookup table.

        :param X: Input data containing columns for all feature_names, shape (N, num_bits)
        :return: Predictions, shape (N,)
        """
        idxs = truth_table_indices(X[:, self.X_cols])
        return self.pred_node[idxs]

    def get_truth_table(self) -> np.ndarray:
        """
        Generates the full truth table for this node.

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
        Also updates the feature indices and prediction table accordingly.
        """
        if not self.expr:
            return
        self.expr = simplify_logic(self.expr, form="dnf")
        self._update_from_expr()

    def _update_from_expr(self):
        """
        Updates X_cols, feature_names, and the truth-table predictions
        from the current expression.
        """
        if not self.expr:
            self.X_cols = np.array([], dtype=int)
            self.feature_names = []
            self.pred_node = np.array([], dtype=bool)
            return

        # Extract used features from the expression
        used_syms = sorted([str(s) for s in self.expr.free_symbols])

        # Map back to original indices
        name_to_idx = {nm: i for i, nm in enumerate(self.feature_names)}
        used_idxs = [name_to_idx[nm] for nm in used_syms]

        # Update X_cols and feature_names to the reduced set
        self.X_cols = np.array(used_idxs, dtype=int)
        self.feature_names = used_syms  # now only the used ones

        # Precompute truth table predictions
        syms = [symbols(nm) for nm in used_syms]
        patterns = truth_table_patterns(len(self.X_cols))
        if syms:
            f = lambdify(syms, self.expr, "numpy")
            pred = f(*patterns.T)
        else:
            pred = np.full(2 ** len(self.X_cols), bool(self.expr))
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
                feat = symbols(cond.feature)  # should already be named properly
                terms.append(feat if cond.val else Not(feat))
            exprs.append(And(*terms))
        return Or(*exprs)


def make_ripper_node_reduced(
        node_name: str,
        feature_names: list[str],
        feature_values: np.ndarray,
        target_values: np.ndarray,
        seed: int,
) -> RipperNodeReduced:
    """
    Simple node factory to be handed over to the `DeepBinaryClassifier`.
    """
    return RipperNodeReduced(node_name, feature_names, feature_values, target_values, seed)
