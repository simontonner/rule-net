# =============================================================
# file: ripper_node_reduced.py
# Nodes that carry names only; no indices; outer network aligns columns
# =============================================================
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
        super().__init__(node_name, feature_names)

        self.seed = seed
        self.node_name = node_name

        # Train RIPPER
        X_df = pd.DataFrame(feature_values.astype(bool), columns=self.feature_names)
        y_df = pd.DataFrame({"y": target_values})
        ripper = lw.RIPPER(random_state=seed)
        ripper.fit(X_df, y_df, pos_class=True)

        # Convert ruleset → Sympy expression
        self.expr = self.ruleset_to_expr(ripper.ruleset_)

        # Precompute LUT
        self._update_from_expr()

    def __call__(self, X_local: np.ndarray) -> np.ndarray:
        """
        X_local columns must match self.feature_names. If the expression is
        constant (no variables), return a constant vector.
        """
        if len(self.feature_names) == 0:
            return np.full(X_local.shape[0], bool(self.expr), dtype=bool)
        idxs = truth_table_indices(X_local)
        return self.pred_node[idxs]

    def get_truth_table(self) -> np.ndarray:
        patterns = truth_table_patterns(len(self.feature_names))
        return np.column_stack((patterns, self.pred_node))

    def get_expr(self):
        return self.expr

    def reduce_expr(self):
        if not self.expr:
            return
        self.expr = simplify_logic(self.expr, form="dnf")
        self._update_from_expr()

    def _update_from_expr(self):
        if not self.expr:
            self.feature_names = []
            self.pred_node = np.array([False], dtype=bool)  # single-entry LUT for constants
            return

        # Extract used features from the expression and make them the new order
        used_syms = sorted([str(s) for s in self.expr.free_symbols])
        self.feature_names = used_syms

        # Build LUT over these names in this exact order
        syms = [symbols(nm) for nm in self.feature_names]
        if syms:
            patterns = truth_table_patterns(len(syms))
            f = lambdify(syms, self.expr, "numpy")
            pred = f(*patterns.T)
            self.pred_node = np.asarray(pred, dtype=bool)
        else:
            self.pred_node = np.array([bool(self.expr)], dtype=bool)

    @staticmethod
    def ruleset_to_expr(ruleset):
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


def make_ripper_node_reduced(
        node_name: str,
        feature_names: list[str],
        feature_values: np.ndarray,
        target_values: np.ndarray,
        seed: int,
) -> RipperNodeReduced:
    return RipperNodeReduced(node_name, feature_names, feature_values, target_values, seed)