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

        Parameters
        ----------
        node_name : str
            The name of this node.
        input_names : list[str]
            Original feature names (column order of `input_values`). This order is preserved.
        input_values : bool array, shape (N, num_bits)
            Input values for this node.
        target_values : bool array, shape (N,)
            Target labels.
        seed : int
            RNG seed for RIPPER and reproducibility.
        """
        super().__init__(node_name, input_names)
        self.seed = seed

        # Keep the original parent order to preserve stable ordering after simplification
        self._parent_order: list[str] = list(input_names)
        self._parent_index = {n: i for i, n in enumerate(self._parent_order)}

        ripper = lw.RIPPER(random_state=seed)
        # Train directly on arrays; tell RIPPER the feature names (in the original order)
        ripper.fit(input_values, target_values, feature_names=self._parent_order, pos_class=True)

        # Convert ruleset → Sympy expression
        self.expr = self.ruleset_to_expr(ripper.ruleset_)

        # Precompute LUT (and shrink used feature set, but keep parent-relative order)
        self._update_from_expr()

    # -------------------------
    # Public API
    # -------------------------
    @property
    def feature_names(self) -> list[str]:
        print("Hitting new one")
        """Original feature order passed at initialization."""
        return list(self._parent_order)

    def test_node(self):
        print("Hitting new one")

    @property
    def used_feature_names(self) -> list[str]:
        """Ordered subset of features actually used by the (possibly simplified) expression."""
        return list(self.input_names)

    def __call__(self, input_values: np.ndarray, feature_order: list[str] | None = None) -> np.ndarray:
        """
        Evaluate the node on a batch.

        Parameters
        ----------
        input_values : bool array, shape (N, M)
            If `feature_order` is None, columns must match `self.input_names` exactly.
            If `feature_order` is provided, the columns are assumed to follow that order
            and will be reindexed internally to `self.input_names`.
        feature_order : list[str] | None
            Column order of `input_values`. If provided, we'll pick the columns corresponding
            to `self.input_names` in that order.

        Returns
        -------
        out : bool array, shape (N,)
        """
        if len(self.input_names) == 0:
            # Constant expression: broadcast the constant
            return np.full(input_values.shape[0], bool(self.expr), dtype=bool)

        X = input_values
        if feature_order is not None:
            # Reindex columns from provided order -> node's expected order
            idxs = [feature_order.index(nm) for nm in self.input_names]
            X = X[:, idxs]
        elif input_values.shape[1] != len(self.input_names):
            raise ValueError(
                f"{self.name}: got {input_values.shape[1]} cols, expected {len(self.input_names)} "
                f"in order {self.input_names}. If you have a different column order, pass `feature_order=...`."
            )

        # Map rows to LUT indices (must match the same bit-order convention used in _update_from_expr)
        lut_idxs = truth_table_indices(X)
        return self.pred_node[lut_idxs]

    def get_truth_table(self, feature_order: list[str] | None = None):
        """
        Build a truth table in a specified column order (default: the node's used features).

        Parameters
        ----------
        feature_order : list[str] | None
            If None, uses `self.input_names` (ordered subset).
            If provided, must be a subset of the original features (we only use those that are in the expression).

        Returns
        -------
        table : bool array, shape (2**k, k+1)
            Truth table patterns followed by predicted output column.
        col_names : list[str]
            Column names for the table: chosen feature_order + [f"{self.name} (output)"].
        """
        if not self.expr:
            # Constant False: if no features are used, produce a 1-row table
            names = self.input_names if feature_order is None else [n for n in feature_order if n in self.input_names]
            if len(names) == 0:
                table = np.array([[False]], dtype=bool)
                return table, [f"{self.name} (output)"]

        # Choose names to show in the table (ordered)
        names = self.input_names if feature_order is None else [n for n in feature_order if n in set(self.input_names)]
        # Re-evaluate the expression *in this order* so the table matches the requested feature order
        syms = [symbols(n) for n in names]
        patterns = truth_table_patterns(len(syms))[:, ::-1]  # keep the same bit convention as __update LUT
        f = lambdify(syms, self.expr, "numpy")
        pred = np.asarray(f(*patterns.T), dtype=bool)

        table = np.column_stack((patterns, pred))
        col_names = names + [f"{self.name} (output)"]
        return table, col_names

    def get_expr(self):
        return self.expr

    def reduce_expr(self):
        """Simplify the boolean expression (DNF) and rebuild LUT w.r.t. the same parent order."""
        if not self.expr:
            return
        self.expr = simplify_logic(self.expr, form="dnf")
        self._update_from_expr()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _update_from_expr(self):
        """
        Build `self.input_names` as parent-ordered subset of features actually used by the expression,
        and (re)build the LUT `self.pred_node` in that order.
        """
        if not self.expr:
            self.input_names = []
            self.pred_node = np.array([False], dtype=bool)
            return

        used = {str(s) for s in getattr(self.expr, "free_symbols", set())}
        # Keep stable parent order, but only keep used ones
        self.input_names = [nm for nm in self._parent_order if nm in used]

        if self.input_names:
            syms = [symbols(nm) for nm in self.input_names]
            # IMPORTANT: this column order + the same convention in truth_table_indices() must match
            patterns = truth_table_patterns(len(syms))[:, ::-1]
            f = lambdify(syms, self.expr, "numpy")
            pred = f(*patterns.T)  # boolean vector length 2**k
            self.pred_node = np.asarray(pred, dtype=bool)
        else:
            # Constant True
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


def make_ripper_node(
        node_name: str,
        input_names: list[str],
        input_values: np.ndarray[np.bool_],
        target_values: np.ndarray[np.bool_],
        seed: int,
) -> RipperNode:
    return RipperNode(node_name, input_names, input_values, target_values, seed)
