from __future__ import annotations
import numpy as np
import sympy as sp

from architecture.nodes.base import BinaryNode
from architecture.utils import truth_table_patterns, truth_table_indices


class QuineNode(BinaryNode):
    def __init__(
            self,
            node_name: str,
            input_names: list[str],
            input_values: np.ndarray[np.bool_],
            target_values: np.ndarray[np.bool_],
            seed: int = 0,
    ):
        super().__init__(node_name, input_names)
        self.seed = seed

        if input_values.dtype != bool or target_values.dtype != bool:
            raise TypeError("input_values and target_values must be boolean arrays")

        num_bits = input_values.shape[1]
        if num_bits != len(input_names):
            raise ValueError("len(input_names) must equal number of columns in input_values")

        # Majority voting using the SAME index convention as __call__()
        target_pm   = target_values.astype(np.int8) * 2 - 1
        pattern_idx = truth_table_indices(input_values)
        votes       = np.bincount(pattern_idx, weights=target_pm, minlength=2**num_bits)

        on_idx  = np.flatnonzero(votes > 0)    # majority True
        off_idx = np.flatnonzero(votes < 0)    # majority False
        dc_idx  = np.flatnonzero(votes == 0)   # ties/unseen

        vars_     = [sp.Symbol(n) for n in self.input_names]
        all_rows  = truth_table_patterns(num_bits)      # (2**n, n), big-endian, descending
        on_rows   = all_rows[on_idx].tolist()
        off_rows  = all_rows[off_idx].tolist()
        dc_rows   = all_rows[dc_idx].tolist()

        # Candidate expressions
        expr_sop = sp.SOPform(vars_, on_rows, dc_rows)
        expr_pos = sp.POSform(vars_, off_rows, dc_rows)

        def _score(e: sp.BooleanFunction) -> int:
            try:
                return sp.count_ops(e, visual=False)
            except Exception:
                return len(str(e))

        # Evaluate both candidates on the full domain and keep only those satisfying constraints
        tt_full_cols = all_rows.T  # columns per variable
        candidates = []
        for tag, e in (("sop", expr_sop), ("pos", expr_pos)):
            f = sp.lambdify(vars_, e, "numpy")
            pred = np.asarray(f(*tt_full_cols), dtype=bool)
            ok = (pred[on_idx].all() if on_idx.size else True) and ((~pred[off_idx]).all() if off_idx.size else True)
            if ok:
                candidates.append((tag, _score(e), e, pred))

        if not candidates:
            # Fallback: enforce constraints explicitly and synthesize exact LUT
            # Try DC assignments implied by each candidate; pick the smaller result
            fallback_options = []
            for tag, e in (("sop", expr_sop), ("pos", expr_pos)):
                f = sp.lambdify(vars_, e, "numpy")
                pred = np.asarray(f(*tt_full_cols), dtype=bool)
                # Enforce majority constraints
                pred_onoff = pred.copy()
                pred_onoff[on_idx] = True
                pred_onoff[off_idx] = False
                on_rows_final = all_rows[np.flatnonzero(pred_onoff)].tolist()
                # Build minimal SOP for the *exact* LUT (no don't-cares)
                e_final = sp.SOPform(vars_, on_rows_final, [])
                fallback_options.append((_score(e_final), e_final, pred_onoff))
            fallback_options.sort(key=lambda t: t[0])
            _, chosen_expr, chosen_pred = fallback_options[0]
            base_expr = chosen_expr
            pred_full = chosen_pred
        else:
            # Choose the smallest valid candidate
            candidates.sort(key=lambda t: t[1])  # by score
            _, _, base_expr, pred_full = candidates[0]

        # Optional multi-level simplification (keeps function)
        self.expression = sp.simplify_logic(base_expr, force=True)

        # Provenance
        self.tie_indices = dc_idx

        # Finalize: prune unused inputs and freeze LUT on reduced domain
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
        """Canonicalize to DNF (optional); predictions must remain identical."""
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
    return QuineNode(node_name, input_names, input_values, target_values, seed)
