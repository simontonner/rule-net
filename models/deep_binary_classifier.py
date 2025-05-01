import numpy as np
import multiprocessing as mp


def truth_table_patterns(num_bits: int) -> np.ndarray:
    """
    Generates bit-patterns like the inputs of a typical truth table.

    :param num_bits: Number of bits in the input
    :return: Boolean array with every possible bit-string, shape (2**num_bits, num_bits)
    """
    row_indices = np.arange(2 ** num_bits, dtype=np.uint32).reshape(-1, 1)
    bit_weights = 1 << np.arange(num_bits - 1, -1, -1, dtype=np.uint32)  # big-endian
    bit_patterns = (row_indices & bit_weights) > 0
    return bit_patterns


def truth_table_indices(bit_patterns: np.ndarray) -> np.ndarray:
    """
    Converts bit patterns to their corresponding row indices in a truth table.

    :param bit_patterns: Boolean array of bit patterns, shape (N, num_bits)
    :return: Integer array of row indices, shape (N,)
    """
    bit_weights = 1 << np.arange(bit_patterns.shape[1] - 1, -1, -1, dtype=np.uint32)
    row_indices = (bit_patterns.astype(np.uint32) * bit_weights).sum(axis=1)
    return row_indices


class DeepBinaryClassifier:
    def __init__(self, nodes_per_layer, bits_per_node, concurrent_nodes=True, rng=None):
        """
        :param nodes_per_layer: List of integers, number of nodes in each layer
        :param bits_per_node: List of integers, number of bits for each node
        :param concurrent_nodes: If True, use multiprocessing for parallel computation
        :param rng: Random seed for reproducibility
        """
        self.nodes_per_layer = nodes_per_layer
        self.bits_per_node = bits_per_node
        self.concurrent_nodes = concurrent_nodes
        self.rng = np.random.default_rng(rng)




# ----------------------------------------------------------------------
# generic network
# ----------------------------------------------------------------------
class BaseNet:
    def __init__(self, bits_per_layer, luts_per_layer, rng=None, use_mp=False):
        assert len(bits_per_layer) == len(luts_per_layer) + 1
        self.bits   = bits_per_layer
        self.widths = luts_per_layer
        self.rng    = np.random.default_rng(rng)
        self.use_mp = use_mp

        # learned parameters
        self._cols   = []       # per layer: (width, bits) int  which inputs feed each LUT
        self._tables = []       # list[ np.ndarray(bool, len=2**bits) ]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.dtype != bool or y.dtype != bool:
            raise TypeError("X and y must be bool")

        y_pm1      = y.astype(np.int8) * 2 - 1          # 0/1 → −1/+1
        layer_out  = X
        pool       = mp.Pool() if self.use_mp else None

        # ---- hidden layers -------------------------------------------
        for width, bits in zip(self.widths, self.bits[:-1]):
            prev_dim    = layer_out.shape[1]
            cols_layer  = self.rng.choice(prev_dim, size=(width, bits), replace=True)
            self._cols.append(cols_layer)

            # indices for every LUT in this layer
            idxs_all = [bits_to_index(layer_out[:, c]) for c in cols_layer]

            if pool:
                tables = pool.starmap(
                    self._build_table,
                    [(idxs_all[i], y_pm1, bits) for i in range(width)]
                )
            else:
                tables = [self._build_table(idxs_all[i], y_pm1, bits)
                          for i in range(width)]

            self._tables.extend(tables)

            # compute outputs of this layer
            layer_next = np.empty((layer_out.shape[0], width), dtype=bool)
            for i, lut in enumerate(tables):
                layer_next[:, i] = lut[idxs_all[i]]
            layer_out = layer_next                            # feed forward

        # ---- final single LUT ----------------------------------------
        bits_last  = self.bits[-1]
        cols_final = self.rng.choice(layer_out.shape[1], size=bits_last, replace=True)
        self._cols.append(cols_final)
        idxs_final = bits_to_index(layer_out[:, cols_final])
        table_last = self._build_table(idxs_final, y_pm1, bits_last)
        self._tables.append(table_last)
        preds_train = table_last[idxs_final]

        if pool:
            pool.close(); pool.join()
        return preds_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != bool:
            raise TypeError("X must be bool")

        layer_out = X
        t_ptr = 0                                             # walk through tables

        for width, bits, cols in zip(self.widths, self.bits[:-1], self._cols[:-1]):
            nxt = np.empty((layer_out.shape[0], width), dtype=bool)
            for lut_i, c in enumerate(cols):
                lut         = self._tables[t_ptr]; t_ptr += 1
                nxt[:, lut_i] = lut[bits_to_index(layer_out[:, c])]
            layer_out = nxt

        lut = self._tables[t_ptr]
        return lut[bits_to_index(layer_out[:, self._cols[-1]])]

    # ------------- override in a sub-class ----------------------------
    def _build_table(self, idxs, labels_pm1, bits):
        raise NotImplementedError
