import numpy as np
import multiprocessing
from tqdm import tqdm


def get_lut_size(bits: int, layers: list):
    single_lut_size = 2 ** bits
    total_lut_size = 0
    for num_luts in layers:
        total_lut_size += num_luts * single_lut_size
    return total_lut_size / 8


def get_idxs(X, bit_pattern_tiled, N, bits):
    assert X.shape == (N, bits), "Shape of X has to be `(N, bits)`"
    return np.where(
        np.all(bit_pattern_tiled == np.repeat(X, 2 ** bits, axis=0), axis=1).reshape((N, 2 ** bits)) == True
    )[1]


def get_lut(indexes, labels, bits, rng):
    lut = np.bincount(indexes, weights=labels, minlength=2 ** bits)
    where_rnd = (lut == 0)
    np.put(lut, np.where(where_rnd)[0], rng.choice([0, 1], size=where_rnd.sum()))
    np.put(lut, np.where(lut < 0)[0], 0)
    np.put(lut, np.where(lut > 0)[0], 1)
    return np.hstack((lut.astype(bool), where_rnd))


def get_bit_pattern(bits):
    bit_pattern = np.empty((2 ** bits, bits), dtype=bool)
    for i in range(2 ** bits):
        bit_string = np.binary_repr(i, width=bits)
        bit_pattern[i] = np.array(list(bit_string), dtype=int).astype(bool)
    return bit_pattern


class RipperLut:  # Name retained for compatibility
    def __init__(self, bits, hidden_layers=[], verbose=False, mode='lut', seed: int | None = None):
        assert len(hidden_layers) > 0, "Single LUT not supported"
        self.bits = bits
        self.hidden_layers = hidden_layers
        assert len(bits) == len(hidden_layers) + 1

        self.cols_arr_ = []
        self.lut_arr_ = []
        self.rnd_arr_ = []

        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.find_rule = get_lut
        self.luts_trained = False

    def train(self, X, y):
        assert X.dtype == bool and y.dtype == bool
        N = X.shape[0]

        y_ = y.copy().astype(int)
        y_[y_ == 0] = -1
        y_[y_ == 1] = 1

        pool = multiprocessing.Pool()

        with tqdm(self.hidden_layers, disable=not self.verbose) as t:
            for j, num_luts in enumerate(t):
                bit_pattern = get_bit_pattern(self.bits[j])
                bit_pattern_tiled = np.tile(bit_pattern, (N, 1))

                cols = self.rng.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1],
                                       size=num_luts * self.bits[j]).reshape((num_luts, self.bits[j]))
                self.cols_arr_.append(cols)

                idxs = np.array(pool.starmap(
                    get_idxs,
                    [
                        [
                            X[:, cols[i]] if j == 0 else X_[:, cols[i]],
                            bit_pattern_tiled,
                            N,
                            self.bits[j],
                        ]
                        for i in range(num_luts)
                    ],
                ))

                tmp = np.array(pool.starmap(
                    self.find_rule,
                    [[idxs[i], y_, self.bits[j], self.rng] for i in range(num_luts)],
                ))

                self.lut_arr_.append(tmp[:, :2**self.bits[j]].copy())
                self.rnd_arr_.append(tmp[:, 2**self.bits[j]:].copy())
                X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T

        cols = self.rng.choice(self.hidden_layers[-1], size=self.bits[-1])
        self.cols_arr_.append(cols)
        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, cols], bit_pattern_tiled, N, self.bits[-1])
        tmp = self.find_rule(idxs, y_, self.bits[-1], self.rng)
        self.lut_arr_.append(tmp[:2**self.bits[-1]].copy())
        self.rnd_arr_.append(tmp[2**self.bits[-1]:].copy())

        return self.lut_arr_[-1][idxs]

    def predict(self, X):
        assert X.dtype == bool
        N = X.shape[0]

        pool = multiprocessing.Pool()
        for j, num_luts in enumerate(self.hidden_layers):
            bit_pattern = get_bit_pattern(self.bits[j])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = np.array(pool.starmap(
                get_idxs,
                [
                    [
                        X[:, self.cols_arr_[0][i]] if j == 0 else X_[:, self.cols_arr_[j][i]],
                        bit_pattern_tiled,
                        N,
                        self.bits[j],
                    ]
                    for i in range(num_luts)
                ],
            ))
            X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T

        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1])
        return self.lut_arr_[-1][idxs]
