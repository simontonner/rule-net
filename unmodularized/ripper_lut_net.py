####################################################################################################
# This code is mostly from Bernhard Gstrein
#
# It has been stripped down to the bare minimum needed for my experiments.
# The documentation has been exchanged with my own understanding of the code.
#
# There are some changes regarding the different modes the LUTs are generated.
####################################################################################################



import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import wittgenstein as lw


def get_lut_size(bits: int, layers: list):
    single_lut_size = 2 ** bits
    total_lut_size = 0
    for num_luts in layers:
        total_lut_size += num_luts * single_lut_size
    return total_lut_size / 8


def get_idxs(X, bit_pattern_tiled, N, bits):
    """
    This method looks up the input bits of the dataset in a table of bit patterns and returns the proper row index.

    Have a look at the get_bit_pattern method to see how the bit patterns are generated.
    """
    assert X.shape == (N, bits), "Shape of X has to be `(N, bits)`"
    return np.where(
        np.all(bit_pattern_tiled == np.repeat(X, 2 ** bits, axis=0), axis=1,).reshape(
            (N, 2 ** bits)
        )
        == True
    )[1]

def get_lut(indexes, labels, bits):
    """
    This method generates a lookup table from training samples.

    Instead of the full training table only the labels are given and an index to look up the input bits in a truth
    table. Look at the  get_bit_pattern method to see how the truth table is generated.

    The whole logic then just operates by counting the occurrences.
    """

    lut = np.bincount(indexes, weights=labels, minlength=2 ** bits)
    where_rnd = (lut == 0)
    np.put(lut, np.where(where_rnd)[0], np.random.choice([0, 1], size=(where_rnd).sum()))
    np.put(lut, np.where(lut < 0)[0], 0)
    np.put(lut, np.where(lut > 0)[0], 1)
    return np.hstack((lut.astype(bool), where_rnd))


def get_ripper(indexes, labels, bits):
    """
    The signature of this method is held in a way that it can be used as a replacement for the get_lut method.

    However, this requires us to generate the input bits again.

    The labels have a similar issue. Namely, that they have been converted to -1 and 1.
    """

    bit_pattern = get_bit_pattern(bits)
    columns = [f'bit{i}' for i in range(bits)]
    bit_pattern_df = pd.DataFrame(bit_pattern, columns=columns)

    # create a dataframe with the corresponding pattern for each index
    input_bits_df = bit_pattern_df.iloc[indexes]

    boolean_labels = labels == 1
    output_df = pd.DataFrame(boolean_labels, columns=['output'])

    ripper_clf = lw.RIPPER()
    ripper_clf.fit(input_bits_df, output_df)

    # get the lookup table but for ripper using the bit_pattern_df
    ripper_preds = ripper_clf.predict(bit_pattern_df)

    # due to compatibility reasons, we need to generate the where_rnd array. we set all entries to False
    where_rnd = np.zeros(2 ** bits, dtype=bool)

    output = np.hstack((ripper_preds, where_rnd))

    return output


def get_bit_pattern(bits):
    """
    Generates the input bits in the schema of a typical truth table.

    Example for 2 bits:
    [[False, False],
     [False, True],
     [True, False],
     [True, True]]
    """

    bit_pattern = np.empty((2 ** bits, bits), dtype=bool)
    for i in range(2 ** bits):
        bit_string = np.binary_repr(i, width=bits)
        bit_pattern[i] = np.array(list(bit_string), dtype=int).astype(bool)

    return bit_pattern


class RipperLut:
    def __init__(self, bits, hidden_layers=[], verbose=False, mode='lut'):

        assert len(hidden_layers) > 0, "Single LUT not supported"

        self.bits = bits
        self.hidden_layers = hidden_layers
        assert len(bits) == len(hidden_layers) + 1

        self.cols_arr_ = []     # holds the indices to link back to the selected columns in the previous layer
        self.lut_arr_ = []      # holds only the output columns for each lookup table
        self.rnd_arr_ = []      # holds a mask for each output column indicating the tie cases

        self.verbose = verbose

        self.mode = mode

        if self.mode == 'ripper':
            self.find_rule = get_ripper
        elif self.mode == 'lut':
            self.find_rule = get_lut
        else:
            raise ValueError(f"Mode {self.mode} not supported")

        self.luts_trained = False

    def train(self, X, y):
        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        assert y.dtype == bool, f"Dtype of y has to be bool, got {y.dtype}"
        N = X.shape[0]

        # convert truth values to -1 and 1
        y_ = y.copy().astype(int)
        y_[y_ == 0] = -1
        y_[y_ == 1] = 1

        pool = multiprocessing.Pool()

        # train layer by layer
        with tqdm(self.hidden_layers, disable=not self.verbose) as t:

            for j, num_luts in enumerate(t):
                # get bit pattern and tile it to match each training sample
                bit_pattern = get_bit_pattern(self.bits[j])
                bit_pattern_tiled = np.tile(bit_pattern, (N, 1))

                # select random columns for the current layer and remember their indices
                cols = np.random.choice(X.shape[1] if j == 0 else self.hidden_layers[j - 1], size=num_luts*self.bits[j]).reshape((num_luts, self.bits[j]))
                self.cols_arr_.append(cols)

                # get the indices according to the get_idxs method (in parallel for each set of random columns)
                idxs = np.array(
                    pool.starmap(
                        get_idxs,
                        [
                            [
                                X[:, self.cols_arr_[j][i]]
                                if j == 0
                                else X_[:, self.cols_arr_[j][i]],
                                bit_pattern_tiled,
                                N,
                                self.bits[j],
                            ]
                            for i in range(num_luts)
                        ],
                    )
                )

                # the starmap code has been replaced here
                tmp = np.array(
                    pool.starmap(
                        self.find_rule,
                        [[idxs[i], y_, self.bits[j]] for i in range(num_luts)],
                    )
                )

                # remember the output columns and the tie cases
                self.lut_arr_.append(tmp[:, :2**self.bits[j]].copy())
                self.rnd_arr_.append(tmp[:, 2**self.bits[j]:].copy())

                # update the prediction for the next layer (don't yet fully know why)
                if "X_" in locals():
                    X_prev_ = X_.copy() # previous prediction

                X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T # current prediction

        # the last layer consists of a single LUT (exemplifies what the code would look like without the thread pool)
        cols = np.random.choice(self.hidden_layers[-1], size=self.bits[-1])
        self.cols_arr_.append(cols)
        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(
            X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1]
        )
        tmp = self.find_rule(idxs, y_, self.bits[-1])
        self.lut_arr_.append(tmp[:2**self.bits[-1]].copy())
        self.rnd_arr_.append(tmp[2**self.bits[-1]:].copy())
        preds_train = self.lut_arr_[-1][idxs]

        return preds_train


    def predict(self, X):
        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        N = X.shape[0]

        # iteratively updates the prediction for each layer until we reach the end
        pool = multiprocessing.Pool()
        for j, num_luts in enumerate(self.hidden_layers):
            bit_pattern = get_bit_pattern(self.bits[j])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = np.array(
                pool.starmap(
                    get_idxs,
                    [
                        [
                            X[:, self.cols_arr_[0][i]]
                            if j == 0
                            else X_[:, self.cols_arr_[j][i]],
                            bit_pattern_tiled,
                            N,
                            self.bits[j],
                        ]
                        for i in range(num_luts)
                    ],
                )
            )
            X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T

        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1])
        preds = self.lut_arr_[-1][idxs]

        return preds

    def get_accuracies_per_layer(self, X, y):
        """
        Similar to the predict method, but also calculates the accuracy for each layer.
        """

        assert X.dtype == bool, f"Dtype of X has to be bool, got {X.dtype}"
        N = X.shape[0]

        if len(self.hidden_layers) == 0:
            X_ = X

        pool = multiprocessing.Pool()
        acc = []
        for j, num_luts in enumerate(tqdm(self.hidden_layers)):
            bit_pattern = get_bit_pattern(self.bits[j])
            bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
            idxs = np.array(
                pool.starmap(
                    get_idxs,
                    [
                        [
                            X[:, self.cols_arr_[0][i]]
                            if j == 0
                            else X_[:, self.cols_arr_[j][i]],
                            bit_pattern_tiled,
                            N,
                            self.bits[j],
                        ]
                        for i in range(num_luts)
                    ],
                )
            )
            X_ = np.array([self.lut_arr_[j][i][idxs[i]] for i in range(num_luts)]).T
            acc_layer = []
            for i in range(X_.shape[1]):
                acc_layer.append(accuracy_score(X_[:, i], y))
            acc.append(acc_layer)

        bit_pattern = get_bit_pattern(self.bits[-1])
        bit_pattern_tiled = np.tile(bit_pattern, (N, 1))
        idxs = get_idxs(X_[:, self.cols_arr_[-1]], bit_pattern_tiled, N, self.bits[-1])
        preds = self.lut_arr_[-1][idxs]
        acc.append([accuracy_score(preds, y)])
        return preds, acc