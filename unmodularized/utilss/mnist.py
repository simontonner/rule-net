####################################################################################################
# This code is from Bernhard Gstrein
####################################################################################################


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import struct
from array import array

def load_binary(img_path, lbl_path):

    # open file in binary read mode
    with open(lbl_path, "rb") as f:
        # first 8 bytes contain metadata as two 32-bit integers in big-endian byte order
        magic, size = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number should be 2049, got {magic}"

        labels = array("B", f.read())

    with open(img_path, "rb") as f:
        # first 16 bytes contain metadata as four integers
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number should be 2051, got {magic}"

        image_data = array("B", f.read())

    # initialize a list to store the image bytes
    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    # populate the list with the image bytes
    for i in range(size):
        images[i][:] = image_data[i * rows * cols : (i + 1) * rows * cols]

    return images, labels


def load_mnist_binary(thresh=0.5):

    train_img_path = r"data/mnist/train-images-idx3-ubyte"
    train_lbl_path = r"data/mnist/train-labels-idx1-ubyte"

    train_img, train_lbl = load_binary(train_img_path, train_lbl_path)

    test_img_path = r"data/mnist/t10k-images-idx3-ubyte"
    test_lbl_path = r"data/mnist/t10k-labels-idx1-ubyte"

    test_img, test_lbl = load_binary(test_img_path, test_lbl_path)

    # stack them to process them together
    X_ = np.vstack((train_img, test_img))
    y_ = np.hstack((train_lbl, test_lbl))

    # scale the image pixel values to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_tf = scaler.fit_transform(X_)

    # binarize the image pixel values
    X = (X_tf > thresh).astype(bool)
    y = (y_ == 5) | (y_ == 6) | (y_ == 7) | (y_ == 8) | (y_ == 9)

    X_train = X[:60_000]
    X_test = X[60_000:]
    y_train = y[:60_000]
    y_test = y[60_000:]

    return X_train, X_test, y_train, y_test
