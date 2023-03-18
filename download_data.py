import scipy
import numpy as np
from mlxtend.data import loadlocal_mnist


def alpha_digit(indexes, path="data/binaryalphadigs.mat"):
    mat = scipy.io.loadmat(path)
    new_mat = mat["dat"][indexes]
    new_mat = new_mat.reshape(new_mat.shape[0] * new_mat.shape[1])

    new_new_mat = np.zeros(
        (new_mat.shape[0], new_mat[0].shape[0] * new_mat[0].shape[1])
    )
    for i in range(len(new_mat)):
        new_new_mat[i] = new_mat[i].reshape(new_mat[i].shape[0] * new_mat[i].shape[1])

    return new_new_mat


def mnist_data(path="data/"):
    X_train, y_train = loadlocal_mnist(
        images_path=path + "train-images.idx3-ubyte",
        labels_path=path + "train-labels.idx1-ubyte",
    )
    X_test, y_test = loadlocal_mnist(
        images_path=path + "t10k-images.idx3-ubyte",
        labels_path=path + "t10k-labels.idx1-ubyte",
    )

    # Binarisation of our data
    X_train = (X_train >= 127).astype(int)
    X_test = (X_test >= 127).astype(int)

    return X_train, X_test, y_train, y_test
