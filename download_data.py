import scipy
import numpy as np


def lire_alpha_digit(indexes, path="data/binaryalphadigs.mat"):
    mat = scipy.io.loadmat(path)
    new_mat = mat["dat"][indexes]
    new_mat = new_mat.reshape(new_mat.shape[0] * new_mat.shape[1])

    new_new_mat = np.zeros(
        (new_mat.shape[0], new_mat[0].shape[0] * new_mat[0].shape[1])
    )
    for i in range(len(new_mat)):
        new_new_mat[i] = new_mat[i].reshape(new_mat[i].shape[0] * new_mat[i].shape[1])

    return new_new_mat
