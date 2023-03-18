import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch


def show_alpha_digits(data, row, col):
    indexes = [np.random.randint(0, data.shape[0]) for _ in range(row * col)]
    fig, axes = plt.subplots(row, col, figsize=(10, 7))
    for i, idx in enumerate(indexes):
        ax = axes[i // col, i % col]
        ax.imshow(data[idx].reshape(20, 16), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"sample {idx}")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()


def show_mnist_digits(X, y, row, col):
    fig, axes = plt.subplots(row, col, figsize=(8, 5))
    for i in range(row * col):
        ax = axes[i // 4, i % 4]
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Digit {np.argmax(y[i])}")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()


def subsample_data_set(X, y, nb_data=None):
    if nb_data is None:
        return X, y
    else:
        indexes = np.random.permutation(X.shape[0])
        return X[indexes[:nb_data]], y[indexes[:nb_data]]


def create_dataloader(X, batch_size=32):
    X_tensor = torch.Tensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
