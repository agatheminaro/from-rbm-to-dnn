import numpy as np
import matplotlib.pyplot as plt


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


def subsample_data_set(X, y, nb_data=None):
    if nb_data is None:
        return X, y
    else:
        indexes = np.random.permutation(X.shape[0])
        return X[indexes[:nb_data]], y[indexes[:nb_data]]
