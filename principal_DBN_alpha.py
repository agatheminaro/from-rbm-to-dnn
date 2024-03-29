from principal_RBM_alpha import RBM
import numpy as np
import matplotlib.pyplot as plt


class DBN:
    def __init__(self, config):
        self.epoch_dbn = []
        self.X_rec_list = []
        self.X_list = []
        self.errors = []

        self.RBM_list = []
        self.config = config

        for i in range(len(self.config) - 1):
            self.RBM_list.append(RBM(config[i], config[i + 1]))

    def train_DBN(self, X, epsilon, batch_size, nb_epochs, verbose=False):
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.losses = []
        X_copy = X.copy()
        if verbose:
            print(f"Training DBN with {len(self.RBM_list)} RBM(s)")
        for i, rbm in enumerate(self.RBM_list):
            if verbose:
                print(f"Training RBM {i+1} / {len(self.RBM_list)}")
            rbm, loss = rbm.train_RBM(
                X=X_copy,
                epsilon=self.epsilon,
                batch_size=self.batch_size,
                nb_epochs=nb_epochs,
                verbose=verbose,
            )
            X_copy = rbm.entree_sortie_RBM(X_copy)
            self.losses.append(loss)
        return self, self.losses

    def generer_image_DBN(self, nb_data, nb_gibbs, type_data="alpha_digits"):
        if type_data == "alpha_digits":
            reshape_size = (20, 16)
        elif type_data == "mnist_digits":
            reshape_size = (28, 28)

        for i in range(nb_data):
            v = self.RBM_list[-1].generer_image_RBM_without_plot(nb_gibbs)

            for j in reversed(range(len(self.RBM_list) - 1)):
                v = self.RBM_list[j].sortie_entree_RBM(v)

            v = np.reshape(v, reshape_size)
            plt.subplot(nb_data // 5, 5, i + 1)
            plt.imshow(v, cmap="gray")
            plt.axis("off")
        plt.show()

    def generate_for_analysis_DBN(
        self, nb_gibbs, col=5, row=1, param_analysed="epsilon", nb_digit=None
    ):
        nb_data = row * col

        fig = plt.figure(figsize=(5, 3))
        grid = plt.GridSpec(row, col, wspace=0.1, hspace=0.1)

        for i in range(nb_data):
            v = self.RBM_list[-1].generer_image_RBM_without_plot(nb_gibbs)

            for i in reversed(range(len(self.RBM_list) - 1)):
                v = self.RBM_list[i].sortie_entree_RBM(v)

            v = np.reshape(v, (20, 16))
            ax = plt.subplot2grid((row, col), (i // col, i % col), fig=fig)
            ax.imshow(v, cmap="gray")
            ax.axis("off")

        if param_analysed == "epsilon":
            plt.title(f"RBM_image : epsilon = {self.epsilon}", fontsize=7, loc="left")
        elif param_analysed == "batch_size":
            plt.title(
                f"RBM_image : batch_size = {self.batch_size}", fontsize=7, loc="left"
            )
        elif param_analysed == "config":
            plt.title(
                f"RBM_image : nb_layer = {len(self.config)}", fontsize=7, loc="left"
            )
        elif param_analysed == "nb_data":
            plt.title(f"RBM_image : nb_data = {nb_digit}", fontsize=7, loc="left")
        plt.tight_layout()
        plt.show()
