import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, p, q):
        self.RBM_a = np.zeros(p)
        self.RBM_b = np.zeros(q)
        self.RBM_W = np.sqrt(0.1) * np.random.randn(p, q)

        self.q = q
        self.p = p

        self.epoch_rbm = []
        self.X_rec_list = []
        self.X_list = []
        self.errors_list = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def entree_sortie_RBM(self, X):
        return self.sigmoid(X @ self.RBM_W + self.RBM_b)

    def sortie_entree_RBM(self, H):
        return self.sigmoid(H @ self.RBM_W.T + self.RBM_a)

    def train_RBM(self, X, epsilon, batch_size, nb_epochs):
        self.epoch_rbm = []
        self.X_rec_list = []
        self.errors_list = []
        self.X_list = []
        errors_all = []

        for epoch in range(nb_epochs):
            X_copy = X.copy()
            np.random.shuffle(X_copy)
            for batch in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[batch : batch + batch_size]
                true_batch_size = X_batch.shape[0]
                v_0 = X_batch
                p_h_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = (np.random.rand(true_batch_size, self.q) < p_h_v_0) * 1
                p_v_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = (np.random.rand(true_batch_size, self.p) < p_v_h_0) * 1
                p_h_v_1 = self.entree_sortie_RBM(v_1)

                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
                grad_W = np.dot(v_0.T, p_h_v_0) - np.dot(v_1.T, p_h_v_1)

                self.RBM_a += epsilon / true_batch_size * grad_a
                self.RBM_b += epsilon / true_batch_size * grad_b
                self.RBM_W += epsilon / true_batch_size * grad_W

            H = self.entree_sortie_RBM(X)
            X_rec = self.sortie_entree_RBM(H)
            errors_all.append(np.sum((X - X_rec) ** 2) / X.shape[0])

            if epoch % 20 == 0:
                rand_idx = np.random.randint(X.shape[0])
                # if (X[rand_idx].shape[0]==320):
                self.epoch_rbm.append(f"Epoch{epoch}/{nb_epochs}")
                self.errors_list.append(
                    np.sum((X[rand_idx] - X_rec[rand_idx]) ** 2) / X[rand_idx].shape[0]
                )
                self.X_rec_list.append(X_rec[rand_idx])
                self.X_list.append(X[rand_idx])
        return errors_all

    def generer_image_RBM(self, nb_data, nb_gibbs):
        for _ in range(nb_data):
            v = (np.random.rand(self.p) < 1 / 2) * 1
            for _ in range(nb_gibbs):
                h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1
            v = np.reshape(v, (20, 16))
            plt.imshow(v, cmap="gray")
            plt.show()

    def generer_image_RBM_without_plot(self, nb_gibbs):
        v = (np.random.rand(self.p) < 1 / 2) * 1
        for _ in range(nb_gibbs):
            h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
            v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1
        return v

    def display_image_RBM_vs_original(self):
        for i in range(len(self.errors_list)):
            print(self.epoch_rbm[i])
            fig, ax = plt.subplots(1, 2, figsize=(5, 3))
            ax[0].imshow(self.X_list[i].reshape(20, 16), cmap="gray")
            ax[0].axis("off")
            ax[0].set_title("Original_image", fontsize=7)

            ax[1].imshow(self.X_rec_list[i].reshape(20, 16), cmap="gray")
            ax[1].axis("off")
            ax[1].set_title(f"RBM_image : RMSE = {self.errors_list[i]}", fontsize=7)

            fig.tight_layout()
            plt.show()

    def features_extracted(self, row, col):
        indexes = [
            np.random.randint(0, self.RBM_W.shape[0]) for idx in range(row * col)
        ]

        fig, axes = plt.subplots(row, col, figsize=(7, 5))
        for i, idx in enumerate(indexes):
            ax = axes[i // col, i % col]
            ax.imshow(self.RBM_W[:, idx].reshape(20, 16), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{idx}")

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.show()

    def generate_for_analysis(self, nb_gibbs, col=5, row=1):
        # p, q = self.RBM_W.shape
        nb_data = row * col

        fig = plt.figure(figsize=(5, 3))
        grid = plt.GridSpec(row, col, wspace=0.1, hspace=0.1)

        for i in range(nb_data):
            v = (np.random.rand(self.p) < 1 / 2) * 1
            for _ in range(nb_gibbs):
                h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1

            v = np.reshape(v, (20, 16))
            ax = plt.subplot2grid((row, col), (i // col, i % col), fig=fig)
            ax.imshow(v, cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
