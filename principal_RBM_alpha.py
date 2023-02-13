import numpy as np
import matplotlib.pyplot as plt


class RBM:
    def __init__(self, p, q):
        self.RBM_a = np.zeros(p)
        self.RBM_b = np.zeros(q)
        self.RBM_W = np.sqrt(0.1) * np.random.randn(p, q)
        self.q = q
        self.p = p

    def entree_sortie_RBM(self, X):
        return 1 / (1 + np.exp(-X @ self.RBM_W + self.RBM_b))

    def sortie_entree_RBM(self, H):
        return 1 / (1 + np.exp(-H @ self.RBM_W.T + self.RBM_a))

    def train_RBM(self, X, epsilon, batch_size, nb_epochs):
        for epoch in range(nb_epochs):
            X_copy = X.copy()
            np.random.shuffle(X_copy)
            for batch in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[batch : min(batch + batch_size, X_copy.shape[0])]
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
            print(np.sum((X - X_rec) ** 2) / X.shape[0])

    def generer_image_RBM(self, nb_data, nb_gibbs):
        for i in range(nb_data):
            v = (np.random.rand(self.p) < 1 / 2) * 1
            for _ in range(nb_gibbs):
                h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1
            v = np.reshape(v, (20, 16))
            plt.imshow(v, cmap="gray")
            plt.show()
