from principal_DBN_alpha import DBN
import numpy as np
import matplotlib.pyplot as plt


class DNN(DBN):
    def __init__(self, config):
        super().__init__(config)
        self.classification_layer = []

    def pretrain_DNN(self, X, epsilon, batch_size, nb_epochs, verbose=False):
        return self.train_DBN(X, epsilon, batch_size, nb_epochs, verbose)

    def calcul_softmax(self, rbm, X):
        Z = X @ rbm.RBM_W + rbm.RBM_b
        exp_z = np.exp(Z)
        return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)

    def cross_entropy(self, y_true, y_pred):
        loss = []
        for k in range(y_true.shape[0]):
            loss.append(
                np.sum(
                    [
                        -y_true[k, j] * np.log(y_pred[k, j])
                        for j in range(y_true.shape[1])
                    ]
                )
            )
        return loss

    def entree_sortie_reseau(self, X):
        sortie_list = [X]
        for rbm in self.RBM_list[:-1]:
            sortie = rbm.entree_sortie_RBM(sortie_list[-1])
            sortie_list.append(sortie)
        sortie_list.append(self.calcul_softmax(self.RBM_list[-1], sortie_list[-1]))
        return sortie_list

    def retropropagation(self, X, y, epsilon, batch_size, nb_epochs, verbose=True):
        loss = []
        self.epsilon = epsilon
        self.batch_size = batch_size
        if verbose:
            print(f"Training DNN")
        for nb_epoch in range(nb_epochs):
            loss_batch = []
            X_copy = X.copy()
            y_copy = y.copy()
            for batch in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[batch : min(batch + batch_size, X_copy.shape[0])]
                y_batch = y_copy[batch : min(batch + batch_size, X_copy.shape[0])]
                true_batch_size = X_batch.shape[0]
                sortie_list = self.entree_sortie_reseau(X_batch)
                last_layer = self.RBM_list[-1]
                delta_sortie = sortie_list[-1] - y_batch

                # Update weights
                last_layer.RBM_W -= (
                    epsilon * (sortie_list[-2].T @ delta_sortie) / true_batch_size
                )
                last_layer.RBM_b -= (
                    epsilon * np.mean(delta_sortie, axis=0) / true_batch_size
                )

                # Update hidden layers
                for i, rbm in reversed(list(enumerate(self.RBM_list[:-1]))):
                    delta_sortie = (delta_sortie @ self.RBM_list[i + 1].RBM_W.T) * (
                        sortie_list[i + 1] * (1 - sortie_list[i + 1])
                    )
                    rbm.RBM_W -= (
                        epsilon * (sortie_list[i].T @ delta_sortie) / true_batch_size
                    )
                    rbm.RBM_b -= (
                        epsilon * np.mean(delta_sortie, axis=0) / true_batch_size
                    )

                loss_batch += self.cross_entropy(y_batch, sortie_list[-1])

            loss.append(np.mean(loss_batch))

            if nb_epoch % 25 == 0 and verbose:
                print(f"    Epoch {nb_epoch}/{nb_epochs}: loss is {loss[-1]}")

        return self, loss

    def test_dnn(self, X, y, verbose=True):
        for rbm in self.RBM_list[:-1]:
            X = rbm.entree_sortie_RBM(X)
        predictions = np.argmax(self.calcul_softmax(self.RBM_list[-1], X), axis=1)
        error_rate = 100 * (
            1 - np.sum(predictions == np.argmax(y, axis=1)) / y.shape[0]
        )
        if verbose:
            print("Error rate: ", error_rate)
        return error_rate
