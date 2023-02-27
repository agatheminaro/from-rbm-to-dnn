from principal_DBN_alpha import DBN
import numpy as np
import matplotlib.pyplot as plt

class DNN:
    def __init__(self, config):
        self.DBN = DNN(config)
        self.classification_layer = []
        
    
    def pretrain_DNN(self, X, epsilon, batch_size, nb_epochs):
        self.DBN.train_DBN(X, epsilon, batch_size, nb_epochs)

    def calcul_softmax(self, RBM, X):
        return np.exp(X @ self.RBM_W + self.RBM_b)/np.sum(np.exp(X @ self.RBM_W + self.RBM_b), axis=1)


    def entree_sortie_reseau(self, X):
        sortie_list = []
        entree = X
        for rbm in self.DBN.RBM_list:
            entree = rbm.entree_sortie_RBM(entree)
            sortie_list.append(entree)
        
        unite_de_sortie = self.calcul_softmax(self.DBN.RBM_list[-1], sortie_list[-1])

        return sortie_list, unite_de_sortie

    def retropropagation(self, X, Y, epsilon, batch_size, nb_epochs):
        for epoch in range(nb_epochs):
            X_copy, Y_copy = shuffle(X, Y)
            for batch in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[batch : min(batch + batch_size, X_copy.shape[0])]
                Y_batch = Y_copy[batch : min(batch + batch_size, X_copy.shape[0])]
                true_batch_size = X_batch.shape[0]

                sortie_list, unite_de_sortie = self.entree_sortie_reseau(X_batch)

                delta_sortie = unite_de_sortie - Y_batch

                #TODO

    def generer_image_DBN(self, nb_data, nb_gibbs):
        for i in range(nb_data):
            v = self.RBM_list[-1].generer_image_RBM_without_plot(nb_gibbs)

            for i in reversed(range(len(self.RBM_list)-1)):
                v = self.RBM_list[i].sortie_entree_RBM(v)

            v = np.reshape(v, (20, 16))
            plt.imshow(v, cmap="gray")
            plt.show()

        
    


        