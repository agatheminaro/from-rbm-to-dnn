from principal_RBM_alpha import RBM
import numpy as np
import matplotlib.pyplot as plt

class DBN:
    def __init__(self, config):
        self.RBM_list = []
        for i in range(len(config) - 1):
            self.RBM_list.append(RBM(config[i], config[i+1]))
    
    def train_DBN(self, X, epsilon, batch_size, nb_epochs):
        for i in range(len(self.RBM_list)):
            self.RBM_list[i].train_RBM(X, epsilon, batch_size, nb_epochs)
            X = self.RBM_list[i].entree_sortie_RBM(X)

    def generer_image_DBN(self, nb_data, nb_gibbs):
        for i in range(nb_data):
            v = self.RBM_list[-1].generer_image_RBM_without_plot(nb_gibbs)

            for i in reversed(range(len(self.RBM_list)-1)):
                v = self.RBM_list[i].sortie_entree_RBM(v)

            v = np.reshape(v, (20, 16))
            plt.imshow(v, cmap="gray")
            plt.show()

        
    


        