from principal_RBM_alpha import RBM
import numpy as np
import matplotlib.pyplot as plt

class DBN:
    def __init__(self, config):
        self.RBM_list = []
        for i in range(len(config) - 1):
            self.RBM_list.append(RBM(config[i], config[i+1]))
    
    def train_DBN(self, X, epsilon, batch_size, nb_epoch):
        for i in range(len(self.RBM_list)):
            self.RBM_list[i].train_RBM(X, epsilon, batch_size, nb_epoch)
            X = self.RBM_list[i].entree_sortie_RBM(X)

    def generer_image_DBN(self, nb_data, nb_gibbs):
        last_RBM = self.RBM_list[-1]
        v = last_RBM.entree_sortie(nb_data, nb_gibbs)
        for i in range(len(self.RBM_list), 0):

            pass

        for i in range(nb_data):
            v = (np.random.rand(self.p) < 1 / 2) * 1
            for _ in range(nb_gibbs):
                h = (np.random.rand(self.q) < self.entree_sortie_RBM(v)) * 1
                v = (np.random.rand(self.p) < self.sortie_entree_RBM(h)) * 1

        return last_RBM.generer_image_RBM(nb_data, nb_gibbs)

        
    


        