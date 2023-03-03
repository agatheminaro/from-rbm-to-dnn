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
            self.RBM_list.append(RBM(config[i], config[i+1]))
    
    
    def entree_sortie(self, X):
        for i in range(len(self.config) - 1):
            X = self.RBM_list[i].entree_sortie_RBM(X)
        return X

    def sortie_entree(self, H):
        for i in range(len(self.config) - 1):
            H = self.RBM_list[-i-1].sortie_entree_RBM(H)
        return H
    
    def train_DBN(self, X, epsilon, batch_size, nb_epochs):
        self.epoch_dbn = []
        self.X_rec_list = []
        self.X_list = []
        self.errors = []
            
        for i in range(len(self.RBM_list)):
            self.RBM_list[i].train_RBM(X, epsilon, batch_size, nb_epochs)
            X = self.RBM_list[i].entree_sortie_RBM(X)
        
            X_rec = self.RBM_list[i].sortie_entree_RBM(X)
            
            rand_idx = np.random.randint(X.shape[0])
            if (X[rand_idx].shape[0]==320):
                self.epoch_dbn.append(f'Epoch{i}/{len(self.RBM_list)}')
                self.errors.append(np.sum((X[rand_idx] - X_rec[rand_idx]) ** 2) / X[rand_idx].shape[0])
                self.X_rec_list.append(X_rec[rand_idx])
                self.X_list.append(X[rand_idx])
        
    def generer_image_DBN(self, nb_data, nb_gibbs):
        for i in range(nb_data):
            v = self.RBM_list[-1].generer_image_RBM_without_plot(nb_gibbs)

            for i in reversed(range(len(self.RBM_list)-1)):
                v = self.RBM_list[i].sortie_entree_RBM(v)

            v = np.reshape(v, (20, 16))
            plt.imshow(v, cmap="gray")
            plt.show()
            
            
    def display_image_DBN_vs_original(self):
        
        for i in range(len(self.errors)):
            print(self.epoch_rbm[i])
            fig, ax = plt.subplots(1, 2, figsize=(3, 2))
            ax[0].imshow(self.X_list[i].reshape(20,16), cmap='gray')
            ax[0].axis('off')
            ax[0].set_title('Original_image', fontsize=7)
            
            ax[1].imshow(self.X_rec_list[i].reshape(20,16), cmap='gray')
            ax[1].axis('off')
            ax[1].set_title(f'RBM_image : RMSE = {self.errors[i]}', fontsize=7)
            
            fig.tight_layout()
            plt.show()

        
    


        