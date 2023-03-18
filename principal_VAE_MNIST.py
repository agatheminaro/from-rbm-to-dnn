import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, n_rows, n_cols, n_channels):
        super(VAE, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = (self.n_rows) * (self.n_cols)
        self.z_dim = z_dim

        # encoder part
        self.fc1 = torch.nn.Linear(in_features=x_dim, out_features=h_dim1)
        self.fc2 = torch.nn.Linear(in_features=h_dim1, out_features=h_dim2)
        self.fc3 = torch.nn.Linear(in_features=h_dim2, out_features=self.z_dim)
        self.fc32 = torch.nn.Linear(in_features=h_dim2, out_features=self.z_dim)
        # decoder part
        self.fc4 = torch.nn.Linear(in_features=z_dim, out_features=h_dim2)
        self.fc5 = torch.nn.Linear(in_features=h_dim2, out_features=h_dim1)
        self.fc6 = torch.nn.Linear(in_features=h_dim1, out_features=x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h), self.fc32(h)  # return mu, log_var

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)).view(
            -1, self.n_channels, self.n_rows, self.n_cols
        )

    def sampling(self, mu, log_var):
        # this function samples a Gaussian distribution, with average (mu) and standard deviation specified (using log_var)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        z_mu, z_log_var = self.encoder(x.view(-1, self.n_pixels))
        z = self.sampling(z_mu, z_log_var)
        return (
            self.decoder(z),
            z_mu,
            z_log_var,
        )

    def loss_function(self, x, y, mu, log_var):
        reconstruction_error = F.binary_cross_entropy(
            y.view(-1, self.n_pixels), x.view(-1, self.n_pixels), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_error + KLD


def train_vae(vae_model, vae_optimizer, data_train_loader, epoch, verbose=True):
    train_loss = 0
    for data in data_train_loader:
        vae_optimizer.zero_grad()
        data = data[0]
        y, z_mu, z_log_var = vae_model(data)
        loss_vae = vae_model.loss_function(data, y, z_mu, z_log_var)
        loss_vae.backward()
        train_loss += loss_vae.item()
        vae_optimizer.step()

    if verbose:
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(data_train_loader.dataset)
            )
        )


def test_vae(vae_model, data_test_loader, verbose=True):
    test_loss = 0
    with torch.no_grad():
        for _, (data, _) in enumerate(data_test_loader):
            y, z_mu, z_log_var = vae_model(data)
            test_loss += vae_model.loss_function(data, y, z_mu, z_log_var).item()

    test_loss /= len(data_test_loader.dataset)
    if verbose:
        print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def generate_images(vae_model, n_samples=10):
    with torch.no_grad():
        z = torch.randn(n_samples, 1, vae_model.z_dim)
        samples = vae_model.decoder(z)
        return samples


def pytorch_to_numpy(x):
    return x.detach().numpy()


def display_images(imgs):
    for i in range(imgs.shape[0]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imgs[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
