from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc_mu = nn.Linear(100, hidden_size)
        self.fc_var = nn.Linear(100, hidden_size)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.tanh(F.dropout(self.fc1(x), 0.3))
        x = F.tanh(self.fc2(x))
        mu  = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

class Reparametrize(nn.Module):
    def __init__(self):
        super(Reparametrize, self).__init__()

    def forward(self, mu, logvar):
        logstd = 0.5 * logvar
        std = torch.exp_(logstd)
        z = torch.randn_like(std, dtype=torch.float32) * std + mu
        return z

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 400)
        self.fc3 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = F.tanh(self.fc1(z))
        x = F.tanh(self.fc2(x))
        x = F.dropout(self.fc3(x), 0.3)
        output = self.sigmoid(x)
        output = output.view(-1, 1, 28, 28)
        return output

class VAE_mnist(BaseModel):
    def __init__(self, hidden_size):
        super(VAE_mnist, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.reparam = Reparametrize()
        self.decoder = Decoder(hidden_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # print(mu)
        # print(logvar)
        z = self.reparam(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar