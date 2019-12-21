import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class VAE(nn.Module):
    def __init__(self, length, n_latent):
        super(VAE, self).__init__()

        self.length = length
        self.n_latent = n_latent

        self.C1 = nn.Conv2d(self.length, 400, kernel_size=1)
        self.C2 = nn.Conv2d(400, 400, kernel_size=1)

        self.C31 = nn.Conv2d(400, self.n_latent, kernel_size=1)
        self.C32 = nn.Conv2d(400, self.n_latent, kernel_size=1)

        self.C4 = nn.Conv2d(self.n_latent, 400, kernel_size=1)
        self.C5 = nn.Conv2d(400, 400, kernel_size=1)
        self.C6 = nn.Conv2d(400, self.length, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def encode(self, x):
        out = self.C1(x)
        out = self.relu(out)
        out = self.C2(out)
        out = self.relu(out)

        mu = self.C31(out)
        logvar = self.C32(out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        out = self.C4(z)
        out = self.relu(out)
        out = self.C5(out)
        out = self.relu(out)
        out = self.C6(out)
        return out

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), z, mu, logvar

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
