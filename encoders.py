import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualBlock,ResidualStack
class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim, variant='VAE'):
        """
        variant: 'AE', 'VAE', or 'VQ-VAE' - determines how to process latent space
        """
        super(Encoder, self).__init__()
        self.variant = variant

        self.conv1 = nn.Conv2d(1, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

        if variant == 'VAE':
            self.pre_latent_conv = nn.Conv2d(num_hiddens, 2 * latent_dim, kernel_size=1, stride=1)
        else:
            self.pre_latent_conv = nn.Conv2d(num_hiddens, latent_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.residual_stack(x)
        x = self.pre_latent_conv(x)

        if self.variant == 'VAE':
            mean, logvar = torch.chunk(x, 2, dim=1)
            return mean, logvar
        else:
            return x
 