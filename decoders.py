import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualBlock,ResidualStack

class AEDecoder(nn.Module):
    def __init__(self, latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(AEDecoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_dim, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        x = F.interpolate(x, size=(150, 150), mode="bilinear", align_corners=False)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VAEDecoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_dim, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + epsilon * std  # Reparameterization trick

        z = F.relu(self.conv1(z))
        z = self.residual_stack(z)
        z = F.relu(self.conv_trans1(z))
        z = self.conv_trans2(z)
        z = F.interpolate(z, size=(150, 150), mode="bilinear", align_corners=False)
        return z

class VQVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VQVAE_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(latent_dim, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Placeholder for vector quantization
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        x = F.interpolate(x, size=(150, 150), mode="bilinear", align_corners=False)
        return x
