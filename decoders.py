import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualBlock,ResidualStack

class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, output_size=150):
        """
        For VQ-VAE or AE:
          - input should have shape [B, embedding_dim, H', W'] 
        For VAE:
          - same shape [B, latent_dim, H', W'] if we pick latent_dim=embedding_dim for channels.
        Upsamples back to 1 channel, output_size x output_size.
        """
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv_trans2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.output_size = output_size  # final spatial size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.residual_stack(x)
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        # Force final resolution
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)
        return x

