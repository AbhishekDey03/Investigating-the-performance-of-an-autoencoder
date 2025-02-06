
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.distributions import Normal, kl_divergence, MultivariateNormal
from torchvision import transforms

from decoders import Decoder
from encoders import Encoder
from vectorquantizer import VectorQuantizer

class UnifiedModel(nn.Module):
    def __init__(self, config):
        super(UnifiedModel, self).__init__()
        self.arch = config["architecture"]       # 'AE', 'VAE', or 'VQ-VAE'
        self.num_hiddens = config["num_hiddens"]
        self.num_residual_layers = config["num_residual_layers"]
        self.num_residual_hiddens = config["num_residual_hiddens"]
        self.embedding_dim = config["embedding_dim"]
        self.latent_dim = config["latent_dim"]   # for VAE
        self.commitment_cost = config["commitment_cost"]
        self.num_embeddings = config["num_embeddings"]

        # Shared Encoder
        self.encoder = Encoder(
            self.num_hiddens,
            self.num_residual_layers,
            self.num_residual_hiddens
        )

        # Based on architecture, define "pre-latent" conv
        if self.arch == "VQ-VAE":
            # We reduce num_hiddens -> embedding_dim
            self.pre_conv = nn.Conv2d(
                in_channels=self.num_hiddens, 
                out_channels=self.embedding_dim, 
                kernel_size=1, 
                stride=1
            )
            self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.commitment_cost)

        elif self.arch == "VAE":
            # We produce 2*latent_dim channels => [mean, logvar]
            self.pre_conv = nn.Conv2d(
                in_channels=self.num_hiddens,
                out_channels=2 * self.latent_dim,
                kernel_size=1,
                stride=1
            )
            # No separate VQ needed for VAE

        elif self.arch == "AE":
            # We produce "embedding_dim" channels for a spatial AE
            self.pre_conv = nn.Conv2d(
                in_channels=self.num_hiddens,
                out_channels=self.embedding_dim,
                kernel_size=1,
                stride=1
            )

        # Shared Decoder 
        # For VAE: we interpret "embedding_dim" = "latent_dim" for channels
        # so if architecture=VAE, we pass latent_dim to decoder as embedding_dim
        final_embedding_dim = self.latent_dim if self.arch == "VAE" else self.embedding_dim
        self.decoder = Decoder(
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
            embedding_dim=final_embedding_dim,
            output_size=config["image_size"]
        )

    def forward(self, x):
        """
        Forward returns different outputs depending on the architecture.
        
        - AE: returns (recon)
        - VQ-VAE: returns (recon, vq_loss, perplexity)
        - VAE: returns (recon, mean, logvar)
        """
        h = self.encoder(x)

        if self.arch == "VQ-VAE":
            h = self.pre_conv(h)
            quantized, vq_loss, perplexity = self.vq(h)
            recon = self.decoder(quantized)
            return recon, vq_loss, perplexity

        elif self.arch == "VAE":
            # produce [B, 2*latent_dim, H', W']
            h = self.pre_conv(h)
            B, C, H, W = h.shape
            # split into mean, logvar
            mean = h[:, :self.latent_dim, :, :]
            logvar = h[:, self.latent_dim:, :, :]

            # reparameterize
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std

            recon = self.decoder(z)
            return recon, mean, logvar

        else:  # "AE"
            h = self.pre_conv(h)
            recon = self.decoder(h)
            return recon
