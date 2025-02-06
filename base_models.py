import torch
import torch.nn as nn
import numpy as np
import os
from residual import ResidualStack

class BaseAutoencoder(nn.Module):
    def __init__(self, encoder, decoder_type, save_latents_path=None, **kwargs):
        """
        encoder: An instance of Encoder.
        decoder_type: 'AE', 'VAE', or 'VQ-VAE'.
        save_latents_path: Path to save latent representations as .npy files.
        kwargs: Additional arguments for the decoder.
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = encoder
        self.save_latents_path = save_latents_path  # Directory for latent storage

        if decoder_type == 'AE':
            self.decoder = AEDecoder(**kwargs)
        elif decoder_type == 'VAE':
            self.decoder = VAEDecoder(**kwargs)
        elif decoder_type == 'VQ-VAE':
            self.decoder = VQVAE_Decoder(**kwargs)
        else:
            raise ValueError("Invalid decoder type. Choose from ['AE', 'VAE', 'VQ-VAE']")

    def forward(self, x, save_latents=False, batch_idx=None):
        """ 
        Forward pass through the encoder and decoder. Optionally saves latents. 
        """
        latent = self.encoder(x)

        if self.save_latents_path and save_latents and batch_idx is not None:
            self.save_latents(latent, batch_idx)

        return self.decoder(latent)

    def save_latents(self, latent, batch_idx):
        """ Saves the latent representation as a .npy file. """
        save_path = os.path.join(self.save_latents_path, f"latents_batch_{batch_idx}.npy")
        np.save(save_path, latent.cpu().detach().numpy())
        print(f"Saved latents to {save_path}")
