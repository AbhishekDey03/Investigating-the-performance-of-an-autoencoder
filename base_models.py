import torch
import torch.nn as nn
import numpy as np
import os
from residual import ResidualStack
from decoders import AEDecoder, VAEDecoder, VQVAE_Decoder
from vectorquantizer import VectorQuantizer

class BaseAutoencoder(nn.Module):
    def __init__(self, encoder, decoder_type, save_latents_path=None, num_embeddings=None, embedding_dim=None, **kwargs):
        """
        encoder: An instance of Encoder.
        decoder_type: 'AE', 'VAE', or 'VQ-VAE'.
        save_latents_path: Path to save latent representations as .npy files.
        num_embeddings: Number of embeddings for VectorQuantizer (required for VQ-VAE).
        embedding_dim: Dimensionality of embedding space (required for VQ-VAE).
        kwargs: Additional arguments for the decoder.
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = encoder
        self.save_latents_path = save_latents_path  # Directory for latent storage
        self.decoder_type = decoder_type
        
        if decoder_type == 'AE':
            self.decoder = AEDecoder(**kwargs)
            self.vq_layer = None
        elif decoder_type == 'VAE':
            self.decoder = VAEDecoder(**kwargs)
            self.vq_layer = None
        elif decoder_type == 'VQ-VAE':
            if num_embeddings is None or embedding_dim is None:
                raise ValueError("num_embeddings and embedding_dim must be specified for VQ-VAE")
            self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
            self.decoder = VQVAE_Decoder(**kwargs)
        else:
            raise ValueError("Invalid decoder type. Choose from ['AE', 'VAE', 'VQ-VAE']")

    def forward(self, x, save_latents=False, batch_idx=None):
        """ Forward pass through the encoder, quantizer (if VQ-VAE), and decoder. Optionally saves latents. """
        latent = self.encoder(x)
        
        if self.decoder_type == 'VQ-VAE':
            quantized_latent, vq_loss = self.vq_layer(latent)  # Apply vector quantization
            decoded = self.decoder(quantized_latent)
        else:
            vq_loss = 0  # No VQ loss for AE and VAE
            decoded = self.decoder(latent)
        
        if self.save_latents_path and save_latents and batch_idx is not None:
            self.save_latents(latent, batch_idx)
        
        return decoded, vq_loss

    def save_latents(self, latent, batch_idx):
        """ Saves the latent representation as a .npy file. """
        save_path = os.path.join(self.save_latents_path, f"latents_batch_{batch_idx}.npy")
        np.save(save_path, latent.cpu().detach().numpy())
        print(f"Saved latents to {save_path}")
