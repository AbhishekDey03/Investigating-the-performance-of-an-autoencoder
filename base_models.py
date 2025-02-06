import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualStack

class BaseAutoencoder(nn.Module):
    def __init__(self, encoder, decoder_type, **kwargs):
        """
        encoder: An instance of Encoder.
        decoder_type: A string ('AE', 'VAE', 'VQ-VAE') to determine which decoder to use.
        kwargs: Additional arguments for the decoder.
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = encoder

        if decoder_type == 'AE':
            self.decoder = AEDecoder(**kwargs)
        elif decoder_type == 'VAE':
            self.decoder = VAEDecoder(**kwargs)
        elif decoder_type == 'VQ-VAE':
            self.decoder = VQVAE_Decoder(**kwargs)
        else:
            raise ValueError("Invalid decoder type. Choose from ['AE', 'VAE', 'VQ-VAE']")

    def forward(self, x):
        latent_representation = self.encoder(x)
        return self.decoder(latent_representation)
