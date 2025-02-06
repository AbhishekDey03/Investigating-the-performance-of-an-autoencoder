import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualBlock,ResidualStack


# Define Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # x shape: [B, embedding_dim, H, W]
        # 1) Flatten input to [B*H*W, embedding_dim]
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # 2) Compute distances (squared L2)
        distances = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(x_flat, self.embeddings.weight.t())
        )

        # 3) Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # shape [B*H*W, 1]
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 4) Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # 5) VQ Loss
        # e_latent_loss = ||sg[z_e(x)] - e||^2
        # q_latent_loss = ||z_e(x) - sg[e]||^2
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 6) Straight-through estimator
        quantized = x + (quantized - x).detach()

        # 7) Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity