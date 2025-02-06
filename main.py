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
import wandb

from decoders import Decoder
from encoders import Encoder
from vectorquantizer import VectorQuantizer
from unified_model import UnifiedModel

config = {
    "architecture": "VQ-VAE",          # Choose: 'AE', 'VAE', or 'VQ-VAE'
    "batch_size": 4,
    "image_size": 150,
    "num_training_updates": 20000,
    "learning_rate": 2e-4,
    "commitment_cost": 0.25,          # For VQ-VAE
    "num_embeddings": 256,            # For VQ-VAE
    "embedding_dim": 64,             # For VQ-VAE or AE dimension
    "latent_dim": 64,                # For VAE (channels of mean, logvar each = 64)
    "num_hiddens": 128,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "dataset_path": "/share/nas2_3/adey/data/galaxy_zoo/",
    "num_workers": 1,
    "wandb_project": "comparing_models",
    "wandb_entity": "your-wandb-entity-or-username"
}


def main():
    # Initialize W&B
    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset & dataloaders
    from datasets import RGZ108k  # Your RGZ108k dataset class
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # These normalization stats come from your code, but adjust as needed
        transforms.Normalize((0.0031,), (0.0352,))
    ])

    train_dataset = RGZ108k(root=config["dataset_path"], train=True, transform=transform)
    valid_dataset = RGZ108k(root=config["dataset_path"], train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    # Initialize model
    model = UnifiedModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_training_updates"], eta_min=1e-6)

    # Training loop
    global_step = 0
    model.train()
    print(f"Starting training with architecture={config['architecture']}")

    while global_step < config["num_training_updates"]:
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            if config["architecture"] == "AE":
                # AE forward pass: recon = model(images)
                recon = model(images)
                recon_loss = F.mse_loss(recon, images, reduction='sum')
                total_loss = recon_loss
                bits_per_dim = recon_loss / (images.size(0) * config["image_size"]**2 * np.log(2))

                total_loss.backward()
                optimizer.step()

                # Logging
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/bits_per_dim": bits_per_dim.item()
                }, step=global_step)

            elif config["architecture"] == "VQ-VAE":
                # VQ-VAE forward pass: recon, vq_loss, perplexity = model(images)
                recon, vq_loss, perplexity = model(images)
                # sum MSE
                recon_loss = F.mse_loss(recon, images, reduction='sum')
                total_loss = recon_loss + vq_loss
                bits_per_dim = recon_loss / (images.size(0) * config["image_size"]**2 * np.log(2))

                total_loss.backward()
                optimizer.step()

                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/vq_loss": vq_loss.item(),
                    "train/perplexity": perplexity.item(),
                    "train/bits_per_dim": bits_per_dim.item()
                }, step=global_step)

            else:  # VAE
                # VAE forward pass: recon, mean, logvar = model(images)
                recon, mean, logvar = model(images)

                # --- Reconstruction Loss: Multivariate Gaussian NLL as in your code ---
                # Flatten for each sample
                B = images.size(0)
                x_recon_flat = recon.view(B, -1)
                x_flat = images.view(B, -1)

                # Identity covariance
                scale = torch.ones_like(x_recon_flat)
                scale_tril = torch.diag_embed(scale)
                
                mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
                recon_loss = -mvn.log_prob(x_flat).sum()  # sum over the batch

                # KL Divergence
                q_z_x = Normal(mean, torch.exp(0.5 * logvar))
                p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
                kl_div_value = kl_divergence(q_z_x, p_z).sum() / B

                total_loss = recon_loss + kl_div_value
                total_loss.backward()
                optimizer.step()

                # Bits per dimension
                bits_per_dim = recon_loss / (B * config["image_size"]**2 * np.log(2))

                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/kl_div": kl_div_value.item(),
                    "train/bits_per_dim": bits_per_dim.item()
                }, step=global_step)

            scheduler.step()
            global_step += 1

            # Simple stopping condition
            if global_step >= config["num_training_updates"]:
                break

    # ==============================
    # Validation (optional example)
    # ==============================
    model.eval()
    with torch.no_grad():
        val_losses = []
        for images, _ in valid_loader:
            images = images.to(device)
            if config["architecture"] == "AE":
                recon = model(images)
                recon_loss = F.mse_loss(recon, images, reduction='sum')
                val_losses.append(recon_loss.item())
            elif config["architecture"] == "VQ-VAE":
                recon, vq_loss, _ = model(images)
                recon_loss = F.mse_loss(recon, images, reduction='sum')
                total_loss = recon_loss + vq_loss
                val_losses.append(total_loss.item())
            else:  # VAE
                recon, mean, logvar = model(images)
                B = images.size(0)
                x_recon_flat = recon.view(B, -1)
                x_flat = images.view(B, -1)
                scale = torch.ones_like(x_recon_flat)
                scale_tril = torch.diag_embed(scale)
                mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
                recon_loss = -mvn.log_prob(x_flat).sum()
                q_z_x = Normal(mean, torch.exp(0.5 * logvar))
                p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
                kl_div_value = kl_divergence(q_z_x, p_z).sum() / B
                total_loss = recon_loss + kl_div_value
                val_losses.append(total_loss.item())

        avg_val_loss = np.mean(val_losses)
        wandb.log({"val/loss": avg_val_loss}, step=global_step)
        print(f"Final Validation Loss: {avg_val_loss:.3f}")

    # ==============================
    # Save model
    # ==============================
    save_dir = "/share/nas2_3/adey/astro/galaxy_out"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['architecture']}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
