import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datasets import RGZ108k
from encoder import Encoder
from decoder import AEDecoder
from base_models import BaseAutoencoder
import plotting_functions

# Configuration
config = {
    "architecture": "AE",
    "batch_size": 4,
    "image_size": 150,
    "num_training_updates": 10000,
    "learning_rate": 2e-4,
    "latent_dim": 512,  # Compressed latent size
    "num_hiddens": 128,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "dataset_path": "/share/nas2_3/adey/data/galaxy_zoo/",
    "wandb_dir": "/share/nas2_3/adey/astro/wandb/",
    "save_dir": "/share/nas2_3/adey/astro/galaxy_out/",
    "save_latents_path": "/share/nas2_3/adey/astro/latents/"
}

wandb.init(config=config, project="comparing_models", entity="deya-03-the-university-of-manchester")

# Load Dataset
transform = torch.nn.Sequential(
    torch.nn.Upsample((config["image_size"], config["image_size"])),
    torch.nn.Conv2d(1, 1, 1),
    torch.nn.ReLU()
)

train_dataset = RGZ108k(root=config["dataset_path"], train=True, transform=transform)
valid_dataset = RGZ108k(root=config["dataset_path"], train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

# Initialize Model
encoder = Encoder(config["num_hiddens"], config["num_residual_layers"], config["num_residual_hiddens"], config["latent_dim"], variant="AE")
decoder = AEDecoder(config["latent_dim"], config["num_hiddens"], config["num_residual_layers"], config["num_residual_hiddens"])

model = BaseAutoencoder(encoder, decoder_type="AE", save_latents_path=config["save_latents_path"]).to("cuda")

# Optimizer and Scheduler
optimizer = Adam(model.parameters(), lr=config["learning_rate"])
scheduler = CosineAnnealingLR(optimizer, T_max=config["num_training_updates"], eta_min=1e-6)

# Training Loop
model.train()
global_step = 0

for epoch in range(10):
    for batch_idx, (images, _) in enumerate(train_loader):
        if global_step >= config["num_training_updates"]:
            break

        images = images.to("cuda")
        optimizer.zero_grad()
        reconstructed = model(images, save_latents=True, batch_idx=batch_idx)
        loss = F.mse_loss(reconstructed, images, reduction="sum")

        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({"train/loss": loss.item()}, step=global_step)

        if global_step % 100 == 0:
            print(f"Step {global_step}, Loss: {loss.item()}")

        # Validation every 1000 iterations
        if global_step % 1000 == 0:
            model.eval()
            val_loss = []
            with torch.no_grad():
                for val_images, _ in valid_loader:
                    val_images = val_images.to("cuda")
                    val_recon = model(val_images)
                    val_loss.append(F.mse_loss(val_recon, val_images, reduction="sum").item())

            avg_val_loss = np.mean(val_loss)
            wandb.log({"val/loss": avg_val_loss}, step=global_step)
            print(f"Validation Loss at step {global_step}: {avg_val_loss}")

            model.train()

        global_step += 1

# Save Model
torch.save(model.state_dict(), os.path.join(config["save_dir"], "ae_model.pth"))
print("Training complete. Model saved.")

with torch.no_grad():
    images, _ = next(iter(valid_loader))  # Get a batch from validation set
    images = images.to("cuda")

    if config["architecture"] == "VAE":
        mean, logvar = model.encoder(images)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std  # Reparameterization Trick
        reconstructions = model.decoder(z)

    elif config["architecture"] == "VQ-VAE":
        z = model.encoder(images)
        quantized, _, _ = vq_layer(z)  # Vector Quantization
        reconstructions = model.decoder(quantized)

    else:  # AE
        z = model.encoder(images)
        reconstructions = model.decoder(z)

    # Plot and log images using provided plotting function
    plotting_functions.display_images(images, reconstructions, num_images=8, step=global_step)

print("Final reconstruction images logged to wandb.")
