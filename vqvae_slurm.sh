#!/bin/bash

#SBATCH --job-name=vqvae_training
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2_3/adey/w_c_21_Oct/test_output/.out/vqvae_train_%j.log
#SBATCH --error=/share/nas2_3/adey/w_c_21_Oct/test_output/.err/vqvae_train_%j.err
#SBATCH --mem=1500GB

# Display GPU status
nvidia-smi

# Set up WandB environment variables
export WANDB_API_KEY="ae52e426af96ba6657d73e9829e28ac8891914d6"
export WANDB_PROJECT="comparing_models"
export WANDB_ENTITY="deya-03-the-university-of-manchester"

export WANDB_CACHE_DIR="/share/nas2_3/adey/wandb_cache"
export WANDB_DATA_DIR="/share/nas2_3/adey/astro/wandb_data/"

echo ">> Starting setup"
ulimit -n 65536

# Activate the virtual environment
source /share/nas2_3/adey/.venv/bin/activate
echo ">> Environment activated"

# Verify Python version
python --version
/share/nas2_3/adey/.venv/bin/python --version

# Define paths
PYTHON_SCRIPT="/share/nas2_3/adey/astro/clean_code/train_vqvae.py"
SWEEP_CONFIG="/share/nas2_3/adey/astro/clean_code/config_vqvae_wandb.yaml"

# Initialize wandb sweep
temp_file=$(mktemp)
wandb sweep "$SWEEP_CONFIG" > "$temp_file" 2>&1

# Extract the sweep ID
SWEEP_PATH=$(grep -o 'deya-03-the-university-of-manchester/comparing_models/[^ ]*' "$temp_file" | tail -n 1)

rm "$temp_file"

if [[ -z "$SWEEP_PATH" ]]; then
  echo "Error: Sweep ID not found. Please check your configuration."
  exit 1
fi

# Start the sweep agent
echo ">> Starting sweep agent for: $SWEEP_PATH"
wandb agent "$SWEEP_PATH"
