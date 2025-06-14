#!/bin/bash

#SBATCH --output=%j_training_trex_med.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH -p gpu3,gpu4
#SBATCH --job-name=training-trex-med

# Load necessary environment variables
source "/etc/slurm/local_job_dir.sh"

# Define output and data directories
OUTPUT_DIR="/home/fe/schmolenski/quanda/output/trex_med"
DATA_DIR="/home/fe/schmolenski/quanda/nanoGPT/data/trex"
mkdir -p "$OUTPUT_DIR"

# Install the datasets library inside the container
echo "Installing datasets library..."
srun apptainer exec --nv \
    --bind /home/fe/schmolenski/quanda:/opt/quanda \
    ./apptainer.sif pip install datasets

# Check if the combined train and val bin files exist, if not, run data preparation
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
    echo "Combined train and validation bin files not found. Preparing data..."
    srun apptainer exec --nv \
        --bind /home/fe/schmolenski/quanda:/opt/quanda \
        ./apptainer.sif python /opt/quanda/nanoGPT/data/trex/prepare.py
else
    echo "Data files already exist. Skipping data preparation."
fi

# Run the training script using Apptainer
srun apptainer exec --nv \
    --bind /home/fe/schmolenski/quanda:/opt/quanda \
    --bind "$OUTPUT_DIR":/opt/quanda/output/trex_med \
    ./apptainer.sif python /opt/quanda/nanoGPT/train_trex_med.py --out_dir=/opt/quanda/nanoGPT/output/trex_med

