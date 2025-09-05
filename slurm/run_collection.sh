#!/bin/bash
#SBATCH --job-name=cma-switching
#SBATCH --output=/home/user/projects/cma_switching/logs/%x_%j.out
#SBATCH --error=/home/user/projects/cma_switching/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Optional: load Python environment
# module load python/3.9
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# Move to project directory
cd /home/pf657206/Dokumente/BT/data_collection/scripts

# Ensure logs directory exists
mkdir -p logs

# Run the script
python3 collect_data.py
