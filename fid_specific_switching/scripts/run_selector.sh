#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/asf-env"

# === Project working directory ===
WORKDIR="$HOME/Dokumente/BT/fid_specific_switching/scripts"

# === Python script name ===
PY_SCRIPT="selector.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=selector_optimized_50trials
#SBATCH --output=${WORKDIR}/logs/selector_optimized_50trials.out
#SBATCH --error=${WORKDIR}/logs/selector_optimized_50trials.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT 

EOF
