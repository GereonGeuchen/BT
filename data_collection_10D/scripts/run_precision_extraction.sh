#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory (adjust as needed) ===
WORKDIR="/home/thes2015/BT/data_collection_10D/scripts"

# === Python script name ===
PY_SCRIPT="postprocessing.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"


sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=precision_extraction
#SBATCH --output=${WORKDIR}/logs/precision_extraction.out
#SBATCH --error=${WORKDIR}/logs/precision_extraction.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT 
EOF

