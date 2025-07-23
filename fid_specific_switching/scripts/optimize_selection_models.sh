#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory ===
WORKDIR="$HOME/Dokumente/BT/fid_specific_switching/scripts"

# === Python script name ===
PY_SCRIPT="switch_model_optimization.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=switch_model_optimization
#SBATCH --output=${WORKDIR}/logs/switch_model_optimization.out
#SBATCH --error=${WORKDIR}/logs/switch_model_optimization.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT 

EOF
