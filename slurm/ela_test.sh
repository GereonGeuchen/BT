#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory (adjust as needed) ===
WORKDIR="$HOME/Dokumente/BT/data_collection/scripts"

# === Python script name ===
PY_SCRIPT="pflacco_slurm.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

# === Test budget ===
BUDGET=50

# === Submit a single test job ===
sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=ela_B${BUDGET}
#SBATCH --output=${WORKDIR}/logs/ela_B${BUDGET}.out
#SBATCH --error=${WORKDIR}/logs/ela_B${BUDGET}.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT --budget $BUDGET

EOF
