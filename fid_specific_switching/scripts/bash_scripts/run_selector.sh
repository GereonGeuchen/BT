#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory ===
WORKDIR="$HOME/Dokumente/BT/fid_specific_switching/scripts"

# === Python script name ===
PY_SCRIPT="selector.py"


# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=selector_normalized_tuned
#SBATCH --output=${WORKDIR}/logs/selector_normalized_tuned.out
#SBATCH --error=${WORKDIR}/logs/selector_normalized_tuned.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run the selector
python $PY_SCRIPT

EOF
