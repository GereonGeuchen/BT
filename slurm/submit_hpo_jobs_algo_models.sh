#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project directory (where your Python code and data are)
WORKDIR="$HOME/Dokumente/BT/optimization/scripts"  

# Path to your Python script
PY_SCRIPT="optimization_slurm.py"

# Create logs directory
mkdir -p "$WORKDIR/logs"

# First sequence: 8 * [1..12]
for i in $(seq 1 12); do
  BUDGETS+=($((8 * i)))
done

# # Second sequence: 50 * [1..20]
for i in $(seq 2 19); do
  BUDGETS+=($((50 * i)))
done

# Loop over both modes and all budgets
for MODE in performance; do
  for BUDGET in "${BUDGETS[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=${MODE}_B${BUDGET}_log_switching
#SBATCH --output=${WORKDIR}/logs/${MODE}_B${BUDGET}_log_switching.out
#SBATCH --error=${WORKDIR}/logs/${MODE}_B${BUDGET}_log_switching.err
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate your packed conda environment
source $ENV_PATH/bin/activate

# Run the Python tuning script
python $PY_SCRIPT $MODE $BUDGET
EOF
  done
done
