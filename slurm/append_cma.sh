#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory ===
WORKDIR="$HOME/Dokumente/BT/data_collection/scripts"

# === Python script name ===
PY_SCRIPT="cma_appending_slurm.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

# === Budgets: [8*i for i in 1..12] and [50*i for i in 1..20] ===
# # First sequence: 8 * [1..12]
for i in $(seq 1 12); do
  BUDGETS+=($((8 * i)))
done

# # Second sequence: 50 * [1..20]
for i in $(seq 1 20); do
  BUDGETS+=($((50 * i)))
done
BUDGETS=(100)

# === Loop over budgets and submit each as a separate job ===
for BUDGET in "${BUDGETS[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=ela_B${BUDGET}
#SBATCH --output=${WORKDIR}/logs/ela_B${BUDGET}.out
#SBATCH --error=${WORKDIR}/logs/ela_B${BUDGET}.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT $BUDGET

EOF

done
