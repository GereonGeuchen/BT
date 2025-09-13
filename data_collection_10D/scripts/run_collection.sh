#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory (adjust as needed) ===
WORKDIR="/home/thes2015/BT/data_collection_10D/scripts"

# === Python script name ===
PY_SCRIPT="collect_data.py"

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

# === Budgets: [8*i for i in 1..12] and [50*i for i in 1..20] ===

# # First sequence: 8 * [1..12]
# for i in $(seq 1 9); do
#   BUDGETS+=($((20 * i)))
# done

# # Second sequence: 50 * [1..20]
# for i in $(seq 2 20); do
#   BUDGETS+=($((100 * i)))
# done
# BUDGETS=(10 30 50 70 90 110 130 150 170 190)
BUDGETS=(20)


# # === Loop over budgets and submit jobs ===
for BUDGET in "${BUDGETS[@]}"; do

  sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=ela_B${BUDGET}
#SBATCH --output=${WORKDIR}/logs/ela_B${BUDGET}.out
#SBATCH --error=${WORKDIR}/logs/ela_B${BUDGET}.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your ELA calculation script with BUDGET
python $PY_SCRIPT --budget $BUDGET

EOF

done
