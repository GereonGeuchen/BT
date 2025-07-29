#!/bin/bash

# === Path to your unpacked conda-pack environment ===
ENV_PATH="$HOME/general-env"

# === Project working directory ===
WORKDIR="$HOME/Dokumente/BT/fid_specific_switching/scripts"

# === Python script name ===
PY_SCRIPT="tune_decision_thresholds.py"

# === Algorithm to tune thresholds for ===
algorithm="l_BFGS_b"

# === Whether to use normalized data ===
normalized=false

# === Construct job name ===
jobname="threshold_tuning_${algorithm}"
if [ "$normalized" = true ]; then
  jobname="${jobname}_normalized"
fi

# === Ensure logs directory exists ===
mkdir -p "$WORKDIR/logs"

# === Submit the job ===
sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=${jobname}
#SBATCH --output=${WORKDIR}/logs/${jobname}.out
#SBATCH --error=${WORKDIR}/logs/${jobname}.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack environment
source $ENV_PATH/bin/activate

# Run the Python script
python $PY_SCRIPT "$algorithm" "$normalized"
EOF
