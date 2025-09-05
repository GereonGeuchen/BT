#!/bin/bash

# Path to your unpacked conda-pack environment
ENV_PATH="$HOME/general-env"

# Project working directory (adjust as needed)
WORKDIR="$HOME/Dokumente/BT/fid_specific_switching/scripts"

# Python script name
PY_SCRIPT="switching_predictors_optimization.py"

# Ensure logs directory exists
mkdir -p "$WORKDIR/logs"

# Outer loop: MODE in all, late
for MODE in all late; do

  if [ "$MODE" = "all" ]; then
    # Generate budgets: [8*i for i in 1..12] + [50*i for i in 2..19]
    BUDGETS=()
    for i in $(seq 1 12); do
      BUDGETS+=($((8 * i)))
    done
    for i in $(seq 2 19); do
      BUDGETS+=($((50 * i)))
    done

  elif [ "$MODE" = "late" ]; then
    # Generate budgets: [50*i for i in 1..19]
    BUDGETS=()
    for i in $(seq 1 19); do
      BUDGETS+=($((50 * i)))
    done

  fi

  # Loop over generated budgets for this mode
  for BUDGET in "${BUDGETS[@]}"; do

    sbatch <<EOF
#!/bin/bash
#SBATCH -A thes2015
#SBATCH --job-name=${MODE}_B${BUDGET}
#SBATCH --output=${WORKDIR}/logs/${MODE}_B${BUDGET}.out
#SBATCH --error=${WORKDIR}/logs/${MODE}_B${BUDGET}.err
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Go to the working directory
cd $WORKDIR

# Activate the conda-pack env
source $ENV_PATH/bin/activate

# Run your tuning script with MODE and BUDGET
python $PY_SCRIPT --mode $MODE --budget $BUDGET

EOF

  done

done
