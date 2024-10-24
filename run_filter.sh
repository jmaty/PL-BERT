#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
EXP="filter_para"
QUEUE="default"
HOURS=24
# QSUB ARGUMENTS
MEM=64gb
LSCRATCH=20gb

if [[ "$#" -lt 2 ]]; then
     echo "Usage: run_filter.sh cfg notebook [hours]"
     exit 1
fi

CFG=$1
INTB=$2

# Select argument
SELECT="-l select=1:ncpus=2:mem=$MEM:scratch_local=$LSCRATCH"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Extract name of the experiment
# EXP=$(grep 'log_dir:' "$CFG" | sed -r 's/log_dir: "models\/(.+)"/\1/')

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d_%H%M%S")

SINGULARITY=/storage/plzen4-ntis/projects/singularity/papermill_23.12-latest.sh

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
ONTB=outputs/notebooks/$(basename "$INTB" .ipynb)_"$EXP".$TIMESTEP.ipynb

# Run PBS script
qsub -N "$EXP" \
     -q $QUEUE \
     -j oe \
     -o outputs/logs/$EXP.$TIMESTEP.log \
     $WALLTIME \
     $SELECT \
     -- $SINGULARITY "$INTB" "$CFG" "$ONTB"

echo "$EXP: $QUEUE$CLUSTER, HOURS: $HOURS"
