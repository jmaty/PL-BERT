#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
EXP="map_tokens"
QUEUE="default"
HOURS=168
# QSUB ARGUMENTS
MEM=64gb
LSCRATCH=20gb
NCPUS=4
INTB=map_tokens.ipynb

if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_map.sh cfg [n_cpus]"
     exit 1
fi

CFG=$1

if [[ "$#" -gt 1 ]]; then
     NCPUS=$2
fi

# Select argument
SELECT="-l select=1:ncpus=$NCPUS:mem=$MEM:scratch_local=$LSCRATCH"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d_%H%M%S")

SINGULARITY=/storage/plzen4-ntis/projects/singularity/papermill_23.12-latest.sh

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
ONTB=outputs/notebooks/$(basename "$INTB" .ipynb)_"$EXP".$TIMESTEP.ipynb
OLOG=outputs/logs/$EXP.$TIMESTEP.log

# Run PBS script
qsub -N "$EXP" \
     -q $QUEUE \
     -j oe \
     -o $OLOG \
     $WALLTIME \
     $SELECT \
     -- $SINGULARITY "$INTB" "$CFG" "$ONTB"

echo "$EXP: $QUEUE$CLUSTER, CPUs: $NCPUS, HOURS: $HOURS"
