#!/bin/bash
# Submits every job of a jobs file (lines "name | script | hydra overrides")
# as its own SLURM job, skipping names in the skip list or already done on
# the cluster. Run on the cluster from ~/bluffjax-dev/reppo.
# Usage: bash runs/2026-11-03_goofspiel_matched/launch/submit_cluster.sh <jobs file> <time> [skip names...]
JOBS=$1; TIME=$2; shift 2; SKIP=" $* "
RUN=runs/2026-11-03_goofspiel_matched
mkdir -p "$RUN/logs"
while IFS='|' read -r name script args; do
  name=$(echo "$name" | xargs); script=$(echo "$script" | xargs); args=$(echo "$args" | sed 's/^ *//')
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  [[ "$SKIP" == *" $name "* ]] && { echo "skip $name"; continue; }
  grep -q "^$name done" "$RUN/logs/status_cluster.txt" 2>/dev/null && { echo "done $name"; continue; }
  eval sbatch -J "$name" -t "$TIME" "$RUN/launch/run_cluster.sbatch" "$script" $args
done < "$JOBS"
