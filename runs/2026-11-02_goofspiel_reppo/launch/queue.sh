#!/bin/bash
# ReMAPPO on nash: runs the jobs of a jobs file (lines
# "name | script | hydra overrides", script in bluffjax/examples/goofspiel/)
# one after another under the shared GPU lock, skipping those already done.
# Usage: setsid nohup bash launch/queue.sh launch/stage1.txt > /dev/null 2>&1 < /dev/null &
JOBS=$(realpath "$1")  # before the cd below
CODE=~/bluffjax-dev/reppo
RUN=$CODE/runs/2026-11-02_goofspiel_reppo
PY=~/bluffjax-dev/cfr-push/.venv/bin/python  # jax 0.7.2 + CUDA 12
cd "$CODE" || exit 1
mkdir -p "$RUN/logs" "$RUN/jobs"
export PYTHONPATH=$CODE XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
STATUS=$RUN/logs/status.txt
while IFS='|' read -r name script args; do
  name=$(echo "$name" | xargs); script=$(echo "$script" | xargs); args=$(echo "$args" | sed 's/^ *//')
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  grep -q "^$name done" "$STATUS" 2>/dev/null && continue
  echo "$name start $(date +%F_%T)" >> "$STATUS"
  eval flock ~/bluffjax-dev/gpu.lock $PY bluffjax/examples/goofspiel/$script \
    job_type=$name save_dir=$RUN/jobs/$name hydra.run.dir=$RUN/hydra/$name $args \
    > "$RUN/logs/$name.log" 2>&1 < /dev/null
  rc=$?
  if [ $rc = 0 ]; then echo "$name done $(date +%F_%T)" >> "$STATUS"
  else echo "$name FAILED rc=$rc $(date +%F_%T)" >> "$STATUS"; fi
done < "$JOBS"
echo "QUEUE_FINISHED $JOBS $(date +%F_%T)" >> "$STATUS"
